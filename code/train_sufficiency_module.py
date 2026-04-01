#!/usr/bin/env python3
"""Train criterion-specific LSTM sample-sufficiency modules."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from models.sufficiency import CriterionSpecificLSTM
from sufficiency_utils import (
    LandscapeSufficiencyDataset,
    compute_normalization_stats,
    default_landscape_json_path,
    load_landscape_runs,
    make_samples,
    method_display_name,
    partition_samples,
    split_trajectories,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def roc_auc_score(labels: list[float], scores: list[float]) -> float:
    pos = sum(1 for x in labels if x >= 0.5)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if label >= 0.5:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def brier_score(labels: list[float], probs: list[float]) -> float:
    if not labels:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)


def evaluate(model, loader, classifier_weight: float, regression_weight: float, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    losses = []
    logits_all = []
    probs_all = []
    labels_all = []
    pred_k_all = []
    true_k_all = []
    current_k_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            label = batch["label"].to(device)
            log_k_star = batch["log_k_star"].to(device)
            target_k = batch["k_star"].to(device)

            outputs = model(x)
            cls_loss = bce(outputs["logits"], label)
            reg_loss = huber(outputs["log_kstar"], log_k_star)
            loss = classifier_weight * cls_loss + regression_weight * reg_loss
            losses.append(loss.item())

            probs = torch.sigmoid(outputs["logits"]).cpu()
            pred_k = torch.exp(outputs["log_kstar"]).cpu()
            logits_all.extend(outputs["logits"].cpu().tolist())
            probs_all.extend(probs.tolist())
            labels_all.extend(label.cpu().tolist())
            pred_k_all.extend(pred_k.tolist())
            true_k_all.extend(target_k.cpu().tolist())
            current_k_all.extend(batch["k"].cpu().tolist())

    mae_k = (
        sum(abs(pred - true) for pred, true in zip(pred_k_all, true_k_all)) / len(true_k_all)
        if true_k_all
        else float("nan")
    )
    return {
        "loss": sum(losses) / len(losses) if losses else float("nan"),
        "auroc": roc_auc_score(labels_all, logits_all),
        "brier": brier_score(labels_all, probs_all),
        "mae_k": mae_k,
        "num_samples": len(labels_all),
        "labels": labels_all,
        "probs": probs_all,
        "pred_k": pred_k_all,
        "true_k": true_k_all,
        "current_k": current_k_all,
    }


def train_one_method(conf, runs: list[dict], method: str, out_dir: Path, device) -> dict:
    samples = make_samples(conf, runs, method=method)
    if not samples:
        raise ValueError(f"No samples built for method {method}")

    splits = split_trajectories(conf, samples)
    partitioned = partition_samples(samples, splits)
    example_dim = len(samples[0]["steps"][0])
    feature_mean, feature_std = compute_normalization_stats(partitioned["train"], example_dim)

    def build_loader(split: str, shuffle: bool) -> DataLoader:
        dataset = LandscapeSufficiencyDataset(
            partitioned[split],
            feature_mean=feature_mean,
            feature_std=feature_std,
            sequence_window=int(conf.sufficiency.sequence_window),
        )
        return DataLoader(
            dataset,
            batch_size=int(conf.sufficiency.batch_size),
            shuffle=shuffle,
        )

    train_loader = build_loader("train", shuffle=True)
    val_loader = build_loader("val", shuffle=False)
    test_loader = build_loader("test", shuffle=False)

    model = CriterionSpecificLSTM(
        input_dim=example_dim,
        hidden_dim=int(conf.sufficiency.hidden_dim),
        num_layers=int(conf.sufficiency.num_layers),
        dropout=float(conf.sufficiency.dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(conf.sufficiency.learning_rate),
        weight_decay=float(conf.sufficiency.weight_decay),
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    classifier_weight = float(conf.sufficiency.classifier_weight)
    regression_weight = float(conf.sufficiency.regression_weight)
    best_state = None
    best_val = float("inf")
    patience = 0
    history = []

    for epoch in range(int(conf.sufficiency.max_epochs)):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            label = batch["label"].to(device)
            log_k_star = batch["log_k_star"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            cls_loss = bce(outputs["logits"], label)
            reg_loss = huber(outputs["log_kstar"], log_k_star)
            loss = classifier_weight * cls_loss + regression_weight * reg_loss
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_metrics = evaluate(
            model,
            val_loader,
            classifier_weight=classifier_weight,
            regression_weight=regression_weight,
            device=device,
        )
        train_loss = sum(train_losses) / len(train_losses)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auroc": val_metrics["auroc"],
                "val_mae_k": val_metrics["mae_k"],
            }
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {
                "model": model.state_dict(),
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "history": history,
                "method": method,
            }
            patience = 0
        else:
            patience += 1
            if patience >= int(conf.sufficiency.early_stopping_patience):
                break

    if best_state is None:
        raise RuntimeError(f"Training failed to produce a checkpoint for {method}")

    model.load_state_dict(best_state["model"])
    val_metrics = evaluate(
        model,
        val_loader,
        classifier_weight=classifier_weight,
        regression_weight=regression_weight,
        device=device,
    )
    test_metrics = evaluate(
        model,
        test_loader,
        classifier_weight=classifier_weight,
        regression_weight=regression_weight,
        device=device,
    )

    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_dir / "model.pt"
    torch.save(best_state, checkpoint_path)

    result = {
        "method": method,
        "display_name": method_display_name(method, int(conf.sufficiency.subspace_dim)),
        "checkpoint_path": str(checkpoint_path),
        "feature_dim": example_dim,
        "splits": {name: sorted(ids) for name, ids in splits.items()},
        "history": history,
        "val_metrics": {k: v for k, v in val_metrics.items() if k not in {"labels", "probs", "pred_k", "true_k"}},
        "test_metrics": {k: v for k, v in test_metrics.items() if k not in {"labels", "probs", "pred_k", "true_k"}},
        "test_predictions": [
            {
                "prob_sufficient": prob,
                "current_k": current_k,
                "predicted_k_star": pred_k,
                "true_k_star": true_k,
                "label": label,
            }
            for prob, current_k, pred_k, true_k, label in zip(
                test_metrics["probs"],
                test_metrics["current_k"],
                test_metrics["pred_k"],
                test_metrics["true_k"],
                test_metrics["labels"],
            )
        ],
    }
    with open(method_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main() -> None:
    conf = OmegaConf.load(CONFIG_PATH)
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_landscape_json_path(conf)
    if not json_path.is_file():
        raise SystemExit(f"Missing {json_path}; run python code/run_experiments.py first.")

    runs = load_landscape_runs(json_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / "code" / "output" / "sufficiency"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method in conf.sufficiency.methods:
        print(f"Training {method_display_name(method, int(conf.sufficiency.subspace_dim))}")
        results.append(train_one_method(conf, runs, method, out_dir=out_dir, device=device))

    summary = {
        "source_json": str(json_path),
        "methods": results,
    }
    with open(out_dir / "sufficiency_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'sufficiency_results.json'}")


if __name__ == "__main__":
    main()
