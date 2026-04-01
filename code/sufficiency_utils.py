"""Utilities for criterion-specific sample-sufficiency prediction."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent


def default_landscape_json_path(conf) -> Path:
    return REPO_ROOT / str(conf.common.output_dir) / "landscape_experiments.json"


def load_landscape_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def trajectory_key(row: dict) -> str:
    return f"{row.get('setting_name', 'default')}::seed{row['seed']}"


def safe_log(value: float, floor: float = 1e-12) -> float:
    return math.log(max(float(value), floor))


def method_display_name(method: str, subspace_dim: int) -> str:
    mapping = {
        "delta1": "LISSA-1",
        "delta2": "LISSA-2",
        "delta2_subspace": f"LISSA-D{int(subspace_dim)}",
        "delta2_subspace_plus_spectral": f"LISSA-D{int(subspace_dim)}+Spec",
    }
    return mapping.get(method, method)


def group_runs_by_trajectory(runs: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in runs:
        grouped[trajectory_key(row)].append(row)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda row: int(row["k"]))
    return grouped


def _metric_value_and_stderr(row: dict, method: str, subspace_dim: int) -> tuple[float, float]:
    if method == "delta1":
        stats = row.get("delta1_stats", {})
        return float(row["delta1"]), float(stats.get("stderr", 0.0))
    if method == "delta2":
        stats = row.get("delta2_stats", {})
        return float(row["delta2"]), float(stats.get("stderr", 0.0))
    if method in {"delta2_subspace", "delta2_subspace_plus_spectral"}:
        key = str(int(subspace_dim))
        value = float(row["delta2_subspace"][key])
        stats = row.get("delta2_subspace_stats", {}).get(key, {})
        return value, float(stats.get("stderr", 0.0))
    raise ValueError(f"Unknown sufficiency method: {method}")


def _local_slope(rows: list[dict], idx: int, method: str, subspace_dim: int) -> float:
    if idx == 0:
        return 0.0
    prev_row = rows[idx - 1]
    row = rows[idx]
    prev_value, _ = _metric_value_and_stderr(prev_row, method, subspace_dim)
    value, _ = _metric_value_and_stderr(row, method, subspace_dim)
    prev_log_k = safe_log(prev_row["k"])
    log_k = safe_log(row["k"])
    denom = log_k - prev_log_k
    if abs(denom) < 1e-12:
        return 0.0
    return (safe_log(value) - safe_log(prev_value)) / denom


def _spectral_features(row: dict, subspace_dim: int) -> list[float]:
    key = str(int(subspace_dim))
    eigenvalues = [abs(float(x)) for x in row.get("top_eigenvalues", [])]
    total_mass = sum(eigenvalues) or 1.0
    leading_mass = sum(eigenvalues[: int(subspace_dim)])
    drift = float(row.get("eigenspace_drift", {}).get(key, 0.0))
    quad = row.get("quadratic_alignment", {})
    return [
        drift,
        leading_mass / total_mass,
        safe_log(leading_mass),
        safe_log(float(row.get("compressed_hessian_gap_fro", 0.0))),
        safe_log(float(row.get("gradient_gap_norm", 0.0))),
        float(quad.get("correlation", 0.0)),
        safe_log(float(quad.get("relative_mean_error", 0.0) + 1.0)),
    ]


def step_features(rows: list[dict], idx: int, method: str, subspace_dim: int) -> list[float]:
    row = rows[idx]
    value, stderr = _metric_value_and_stderr(row, method, subspace_dim)
    feats = [
        safe_log(row["k"]),
        safe_log(value),
        safe_log(stderr + 1e-12),
        _local_slope(rows, idx, method, subspace_dim),
    ]
    if method in {"delta2_subspace", "delta2_subspace_plus_spectral"}:
        feats.append(safe_log(float(subspace_dim)))
    if method == "delta2_subspace_plus_spectral":
        feats.extend(_spectral_features(row, subspace_dim))
    return feats


def make_samples(conf, runs: list[dict], method: str) -> list[dict]:
    grouped = group_runs_by_trajectory(runs)
    window = int(conf.sufficiency.sequence_window)
    label_metric = str(conf.sufficiency.label_metric)
    epsilon = float(conf.sufficiency.label_epsilon)
    subspace_dim = int(conf.sufficiency.subspace_dim)
    samples = []

    for traj_id, rows in grouped.items():
        if any(row.get(label_metric) is None for row in rows):
            raise ValueError(
                f"Rows for {traj_id} are missing {label_metric}. "
                "Rerun landscape experiments with validation logging enabled."
            )
        target_values = [float(row[label_metric]) for row in rows]
        final_value = target_values[-1]
        suff_flags = [(value - final_value) <= epsilon for value in target_values]
        if not any(suff_flags):
            suff_flags[-1] = True
        k_star = int(next(row["k"] for row, is_sufficient in zip(rows, suff_flags) if is_sufficient))
        log_k_star = safe_log(k_star)

        for idx, row in enumerate(rows):
            start = max(0, idx - window + 1)
            steps = [
                step_features(rows, j, method=method, subspace_dim=subspace_dim)
                for j in range(start, idx + 1)
            ]
            samples.append(
                {
                    "trajectory_id": traj_id,
                    "setting_name": row.get("setting_name", "default"),
                    "seed": int(row["seed"]),
                    "k": int(row["k"]),
                    "steps": steps,
                    "label": float(suff_flags[idx]),
                    "k_star": k_star,
                    "log_k_star": log_k_star,
                }
            )
    return samples


def split_trajectories(conf, samples: list[dict]) -> dict[str, set[str]]:
    ids = sorted({sample["trajectory_id"] for sample in samples})
    if len(ids) < 3:
        raise ValueError("Need at least 3 trajectories to create train/val/test splits.")
    rng = random.Random(int(conf.sufficiency.random_seed))
    rng.shuffle(ids)

    train_fraction = float(conf.sufficiency.train_fraction)
    val_fraction = float(conf.sufficiency.val_fraction)
    n_total = len(ids)
    n_train = max(1, int(round(n_total * train_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train : n_train + n_val])
    test_ids = set(ids[n_train + n_val :])
    if not test_ids:
        test_ids = {ids[-1]}
        if ids[-1] in val_ids:
            val_ids.remove(ids[-1])
            val_ids.add(ids[-2])
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def partition_samples(samples: list[dict], splits: dict[str, set[str]]) -> dict[str, list[dict]]:
    return {
        split: [sample for sample in samples if sample["trajectory_id"] in split_ids]
        for split, split_ids in splits.items()
    }


def compute_normalization_stats(samples: list[dict], feature_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for sample in samples:
        rows.extend(sample["steps"])
    tensor = torch.tensor(rows, dtype=torch.float32)
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    if mean.numel() != feature_dim:
        raise ValueError("Feature dimension mismatch while computing normalization stats.")
    return mean, std


class LandscapeSufficiencyDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        sequence_window: int,
    ):
        self.samples = samples
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.sequence_window = int(sequence_window)
        self.feature_dim = int(feature_mean.numel())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        steps = torch.tensor(sample["steps"], dtype=torch.float32)
        steps = (steps - self.feature_mean) / self.feature_std
        padded = torch.zeros((self.sequence_window, self.feature_dim), dtype=torch.float32)
        padded[-steps.shape[0] :] = steps
        return {
            "x": padded,
            "label": torch.tensor(sample["label"], dtype=torch.float32),
            "log_k_star": torch.tensor(sample["log_k_star"], dtype=torch.float32),
            "k": torch.tensor(sample["k"], dtype=torch.float32),
            "k_star": torch.tensor(sample["k_star"], dtype=torch.float32),
        }
