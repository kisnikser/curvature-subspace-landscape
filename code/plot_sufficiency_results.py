#!/usr/bin/env python3
"""Build paper-ready plots and a LaTeX table from sufficiency_results.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "code" / "output" / "sufficiency" / "sufficiency_results.json"
FIG_DIR = REPO_ROOT / "paper" / "figures"


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def sort_predictions(method_result: dict) -> list[dict]:
    preds = method_result.get("test_predictions", [])
    return sorted(preds, key=lambda row: float(row["current_k"]))


def filter_methods(results: dict, epsilon: float) -> list[dict]:
    return [
        row
        for row in results["methods"]
        if abs(float(row.get("label_epsilon", epsilon)) - epsilon) < 1e-12
    ]


def plot_metric_bars(methods: list[dict], out_path: Path, epsilon: float) -> None:
    names = [row["display_name"] for row in methods]
    auroc = [row["test_summary"]["auroc"]["mean"] for row in methods]
    auroc_err = [row["test_summary"]["auroc"]["std"] for row in methods]
    brier = [row["test_summary"]["brier"]["mean"] for row in methods]
    brier_err = [row["test_summary"]["brier"]["std"] for row in methods]
    mae = [row["test_summary"]["mae_k"]["mean"] for row in methods]
    mae_err = [row["test_summary"]["mae_k"]["std"] for row in methods]
    xs = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    datasets = [
        (auroc, auroc_err, "Test AUROC", (0.0, 1.05)),
        (brier, brier_err, "Test Brier", None),
        (mae, mae_err, r"Test MAE$(\hat{k}_*)$", None),
    ]
    for ax, (vals, errs, ylabel, ylim) in zip(axes, datasets):
        bars = ax.bar(
            xs,
            vals,
            yerr=errs,
            capsize=4,
            color=colors[: len(names)],
            width=0.72,
        )
        ax.set_xticks(xs, names, rotation=15)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.2f}" if val < 100 else f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle(
        rf"Sufficiency metrics by criterion-specific LSTM ($\varepsilon={epsilon:.2f}$)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def _aggregate_prediction_curves(method_result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_k: dict[float, dict[str, list[float]]] = {}
    for repeat in method_result.get("repeats", []):
        for pred in repeat.get("test_predictions", []):
            k = float(pred["current_k"])
            bucket = by_k.setdefault(
                k,
                {"prob": [], "pred_k": [], "true_k": []},
            )
            bucket["prob"].append(float(pred["prob_sufficient"]))
            bucket["pred_k"].append(float(pred["predicted_k_star"]))
            bucket["true_k"].append(float(pred["true_k_star"]))
    ks = np.array(sorted(by_k.keys()))
    prob_mean = np.array([np.mean(by_k[k]["prob"]) for k in ks])
    pred_mean = np.array([np.mean(by_k[k]["pred_k"]) for k in ks])
    pred_std = np.array([np.std(by_k[k]["pred_k"], ddof=0) for k in ks])
    true_mean = np.array([np.mean(by_k[k]["true_k"]) for k in ks])
    return ks, prob_mean, pred_mean, pred_std, true_mean


def plot_prediction_curves(methods: list[dict], out_path: Path, epsilon: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    for row in methods:
        ks, probs, pred_k, pred_k_std, true_k = _aggregate_prediction_curves(row)
        axes[0].plot(ks, probs, marker="o", label=row["display_name"])
        axes[1].plot(ks, pred_k, marker="o", label=row["display_name"])
        axes[1].fill_between(
            ks,
            np.maximum(pred_k - pred_k_std, 1e-12),
            np.maximum(pred_k + pred_k_std, 1e-12),
            alpha=0.15,
        )
        axes[1].plot(ks, true_k, "k--", alpha=0.12)

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Current sample size $k$")
    axes[0].set_ylabel(r"Predicted sufficiency probability")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Current sample size $k$")
    axes[1].set_ylabel(r"Predicted $\hat{k}_*^{(\varepsilon)}$")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        rf"Aggregated sufficiency predictions over repeated splits ($\varepsilon={epsilon:.2f}$)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def write_latex_table(methods: list[dict], out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Test AUROC & Test Brier & Test MAE$(\hat{k}_*)$ \\",
        r"\midrule",
    ]
    for row in methods:
        metrics = row["test_summary"]
        lines.append(
            f"{row['display_name']} & "
            f"{metrics['auroc']['mean']:.2f}$\\pm${metrics['auroc']['std']:.2f} & "
            f"{metrics['brier']['mean']:.3f}$\\pm${metrics['brier']['std']:.3f} & "
            f"{metrics['mae_k']['mean']:.1f}$\\pm${metrics['mae_k']['std']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    epsilon = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    if not results_path.is_file():
        raise SystemExit(f"Missing {results_path}; run python code/train_sufficiency_module.py first.")
    results = load_results(results_path)
    methods = filter_methods(results, epsilon=epsilon)
    if not methods:
        raise SystemExit(f"No sufficiency results found for epsilon={epsilon:.4f}")

    plot_metric_bars(methods, FIG_DIR / "sufficiency_metrics_series1.pdf", epsilon=epsilon)
    plot_prediction_curves(methods, FIG_DIR / "sufficiency_predictions_series1.pdf", epsilon=epsilon)
    write_latex_table(methods, FIG_DIR / "sufficiency_metrics_series1_table.tex")
    print(f"Wrote {FIG_DIR / 'sufficiency_metrics_series1.pdf'}")
    print(f"Wrote {FIG_DIR / 'sufficiency_predictions_series1.pdf'}")
    print(f"Wrote {FIG_DIR / 'sufficiency_metrics_series1_table.tex'}")


if __name__ == "__main__":
    main()
