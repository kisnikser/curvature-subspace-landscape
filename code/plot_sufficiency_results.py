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


def plot_metric_bars(results: dict, out_path: Path) -> None:
    methods = results["methods"]
    names = [row["display_name"] for row in methods]
    auroc = [row["test_metrics"]["auroc"] for row in methods]
    brier = [row["test_metrics"]["brier"] for row in methods]
    mae = [row["test_metrics"]["mae_k"] for row in methods]
    xs = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    datasets = [
        (auroc, "Test AUROC", (0.0, 1.05)),
        (brier, "Test Brier", None),
        (mae, r"Test MAE$(\hat{k}_*)$", None),
    ]
    for ax, (vals, ylabel, ylim) in zip(axes, datasets):
        bars = ax.bar(xs, vals, color=colors[: len(names)], width=0.72)
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
    fig.suptitle("Pilot sufficiency metrics by criterion-specific LSTM", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def plot_prediction_curves(results: dict, out_path: Path) -> None:
    methods = results["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    for row in methods:
        preds = sort_predictions(row)
        ks = np.array([float(item["current_k"]) for item in preds])
        probs = np.array([float(item["prob_sufficient"]) for item in preds])
        pred_k = np.array([float(item["predicted_k_star"]) for item in preds])
        true_k = np.array([float(item["true_k_star"]) for item in preds])
        axes[0].plot(ks, probs, marker="o", label=row["display_name"])
        axes[1].plot(ks, pred_k, marker="o", label=row["display_name"])
        axes[1].plot(ks, true_k, "k--", alpha=0.15)

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

    fig.suptitle("Pilot sufficiency predictions on the held-out trajectory", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def write_latex_table(results: dict, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Test AUROC & Test Brier & Test MAE$(\hat{k}_*)$ \\",
        r"\midrule",
    ]
    for row in results["methods"]:
        metrics = row["test_metrics"]
        lines.append(
            f"{row['display_name']} & "
            f"{metrics['auroc']:.2f} & "
            f"{metrics['brier']:.3f} & "
            f"{metrics['mae_k']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    if not results_path.is_file():
        raise SystemExit(f"Missing {results_path}; run python code/train_sufficiency_module.py first.")
    results = load_results(results_path)

    plot_metric_bars(results, FIG_DIR / "sufficiency_metrics_pilot.pdf")
    plot_prediction_curves(results, FIG_DIR / "sufficiency_predictions_pilot.pdf")
    write_latex_table(results, FIG_DIR / "sufficiency_metrics_pilot_table.tex")
    print(f"Wrote {FIG_DIR / 'sufficiency_metrics_pilot.pdf'}")
    print(f"Wrote {FIG_DIR / 'sufficiency_predictions_pilot.pdf'}")
    print(f"Wrote {FIG_DIR / 'sufficiency_metrics_pilot_table.tex'}")


if __name__ == "__main__":
    main()
