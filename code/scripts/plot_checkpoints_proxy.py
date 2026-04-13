"""
Plot quadratic proxy validity across training checkpoints (Experiment 3).

Produces a single figure with one line per checkpoint (training step),
showing mean relative error vs sigma on a log-x scale.

Usage:
  cd code
  python -m scripts.plot_checkpoints_proxy \\
      --json .nanochat/reports/checkpoints_proxy_<ts>.json \\
      --out figures/checkpoints_proxy.pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_times_font_path() -> Path:
    return _repo_root() / "code-old" / "Times New Roman.ttf"


def _setup_times_font(font_path: Path) -> None:
    path = Path(font_path)
    if not path.is_file():
        return  # graceful fallback
    font_manager.fontManager.addfont(str(path))
    fp = font_manager.FontProperties(fname=str(path))
    name = fp.get_name()
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [name] + [f for f in mpl.rcParams["font.serif"] if f != name]


def _axis_label_fontsize() -> float:
    try:
        return float(mpl.rcParams["font.size"]) + 2.0
    except (TypeError, ValueError):
        return 12.0


def main() -> None:
    p = argparse.ArgumentParser(description="Plot proxy validity across checkpoints.")
    p.add_argument("--json", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--font", type=str, default=str(_default_times_font_path()))
    args = p.parse_args()

    _setup_times_font(Path(args.font))
    label_fs = _axis_label_fontsize()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    by_step = data["by_step"]
    if not by_step:
        raise SystemExit("No by_step entries in JSON.")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(by_step)))

    for entry, color in zip(by_step, colors):
        step = entry["step"]
        agg = entry["aggregated_by_sigma"]
        sigmas = np.array([r["sigma"] for r in agg], dtype=float)
        means = np.array([r["rel_err_mean_mean"] for r in agg], dtype=float)
        stds = np.array([r["rel_err_mean_std"] for r in agg], dtype=float)
        lo = np.maximum(means - stds, 0.0)
        hi = means + stds

        ax.plot(sigmas, means, color=color, lw=1.8, marker="o", markersize=4,
                label=f"step {step}")
        ax.fill_between(sigmas, lo, hi, alpha=0.15, color=color)

    # Mark the sigma <= 1e-3 boundary
    ax.axvline(1e-3, color="red", lw=1.2, ls="--", alpha=0.7, label=r"$\sigma = 10^{-3}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Isotropic Gaussian scale $\sigma$", fontsize=label_fs)
    ax.set_ylabel("Mean relative error", fontsize=label_fs)
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)

    out = args.out
    if out is None:
        base = Path(args.json)
        out = str(base.with_name(base.stem + "_plot.pdf"))
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
