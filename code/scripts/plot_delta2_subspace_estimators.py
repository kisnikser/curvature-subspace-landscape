"""
Bar chart comparing direct subspace MC, quadratic MC, and Gaussian-moment closed form.

Styling matches plot_quadratic_sigma.py (Times New Roman, axis labels +2 pt, 4×3 in).

  python -m scripts.plot_delta2_subspace_estimators --json .nanochat/reports/delta2_subspace_estimators_....json
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
        raise SystemExit(f"Font file not found: {path}")
    font_manager.fontManager.addfont(str(path))
    fp = font_manager.FontProperties(fname=str(path))
    name = fp.get_name()
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [name] + [f for f in mpl.rcParams["font.serif"] if f != name]


def _axis_label_fontsize() -> float:
    fs = mpl.rcParams["font.size"]
    try:
        return float(fs) + 2.0
    except (TypeError, ValueError):
        return 12.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument(
        "--font",
        type=str,
        default=str(_default_times_font_path()),
        help="Path to Times New Roman .ttf (default: repo code-old/Times New Roman.ttf)",
    )
    args = ap.parse_args()

    _setup_times_font(Path(args.font))
    label_fs = _axis_label_fontsize()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    d2 = data["delta2_subspace"]
    direct = d2["direct_mc"]["mean"]
    dstd = d2["direct_mc"].get("std", 0.0)
    qmc = d2["quadratic_mc"]["mean"]
    qstd = d2["quadratic_mc"].get("std", 0.0)
    gm = float(d2["gaussian_moment_closed_form"])

    labels = ["Direct\nsubspace MC", "Quadratic\nMC", "Gaussian\nmoment"]
    means = np.maximum(np.array([direct, qmc, gm], dtype=float), 1e-30)
    yerr = np.array([dstd, qstd, 0.0], dtype=float)

    meta = data.get("meta", {})
    fig, ax = plt.subplots(figsize=(4, 3))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=yerr, capsize=4, color=["C0", "C1", "C2"], edgecolor="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\hat{\Delta}_2^{(D)}$ estimate", fontsize=label_fs)
    ax.set_yscale("log")
    # ax.grid(True, which="major", alpha=0.2)
    title = (
        rf"$D$={meta.get('subspace_dim', '?')}, $\sigma$={meta.get('sigma', '?')}, "
        rf"$k$ seq={meta.get('k_sequences', '?')}, MC $S$={meta.get('num_samples_mc', '?')}"
    )
    ax.set_title(title, fontsize=mpl.rcParams["font.size"])

    out = args.out
    if out is None:
        out = str(Path(args.json).with_name(Path(args.json).stem + "_plot.pdf"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
