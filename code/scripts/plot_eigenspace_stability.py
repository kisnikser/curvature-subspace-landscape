"""
Plot eigenspace stability results (Experiment 4).

Produces two figures:
  (a) overlap_mean vs k, one line per D  — shows whether stability grows with k
  (b) heatmap: (k x D) of overlap_mean  — quick overview of all cells

Usage:
  cd code
  python -m scripts.plot_eigenspace_stability \\
      --json .nanochat/reports/eigenspace_stability_<ts>.json \\
      --out-prefix figures/eigenspace_stability
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
        return
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
    p = argparse.ArgumentParser(description="Plot eigenspace stability (Experiment 4).")
    p.add_argument("--json", type=str, required=True)
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output path prefix; produces <prefix>_lines.pdf and <prefix>_heatmap.pdf",
    )
    p.add_argument("--font", type=str, default=str(_default_times_font_path()))
    args = p.parse_args()

    _setup_times_font(Path(args.font))
    label_fs = _axis_label_fontsize()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    meta = data["meta"]

    k_values = sorted(set(r["k"] for r in results))
    D_values = sorted(set(r["D"] for r in results))

    # Build lookup: (k, D) -> (overlap_mean, overlap_std)
    lookup: dict[tuple[int, int], tuple[float, float]] = {}
    for r in results:
        lookup[(r["k"], r["D"])] = (r["overlap_mean"], r["overlap_std"])

    # ----------------------------------------------------------------
    # Figure (a): overlap vs k, one line per D
    # ----------------------------------------------------------------
    fig_lines, ax_lines = plt.subplots(figsize=(5, 3.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(D_values)))

    for D, color in zip(D_values, colors):
        ks = []
        oms = []
        oss = []
        for k in k_values:
            if (k, D) in lookup:
                om, os_ = lookup[(k, D)]
                ks.append(k)
                oms.append(om)
                oss.append(os_)
        if not ks:
            continue
        ks_arr = np.array(ks)
        oms_arr = np.array(oms)
        oss_arr = np.array(oss)
        lo = np.clip(oms_arr - oss_arr, 0.0, 1.0)
        hi = np.clip(oms_arr + oss_arr, 0.0, 1.0)

        ax_lines.plot(ks_arr, oms_arr, color=color, lw=1.8, marker="o", markersize=5,
                      label=f"$D={D}$")
        ax_lines.fill_between(ks_arr, lo, hi, alpha=0.15, color=color)

    ax_lines.set_xlabel("Dataset size $k$", fontsize=label_fs)
    ax_lines.set_ylabel(
        r"Subspace overlap $\frac{1}{D}\|U_D^{(k)\top}U_D^{(k+1)}\|_F^2$",
        fontsize=label_fs - 1,
    )
    ax_lines.set_ylim(0.0, 1.05)
    ax_lines.grid(True, which="major", alpha=0.2)
    ax_lines.legend(loc="lower right", fontsize=8, framealpha=0.8)

    step_str = str(meta.get("model_step", "?"))
    tag_str = str(meta.get("model_tag", ""))

    # ----------------------------------------------------------------
    # Figure (b): heatmap (rows=D, cols=k)
    # ----------------------------------------------------------------
    mat = np.full((len(D_values), len(k_values)), np.nan)
    for i, D in enumerate(D_values):
        for j, k in enumerate(k_values):
            if (k, D) in lookup:
                mat[i, j] = lookup[(k, D)][0]

    fig_heat, ax_heat = plt.subplots(figsize=(max(4, len(k_values) * 0.9), max(3, len(D_values) * 0.7)))
    im = ax_heat.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto", origin="lower")
    fig_heat.colorbar(im, ax=ax_heat, label="Subspace overlap")

    ax_heat.set_xticks(range(len(k_values)))
    ax_heat.set_xticklabels([str(k) for k in k_values])
    ax_heat.set_yticks(range(len(D_values)))
    ax_heat.set_yticklabels([str(D) for D in D_values])
    ax_heat.set_xlabel("Dataset size $k$", fontsize=label_fs)
    ax_heat.set_ylabel("Subspace dimension $D$", fontsize=label_fs)

    # Annotate cells with the numeric value
    for i in range(len(D_values)):
        for j in range(len(k_values)):
            v = mat[i, j]
            if not np.isnan(v):
                text_color = "white" if v < 0.5 else "black"
                ax_heat.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=7, color=text_color)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    prefix = args.out_prefix
    if prefix is None:
        base = Path(args.json)
        prefix = str(base.with_name(base.stem))

    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    lines_path = prefix + "_lines.pdf"
    heat_path = prefix + "_heatmap.pdf"

    fig_lines.tight_layout()
    fig_lines.savefig(lines_path)
    plt.close(fig_lines)
    print(f"Wrote {lines_path}")

    fig_heat.tight_layout()
    fig_heat.savefig(heat_path)
    plt.close(fig_heat)
    print(f"Wrote {heat_path}")


if __name__ == "__main__":
    main()
