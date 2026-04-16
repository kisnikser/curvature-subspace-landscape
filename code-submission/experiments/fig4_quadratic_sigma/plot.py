"""
Plot quadratic Taylor error vs σ from exp_quadratic_sigma JSON.

  --mode rel   relative error (default), x log
  --mode abs   mean |Δ_true − Δ_quad| vs σ, log-log (good for small σ)

  cd code
  python -m scripts.plot_quadratic_sigma --json .nanochat/reports/....json --mode abs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_Y_FLOOR = 1e-20


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
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, required=True, help="Report JSON from exp_quadratic_sigma")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG (default: *_plot.pdf or *_plot_abs.pdf)",
    )
    p.add_argument("--title", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=("rel", "abs"),
        default="rel",
        help="rel: relative error; abs: mean absolute error, log-log",
    )
    p.add_argument(
        "--font",
        type=str,
        default=str(_default_times_font_path()),
        help="Path to Times New Roman .ttf (default: repo code-old/Times New Roman.ttf)",
    )
    args = p.parse_args()

    _setup_times_font(Path(args.font))
    label_fs = _axis_label_fontsize()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    agg = data.get("aggregated_by_sigma")
    if not agg:
        raise SystemExit(
            "No aggregated_by_sigma in JSON; re-run exp_quadratic_sigma.py with --seeds (default 0,1,2)."
        )

    sigmas = np.array([row["sigma"] for row in agg], dtype=float)

    if args.mode == "rel":
        means = np.array([row["rel_err_mean_mean"] for row in agg], dtype=float)
        stds = np.array([row["rel_err_mean_std"] for row in agg], dtype=float)
        lo = np.maximum(means - stds, 0.0)
        hi = means + stds
        y_label = "Mean relative error (MC, then mean over seeds)"
        line_label = "mean rel. err. (across seeds)"
        band_label = "±1 std across seeds"
    else:
        missing = [i for i, row in enumerate(agg) if "abs_err_mean_mean" not in row]
        if missing:
            raise SystemExit(
                "JSON missing abs_err_mean_mean; re-run exp_quadratic_sigma.py (current version writes absolute-error aggregates)."
            )
        means = np.array([row["abs_err_mean_mean"] for row in agg], dtype=float)
        stds = np.array([row["abs_err_mean_std"] for row in agg], dtype=float)
        lo = np.maximum(means - stds, _Y_FLOOR)
        hi = np.maximum(means + stds, _Y_FLOOR)
        y_label = r"Approximation error"
        line_label = "mean"
        band_label = "±1 std"

    fig, ax = plt.subplots(figsize=(4, 3))
    n = len(sigmas)
    if n == 1:
        ax.errorbar(
            sigmas,
            means,
            yerr=stds,
            fmt="o",
            color="C0",
            capsize=6,
            capthick=1.5,
            elinewidth=1.5,
            markersize=8,
            label="mean ±1 std (across seeds)",
        )
        s0 = float(sigmas[0])
        ax.set_xlim(s0 / 3.0, s0 * 3.0)
    else:
        ax.plot(sigmas, means, color="C0", lw=2, marker="o", markersize=6, label=line_label)
        ax.fill_between(sigmas, lo, hi, alpha=0.2, color="C0", label=band_label)
        ax.set_xlim(sigmas.min() / 1.4, sigmas.max() * 1.4)

    ax.set_xscale("log")
    ax.set_xlabel(r"Isotropic Gaussian scale $\sigma$", fontsize=label_fs)
    ax.set_ylabel(y_label, fontsize=label_fs)
    if args.mode == "abs":
        ax.set_yscale("log")

    meta = data.get("meta", {})
    # title = args.title
    # if title is None:
    #     suffix = " |Δ−Δquad|" if args.mode == "abs" else " rel. err."
    #     title = f"Quadratic approx. vs σ{suffix} (seeds={meta.get('seeds', '?')}, n_samples={meta.get('num_samples', '?')})"
    # ax.set_title(title)
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best")

    out = args.out
    if out is None:
        base = Path(args.json)
        stem = base.stem + ("_plot_abs.pdf" if args.mode == "abs" else "_plot.pdf")
        out = str(base.with_name(stem))
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
