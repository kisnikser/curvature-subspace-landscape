"""
Plots from run_delta2_robustness JSON:

  1) Heatmap: |direct − GM| / (|direct| + ε) over (σ, D)
  2) S-curve: Direct MC & Quadratic MC vs S with stderr; GM horizontal line

Styling matches plot_quadratic_sigma.py (Times New Roman, axis labels +2 pt, grid major α=0.2).

  cd code
  python -m scripts.plot_delta2_robustness --json .nanochat/reports/delta2_robustness_....json
  python -m scripts.plot_delta2_robustness --json ...json --heatmap-norm data
  python -m scripts.plot_delta2_robustness --json ...json --heatmap-norm fixed --heatmap-vmin 0 --heatmap-vmax 1
  python -m scripts.plot_delta2_robustness --json ...json --heatmap-sigma-all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import Normalize


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


def _edges_log_sigma(sigmas: np.ndarray) -> np.ndarray:
    """Return length len(sigmas)+1 edges on linear σ for pcolormesh (y-axis log scale)."""
    sig = np.asarray(sigmas, dtype=float)
    if sig.size == 0:
        raise ValueError("empty sigmas")
    ls = np.log10(sig)
    if sig.size == 1:
        w = 0.05
        return np.array([10 ** (ls[0] - w), 10 ** (ls[0] + w)])
    out = np.zeros(sig.size + 1)
    out[1:-1] = (ls[:-1] + ls[1:]) / 2.0
    out[0] = ls[0] - (ls[1] - ls[0]) / 2.0
    out[-1] = ls[-1] + (ls[-1] - ls[-2]) / 2.0
    return 10**out


def _edges_D(D_list: list) -> np.ndarray:
    D_arr = np.array(D_list, dtype=float)
    if D_arr.size == 0:
        raise ValueError("empty D_list")
    if D_arr.size == 1:
        return np.array([D_arr[0] - 0.5, D_arr[0] + 0.5])
    out = np.zeros(D_arr.size + 1)
    out[0] = D_arr[0] - (D_arr[1] - D_arr[0]) / 2.0
    out[-1] = D_arr[-1] + (D_arr[-1] - D_arr[-2]) / 2.0
    out[1:-1] = (D_arr[:-1] + D_arr[1:]) / 2.0
    return out


def plot_heatmap(
    data: dict,
    out: Path,
    font_path: Path,
    *,
    norm_mode: str = "data",
    vmin: float = 0.0,
    vmax: float = 1.0,
    sigma_lt: float | None = 1e-2,
    figsize: tuple[float, float] = (5.0, 3.5),
) -> None:
    _setup_times_font(font_path)
    label_fs = _axis_label_fontsize()

    hm = data["heatmap"]
    sigmas = np.asarray(hm["sigmas"], dtype=float)
    D_list = hm["D_list"]
    Z = np.asarray(hm["rel_err_direct_gm"], dtype=float)

    if sigma_lt is not None:
        mask = sigmas < float(sigma_lt)
        if not np.any(mask):
            raise SystemExit(
                f"No heatmap rows with sigma < {sigma_lt}; relax --heatmap-sigma-lt or use --heatmap-sigma-all."
            )
        sigmas = sigmas[mask]
        Z = Z[mask, :]

    if norm_mode == "data":
        z_min = float(np.nanmin(Z)) if Z.size else 0.0
        z_max = float(np.nanmax(Z)) if Z.size else 1.0
        if not np.isfinite(z_min) or not np.isfinite(z_max):
            raise SystemExit("Heatmap Z has no finite values.")
        if z_max <= z_min:
            z_max = z_min + 1e-15
        vmin, vmax = z_min, z_max
        clip = False
        cbar_extend = "neither"
    else:
        clip = True
        z_max = float(np.nanmax(Z)) if Z.size else 0.0
        cbar_extend = "max" if z_max > vmax + 1e-15 else "neither"

    d_edges = _edges_D(D_list)
    s_edges = _edges_log_sigma(sigmas)
    X, Y = np.meshgrid(d_edges, s_edges)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=clip)

    fig, ax = plt.subplots(figsize=figsize)
    pcm = ax.pcolormesh(X, Y, Z, shading="flat", cmap="viridis", norm=norm)
    ax.set_yscale("log")
    ax.set_xlabel(r"Subspace dimension $D$", fontsize=label_fs)
    ax.set_ylabel(r"Isotropic Gaussian scale $\sigma$", fontsize=label_fs)
    cbar = fig.colorbar(pcm, ax=ax, extend=cbar_extend)
    cbar.ax.set_ylabel(
        r"$|\widehat{\Delta}_{\mathrm{direct}} - \mathrm{GM}|"
        r" / (|\widehat{\Delta}_{\mathrm{direct}}| + \varepsilon)$",
        fontsize=label_fs,
    )
    ax.grid(True, which="major", alpha=0.2)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_s_curve(
    data: dict,
    out: Path,
    font_path: Path,
    *,
    figsize: tuple[float, float] = (4.0, 3.0),
) -> None:
    _setup_times_font(font_path)
    label_fs = _axis_label_fontsize()

    sc = data["s_curve"]
    S_list = np.asarray(sc["S_list"], dtype=float)
    direct_m = np.asarray(sc["direct_mean"], dtype=float)
    direct_s = np.asarray(sc["direct_stderr"], dtype=float)
    qmc_m = np.asarray(sc["qmc_mean"], dtype=float)
    qmc_s = np.asarray(sc["qmc_stderr"], dtype=float)
    gm = float(sc["gm"])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(S_list, direct_m, color="C0", lw=2, marker="o", markersize=6, label="Direct MC")
    ax.fill_between(S_list, direct_m - direct_s, direct_m + direct_s, alpha=0.2, color="C0")
    ax.plot(S_list, qmc_m, color="C1", lw=2, marker="s", markersize=5, label="Quadratic MC")
    ax.fill_between(S_list, qmc_m - qmc_s, qmc_m + qmc_s, alpha=0.2, color="C1")
    ax.axhline(gm, color="k", ls="--", lw=1.5, label="Gaussian moment (closed form)")
    ax.set_xlabel(r"MC sample count $S$", fontsize=label_fs)
    ax.set_ylabel(r"$\widehat{\Delta}_2^{(D)}$", fontsize=label_fs)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True, help="JSON from run_delta2_robustness")
    ap.add_argument(
        "--which",
        choices=("both", "heatmap", "scurve"),
        default="both",
        help="Which figure(s) to write",
    )
    ap.add_argument("--out-heatmap", type=str, default=None)
    ap.add_argument("--out-scurve", type=str, default=None)
    ap.add_argument(
        "--font",
        type=str,
        default=str(_default_times_font_path()),
        help="Path to Times New Roman .ttf (default: repo code-old/Times New Roman.ttf)",
    )
    ap.add_argument(
        "--heatmap-norm",
        choices=("data", "fixed"),
        default="data",
        help="data: vmin/vmax = min/max of Z in the plotted (sigma-filtered) region; fixed: use --heatmap-vmin/--heatmap-vmax",
    )
    ap.add_argument(
        "--heatmap-vmin",
        type=float,
        default=0.0,
        help="Heatmap color scale min (only with --heatmap-norm fixed)",
    )
    ap.add_argument(
        "--heatmap-vmax",
        type=float,
        default=1.0,
        help="Heatmap color scale max (only with --heatmap-norm fixed)",
    )
    ap.add_argument(
        "--heatmap-sigma-lt",
        type=float,
        default=1e-2,
        help="Heatmap: keep only rows with sigma < this (default 1e-2 drops σ in [1e-2, 1e-1] for typical log grids)",
    )
    ap.add_argument(
        "--heatmap-sigma-all",
        action="store_true",
        help="Heatmap: plot all sigma rows from JSON (ignore --heatmap-sigma-lt)",
    )
    args = ap.parse_args()
    if args.heatmap_norm == "fixed" and args.heatmap_vmax <= args.heatmap_vmin:
        raise SystemExit("--heatmap-vmax must be greater than --heatmap-vmin when using --heatmap-norm fixed")

    font_path = Path(args.font)
    base = Path(args.json)

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)

    if "heatmap" not in data or "s_curve" not in data:
        raise SystemExit("JSON must contain 'heatmap' and 's_curve' (from run_delta2_robustness).")

    stem = base.stem
    out_h = args.out_heatmap or str(base.with_name(f"{stem}_heatmap.pdf"))
    out_s = args.out_scurve or str(base.with_name(f"{stem}_scurve.pdf"))

    sigma_lt = None if args.heatmap_sigma_all else args.heatmap_sigma_lt
    if args.which in ("both", "heatmap"):
        plot_heatmap(
            data,
            Path(out_h),
            font_path,
            norm_mode=args.heatmap_norm,
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax,
            sigma_lt=sigma_lt,
        )
    if args.which in ("both", "scurve"):
        plot_s_curve(data, Path(out_s), font_path)


if __name__ == "__main__":
    main()
