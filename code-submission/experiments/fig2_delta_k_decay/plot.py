"""
Two figures from run_delta_k_curriculum JSON:

  1) Δ_1, Δ_2 (full-space), Δ_2^(D) (GM) vs k (log-log)
  2) Validation CE and PPL vs k (log-linear)

  cd code
  python -m scripts.plot_delta_k_curriculum --json .nanochat/reports/delta_k_curriculum.json \\
      --out-prefix .nanochat/reports/delta_k_curriculum
"""
from __future__ import annotations

import argparse
import json
import math
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
        mpl.rcParams.update({"font.family": "serif", "font.size": 11})
        return
    font_manager.fontManager.addfont(str(path))
    fp = font_manager.FontProperties(fname=str(path))
    name = fp.get_name()
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [name] + [f for f in mpl.rcParams["font.serif"] if f != name]
    mpl.rcParams["font.size"] = 11


def _sigma_label(sig: float) -> str:
    if sig >= 0.01:
        return f"{sig:g}"
    return f"{sig:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def plot_criteria(data: dict, out: Path, font_path: Path, figsize: tuple[float, float]) -> None:
    _setup_times_font(font_path)
    meta = data["meta"]
    rows = sorted(data["by_k"], key=lambda r: r["k"])
    ks = np.array([r["k"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ (sequences consumed in training)")
    ax.set_ylabel(r"Criteria")

    d1 = np.array([r["delta1"]["mean"] for r in rows])
    ax.plot(ks, d1, marker="o", linewidth=1.6, label=r"$\Delta_1$")

    sigmas = meta.get("sigmas") or list({s for r in rows for s in [x["sigma"] for x in r["delta2_fullspace"]]})
    sigmas = sorted(set(float(s) for s in sigmas))
    for sig in sigmas:
        ys = []
        for r in rows:
            m = next(
                x
                for x in r["delta2_fullspace"]
                if math.isclose(float(x["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
            )
            ys.append(m["mean"])
        ax.plot(ks, ys, marker="s", linewidth=1.4, linestyle="--", label=rf"$\Delta_2$, $\sigma={_sigma_label(float(sig))}$")

    d_list = meta.get("D_list") or [1, 4, 16]
    markers = [".", "^", "v", "D", "P"]
    mi = 0
    for D in sorted(set(d_list)):
        for sig in sigmas:
            ys = []
            for r in rows:
                ent = next(
                    e
                    for e in r["delta2_subspace_gaussian"]
                    if int(e["D"]) == int(D)
                    and math.isclose(float(e["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
                )
                ys.append(ent["gaussian_moment"])
            mkr = markers[mi % len(markers)]
            mi += 1
            ax.plot(
                ks,
                ys,
                marker=mkr,
                linewidth=1.2,
                linestyle="-.",
                markersize=5,
                label=rf"$\Delta_2^{{({D})}}$, $\sigma={_sigma_label(float(sig))}$ (GM)",
            )

    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best", fontsize=6, ncol=2, framealpha=0.92)
    tag = meta.get("depth", "?")
    ax.set_title(f"Landscape criteria vs $k$ (depth {tag})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_val(data: dict, out: Path, font_path: Path, figsize: tuple[float, float]) -> None:
    _setup_times_font(font_path)
    meta = data["meta"]
    rows = sorted(data["by_k"], key=lambda r: r["k"])
    ks = np.array([r["k"] for r in rows], dtype=float)
    ce = np.array([r["val_loss_mean_ce"] for r in rows])
    ppl = np.array([r["val_ppl"] for r in rows])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ (sequences consumed in training)")
    ax.set_ylabel("Validation loss / perplexity")
    ax.plot(ks, ce, marker="o", linewidth=1.8, label="val CE (nats)")
    ax.plot(ks, ppl, marker="s", linewidth=1.8, label="val PPL")
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"Validation metrics vs $k$ (depth {meta.get('depth', '?')})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out-prefix", type=str, default=None, help="Writes <prefix>_criteria.pdf and <prefix>_val.pdf")
    ap.add_argument("--font", type=str, default=str(_default_times_font_path()))
    ap.add_argument("--figsize", type=str, default="7,4.5")
    args = ap.parse_args()

    path = Path(args.json)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    w, h = (float(x) for x in args.figsize.split(","))
    fp = Path(args.font)
    prefix = Path(args.out_prefix) if args.out_prefix else path.with_suffix("")
    plot_criteria(data, Path(str(prefix) + "_criteria.pdf"), fp, (w, h))
    plot_val(data, Path(str(prefix) + "_val.pdf"), fp, (w, h))
    print(f"Wrote {prefix}_criteria.pdf")
    print(f"Wrote {prefix}_val.pdf")


if __name__ == "__main__":
    main()
