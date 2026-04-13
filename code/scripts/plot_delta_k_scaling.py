"""
Plot Δ_1, Δ_2 (full-space, per σ), Δ_2^(D) (Gaussian closed form, per D and σ), and validation
CE / perplexity vs k from run_delta_k_scaling JSON.

Validation curves are horizontal (same checkpoint); merge several JSONs later for bands.

  cd code
  python -m scripts.plot_delta_k_scaling --json .nanochat/reports/delta_k_scaling_....json \\
      --out .nanochat/reports/delta_k_scaling.pdf
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


def plot_from_json(data: dict, out: Path, font_path: Path, figsize: tuple[float, float]) -> None:
    _setup_times_font(font_path)
    meta = data["meta"]
    val = data["validation"]
    rows = sorted(data["by_k"], key=lambda r: r["k"])
    ks = np.array([r["k"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"$k$ (sequences in $\mathcal{L}_k$)")
    ax.set_ylabel(r"Landscape criteria (log scale)")
    ax.set_yscale("log")

    # Δ_1
    d1 = np.array([r["delta1"]["mean"] for r in rows])
    ax.plot(ks, d1, marker="o", linewidth=1.6, label=r"$\Delta_1$")

    # Δ_2 full-space per σ
    sigmas = meta.get("sigmas") or sorted({entry["sigma"] for r in rows for entry in r["delta2_fullspace"]})
    for sig in sigmas:
        ys = []
        for r in rows:
            m = next(
                x
                for x in r["delta2_fullspace"]
                if math.isclose(float(x["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
            )
            ys.append(m["mean"])
        lab = rf"$\Delta_2$, $\sigma={_sigma_label(float(sig))}$"
        ax.plot(ks, ys, marker="s", linewidth=1.4, linestyle="--", label=lab)

    # Δ_2^(D) Gaussian per (D, σ)
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
            lab = rf"$\Delta_2^{{({D})}}$, $\sigma={_sigma_label(float(sig))}$ (GM)"
            ax.plot(ks, ys, marker=mkr, linewidth=1.2, linestyle="-.", markersize=5, label=lab)

    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="upper right", fontsize=6, ncol=2, framealpha=0.92)

    # Right axis: validation (same for all k; horizontal reference lines)
    ax2 = ax.twinx()
    ce = float(val["val_loss_mean_ce"])
    ppl = float(val["val_ppl"])
    ax2.plot(ks, np.full_like(ks, ce), color="black", linewidth=2.0, linestyle=":", label="val CE")
    ax2.plot(ks, np.full_like(ks, ppl), color="#555555", linewidth=2.0, linestyle=(0, (3, 2)), label="val PPL")
    ax2.set_ylabel("Validation CE / PPL (linear)")
    ax2.set_ylim(0.0, max(ce * 1.2, ppl * 1.05))
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.92)

    tag = meta.get("model_tag") or "?"
    step = meta.get("model_step")
    title = f"k-scaling ({tag}" + (f", step {step}" if step is not None else "") + ")"
    ax.set_title(title)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--font", type=str, default=str(_default_times_font_path()))
    ap.add_argument("--figsize", type=str, default="7,4.5")
    args = ap.parse_args()

    path = Path(args.json)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    w, h = (float(x) for x in args.figsize.split(","))
    out = Path(args.out) if args.out else path.with_suffix(".pdf")
    plot_from_json(data, out, Path(args.font), (w, h))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
