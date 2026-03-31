#!/usr/bin/env python3
"""
Build log-log scaling figures for the NeurIPS paper from landscape_experiments.json.

From repository root:
  python code/run_experiments.py       # writes output/landscape/landscape_experiments.json
  python code/plot_scaling_from_json.py

PDFs are written to paper/figures/ (matches \\graphicspath in main.tex).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIGURES = _REPO_ROOT / "paper" / "figures"


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate_by_k(runs: list[dict]) -> dict[int, dict[str, np.ndarray]]:
    """Mean/std over seeds for each k."""
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in runs:
        by_k[int(row["k"])].append(row)

    out: dict[int, dict[str, np.ndarray]] = {}
    for k, rows in sorted(by_k.items()):
        d1 = np.array([r["delta1"] for r in rows])
        d2 = np.array([r["delta2"] for r in rows])
        out[k] = {
            "delta1_mean": d1.mean(),
            "delta1_std": d1.std(ddof=1) if len(d1) > 1 else 0.0,
            "delta2_mean": d2.mean(),
            "delta2_std": d2.std(ddof=1) if len(d2) > 1 else 0.0,
        }
        # subspace keys e.g. "5", "10", "20"
        if rows and "delta2_subspace" in rows[0]:
            dims = sorted(int(x) for x in rows[0]["delta2_subspace"].keys())
            for D in dims:
                vals = np.array([r["delta2_subspace"][str(D)] for r in rows])
                out[k][f"d2s_{D}_mean"] = vals.mean()
                out[k][f"d2s_{D}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
    return out


def plot_main(agg: dict[int, dict], out_path: Path) -> None:
    ks = np.array(sorted(agg.keys()))
    d1 = np.array([agg[k]["delta1_mean"] for k in ks])
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(ks, d1, "o-", label=r"$\Delta_1$ (est.)", ms=4)
    ax.loglog(ks, d2, "s-", label=r"$\Delta_2$ (est.)", ms=4)

    # Optional: plot reference slopes -1 and -2 (shifted)
    ref = ks.astype(float)
    if d1.size and d1[0] > 0:
        c1 = d1[len(d1) // 2] / (ref[len(ref) // 2] ** (-1))
        ax.loglog(ks, c1 * ref ** (-1), "k--", alpha=0.35, label=r"slope $-1$")
    if d2.size and d2[0] > 0:
        c2 = d2[len(d2) // 2] / (ref[len(ref) // 2] ** (-2))
        ax.loglog(ks, c2 * ref ** (-2), "k:", alpha=0.35, label=r"slope $-2$")

    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(r"Criterion value")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_subspace(agg: dict[int, dict], D: int, out_path: Path) -> None:
    ks = np.array(sorted(agg.keys()))
    key_m = f"d2s_{D}_mean"
    if ks.size == 0 or key_m not in agg[int(ks[0])]:
        print(f"No subspace key {key_m} in data; skip {out_path.name}", file=sys.stderr)
        return
    vals = np.array([agg[k][key_m] for k in ks])
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.loglog(ks, d2, "s-", label=r"$\Delta_2$", ms=4)
    ax.loglog(ks, vals, "^-", label=rf"$\Delta_2^{{({D})}}$", ms=4)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(r"Mean-squared increment (est.)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    default_json = _REPO_ROOT / "output" / "landscape" / "landscape_experiments.json"
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    out_dir = _FIGURES

    if not json_path.is_file():
        print(f"Missing {json_path}; run: python code/run_experiments.py", file=sys.stderr)
        raise SystemExit(1)

    runs = load_runs(json_path)
    agg = aggregate_by_k(runs)
    plot_main(agg, out_dir / "scaling_delta_loglog.pdf")
    for D in (5, 10, 20):
        plot_subspace(agg, D, out_dir / f"scaling_delta2_subspace_D{D}.pdf")


if __name__ == "__main__":
    main()
