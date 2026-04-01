#!/usr/bin/env python3
"""
Build log-log scaling figures for the NeurIPS paper from landscape_experiments.json.

From repository root:
  python code/run_experiments.py       # writes code/output/landscape/landscape_experiments.json
  python code/plot_scaling_from_json.py

PDFs are written to paper/figures/ (matches \\graphicspath in main.tex).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIGURES = _REPO_ROOT / "paper" / "figures"
_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def default_json_path() -> Path:
    conf = OmegaConf.load(_CONFIG)
    return _REPO_ROOT / str(conf.common.output_dir) / "landscape_experiments.json"


def load_runs(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    stderr = std / np.sqrt(arr.size) if arr.size else 0.0
    return {"mean": mean, "std": std, "stderr": stderr}


def available_settings(runs: list[dict]) -> list[str]:
    settings = []
    for row in runs:
        name = row.get("setting_name", "default")
        if name not in settings:
            settings.append(name)
    return settings


def primary_setting(runs: list[dict]) -> str:
    for row in runs:
        if row.get("setting_primary", False):
            return row["setting_name"]
    return available_settings(runs)[0]


def aggregate_by_k(runs: list[dict]) -> dict[int, dict[str, float]]:
    """Aggregate seed-level summaries for each k."""
    by_k: dict[int, list[dict]] = defaultdict(list)
    for row in runs:
        by_k[int(row["k"])].append(row)

    out: dict[int, dict[str, float]] = {}
    for k, rows in sorted(by_k.items()):
        d1 = np.array([r["delta1"] for r in rows])
        d2 = np.array([r["delta2"] for r in rows])
        val_rows = [r for r in rows if r.get("validation_loss") is not None]
        out[k] = {
            "delta1_mean": d1.mean(),
            "delta1_stderr": d1.std(ddof=1) / np.sqrt(len(d1)) if len(d1) > 1 else 0.0,
            "delta2_mean": d2.mean(),
            "delta2_stderr": d2.std(ddof=1) / np.sqrt(len(d2)) if len(d2) > 1 else 0.0,
        }
        if val_rows:
            v = np.array([r["validation_loss"] for r in val_rows], dtype=float)
            out[k]["validation_loss_mean"] = v.mean()
            out[k]["validation_loss_stderr"] = (
                v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            )
        if rows and "delta2_subspace" in rows[0]:
            dims = sorted(int(x) for x in rows[0]["delta2_subspace"].keys())
            for D in dims:
                vals = np.array([r["delta2_subspace"][str(D)] for r in rows])
                out[k][f"d2s_{D}_mean"] = vals.mean()
                out[k][f"d2s_{D}_stderr"] = (
                    vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                )
    return out


def setting_runs(runs: list[dict], setting_name: str) -> list[dict]:
    return [row for row in runs if row.get("setting_name", "default") == setting_name]


def setting_slug(setting_name: str, is_primary: bool) -> str:
    return "" if is_primary else f"_{setting_name}"


def available_dims(rows: list[dict]) -> list[int]:
    if not rows:
        return []
    return sorted(int(x) for x in rows[0].get("delta2_subspace", {}).keys())


def default_main_dim(rows: list[dict]) -> int:
    dims = available_dims(rows)
    if not dims:
        return 10
    if 10 in dims:
        return 10
    return dims[min(len(dims) - 1, 1)]


def available_sigmas(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return sorted(rows[0].get("delta2_sigma_sweep", {}).keys(), key=float)


def collect_nested_metric(rows: list[dict], key_fn) -> dict[int, dict[str, float]]:
    by_k = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(float(key_fn(row)))
    return {k: summarize(vals) for k, vals in sorted(by_k.items())}


def plot_with_band(ax, xs, ys, errs, fmt, label):
    ax.loglog(xs, ys, fmt, label=label, ms=4)
    lower = np.maximum(ys - errs, 1e-12)
    upper = np.maximum(ys + errs, 1e-12)
    ax.fill_between(xs, lower, upper, alpha=0.18)


def plot_main(agg: dict[int, dict], out_path: Path, title: str | None = None) -> None:
    ks = np.array(sorted(agg.keys()))
    d1 = np.array([agg[k]["delta1_mean"] for k in ks])
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])
    d1_err = np.array([agg[k]["delta1_stderr"] for k in ks])
    d2_err = np.array([agg[k]["delta2_stderr"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_with_band(ax, ks, d1, d1_err, "o-", r"$\Delta_1$ (est.)")
    plot_with_band(ax, ks, d2, d2_err, "s-", r"$\Delta_2$ (est.)")

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
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_validation_loss(agg: dict[int, dict], out_path: Path, title: str | None = None) -> None:
    ks = np.array(sorted(k for k in agg.keys() if "validation_loss_mean" in agg[k]))
    if ks.size == 0:
        return
    vals = np.array([agg[k]["validation_loss_mean"] for k in ks])
    errs = np.array([agg[k]["validation_loss_stderr"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot(ks, vals, "o-", ms=4, label="validation loss")
    ax.fill_between(ks, vals - errs, vals + errs, alpha=0.18)
    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel("Validation loss")
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_subspace(agg: dict[int, dict], D: int, out_path: Path, title: str | None = None) -> None:
    ks = np.array(sorted(agg.keys()))
    key_m = f"d2s_{D}_mean"
    if ks.size == 0 or key_m not in agg[int(ks[0])]:
        print(f"No subspace key {key_m} in data; skip {out_path.name}", file=sys.stderr)
        return
    vals = np.array([agg[k][key_m] for k in ks])
    vals_err = np.array([agg[k][f"d2s_{D}_stderr"] for k in ks])
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])
    d2_err = np.array([agg[k]["delta2_stderr"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_with_band(ax, ks, d2, d2_err, "s-", r"$\Delta_2$")
    plot_with_band(ax, ks, vals, vals_err, "^-", rf"$\Delta_2^{{({D})}}$")
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(r"Mean-squared increment (est.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_sigma_sweep(rows: list[dict], out_path: Path, title: str | None = None) -> None:
    sigmas = available_sigmas(rows)
    if not sigmas:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for sigma in sigmas:
        stats = collect_nested_metric(
            rows, lambda row, s=sigma: row["delta2_sigma_sweep"][s]["mean"]
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, "o-", rf"$\sigma={float(sigma):.2f}$")
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(r"$\Delta_2$ (est.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_strategy_comparison(rows: list[dict], D: int, out_path: Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for strategy, fmt in (("hessian", "^-"), ("random", "o-")):
        stats = collect_nested_metric(
            rows,
            lambda row, strategy=strategy, D=D:
                row["subspace_strategy_comparison"][strategy][str(D)]["mean"],
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, fmt, strategy)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(rf"$\Delta_2^{{({D})}}$ (est.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_refresh_comparison(rows: list[dict], D: int, out_path: Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for refresh, fmt in (("recompute", "^-"), ("freeze", "o-")):
        stats = collect_nested_metric(
            rows,
            lambda row, refresh=refresh, D=D:
                row["subspace_refresh_comparison"][refresh][str(D)]["mean"],
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, fmt, refresh)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(rf"$\Delta_2^{{({D})}}$ (est.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_drift(rows: list[dict], out_path: Path, title: str | None = None) -> None:
    dims = sorted(int(x) for x in rows[0].get("eigenspace_drift", {}).keys())
    if not dims:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for D in dims:
        stats = collect_nested_metric(
            rows,
            lambda row, D=D: row["eigenspace_drift"][str(D)],
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        ax.plot(ks, ys, "o-", label=rf"overlap $D={D}$", ms=4)
        ax.fill_between(ks, np.maximum(ys - errs, 0.0), np.minimum(ys + errs, 1.0), alpha=0.18)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel("Mean principal cosine overlap")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_quadratic_alignment(rows: list[dict], out_path: Path, title: str | None = None) -> None:
    stats_true = collect_nested_metric(rows, lambda row: row["quadratic_alignment"]["true_mean"])
    stats_quad = collect_nested_metric(rows, lambda row: row["quadratic_alignment"]["quadratic_mean"])
    ks = np.array(sorted(stats_true.keys()))
    true_vals = np.array([stats_true[k]["mean"] for k in ks])
    quad_vals = np.array([stats_quad[k]["mean"] for k in ks])
    true_err = np.array([stats_true[k]["stderr"] for k in ks])
    quad_err = np.array([stats_quad[k]["stderr"] for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_with_band(ax, ks, true_vals, true_err, "s-", "true increment")
    plot_with_band(ax, ks, quad_vals, quad_err, "^-", "quadratic proxy")
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(r"Subspace mean-squared increment")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_setting_comparison(runs: list[dict], out_path: Path, D: int) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for setting_name in available_settings(runs):
        rows = setting_runs(runs, setting_name)
        stats = collect_nested_metric(rows, lambda row, D=D: row["delta2_subspace"][str(D)])
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, "o-", setting_name)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel(rf"$\Delta_2^{{({D})}}$ (est.)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    default_json = default_json_path()
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    out_dir = _FIGURES

    if not json_path.is_file():
        print(f"Missing {json_path}; run: python code/run_experiments.py", file=sys.stderr)
        raise SystemExit(1)

    runs = load_runs(json_path)
    main_setting = primary_setting(runs)
    for setting_name in available_settings(runs):
        rows = setting_runs(runs, setting_name)
        agg = aggregate_by_k(rows)
        is_primary = setting_name == main_setting
        suffix = setting_slug(setting_name, is_primary)
        title = setting_name.replace("_", " ")
        plot_main(agg, out_dir / f"scaling_delta_loglog{suffix}.pdf", title=title)
        plot_validation_loss(
            agg,
            out_dir / f"validation_loss_vs_k{suffix}.pdf",
            title=title,
        )
        for D in available_dims(rows):
            plot_subspace(
                agg,
                D,
                out_dir / f"scaling_delta2_subspace_D{D}{suffix}.pdf",
                title=f"{title} (D={D})",
            )
        if rows:
            main_D = default_main_dim(rows)
            plot_sigma_sweep(rows, out_dir / f"ablation_sigma_sweep{suffix}.pdf", title=title)
            plot_strategy_comparison(
                rows,
                main_D,
                out_dir / f"ablation_random_vs_hessian_D{main_D}{suffix}.pdf",
                title=title,
            )
            plot_refresh_comparison(
                rows,
                main_D,
                out_dir / f"ablation_refresh_policy_D{main_D}{suffix}.pdf",
                title=title,
            )
            plot_drift(rows, out_dir / f"eigenspace_drift{suffix}.pdf", title=title)
            plot_quadratic_alignment(
                rows,
                out_dir / f"quadratic_alignment{suffix}.pdf",
                title=title,
            )

    if len(available_settings(runs)) > 1:
        primary_rows = setting_runs(runs, main_setting)
        primary_dims = available_dims(primary_rows)
        if primary_dims:
            main_D = default_main_dim(primary_rows)
            plot_setting_comparison(
                runs,
                out_dir / f"setting_comparison_delta2_subspace_D{main_D}.pdf",
                D=main_D,
            )


if __name__ == "__main__":
    main()
