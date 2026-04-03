#!/usr/bin/env python3
"""
Build all figures for the paper from landscape_experiments.json.

From repository root:
  python code/run_experiments.py
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
    from matplotlib.ticker import LogLocator
except ImportError as e:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIGURES = _REPO_ROOT / "paper" / "figures"
_CONFIG = Path(__file__).resolve().parent / "config.yaml"

COLORS = {
    "delta1": "#1f77b4",
    "delta2": "#ff7f0e",
    "d2s": "#2ca02c",
    "random": "#d62728",
    "bottom": "#9467bd",
    "mixed": "#8c564b",
    "direct": "#2ca02c",
    "quadMC": "#ff7f0e",
    "gm": "#d62728",
    "freeze": "#9467bd",
}


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
    seen = []
    for row in runs:
        name = row.get("setting_name", "default")
        if name not in seen:
            seen.append(name)
    return seen


def primary_setting(runs: list[dict]) -> str:
    for row in runs:
        if row.get("setting_primary", False):
            return row["setting_name"]
    return available_settings(runs)[0]


def setting_runs(runs: list[dict], name: str) -> list[dict]:
    return [r for r in runs if r.get("setting_name", "default") == name]


def available_dims(rows: list[dict]) -> list[int]:
    if not rows:
        return []
    return sorted(int(x) for x in rows[0].get("delta2_subspace", {}).keys())


def default_main_dim(rows: list[dict]) -> int:
    dims = available_dims(rows)
    if 10 in dims:
        return 10
    return dims[min(len(dims) - 1, 1)] if dims else 10


def available_sigmas(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    key = "sigma_subspace_sweep" if "sigma_subspace_sweep" in rows[0] else "delta2_sigma_sweep"
    return sorted(rows[0].get(key, {}).keys(), key=float)


def collect_nested_metric(rows, key_fn):
    by_k = defaultdict(list)
    for row in rows:
        by_k[int(row["k"])].append(float(key_fn(row)))
    return {k: summarize(vals) for k, vals in sorted(by_k.items())}


def aggregate_by_k(runs):
    by_k = defaultdict(list)
    for row in runs:
        by_k[int(row["k"])].append(row)
    out = {}
    for k, rows in sorted(by_k.items()):
        d1 = np.array([r["delta1"] for r in rows])
        d2 = np.array([r["delta2"] for r in rows])
        out[k] = {
            "delta1_mean": d1.mean(),
            "delta1_stderr": d1.std(ddof=1) / np.sqrt(len(d1)) if len(d1) > 1 else 0.0,
            "delta2_mean": d2.mean(),
            "delta2_stderr": d2.std(ddof=1) / np.sqrt(len(d2)) if len(d2) > 1 else 0.0,
        }
        if rows and "delta2_subspace" in rows[0]:
            for D_str in rows[0]["delta2_subspace"]:
                vals = np.array([r["delta2_subspace"][D_str] for r in rows])
                out[k][f"d2s_{D_str}_mean"] = vals.mean()
                out[k][f"d2s_{D_str}_stderr"] = (
                    vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                )
    return out


def plot_with_band(ax, xs, ys, errs, color, label, marker="o", ls="-"):
    ax.loglog(xs, ys, marker=marker, ls=ls, color=color, label=label, ms=4, lw=1.4)
    lower = np.maximum(ys - errs, 1e-15)
    upper = np.maximum(ys + errs, 1e-15)
    ax.fill_between(xs, lower, upper, alpha=0.15, color=color)


def add_reference_slope(ax, ks, ys, slope, style, label):
    ref = ks.astype(float)
    mid = len(ref) // 2
    if ys[mid] > 0:
        c = ys[mid] / (ref[mid] ** slope)
        ax.loglog(ks, c * ref ** slope, style, alpha=0.35, label=label, lw=1.2)


def finish(fig, ax, xlabel, ylabel, out_path, title=None, legend_kw=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    kw = dict(fontsize=7, framealpha=0.8)
    if legend_kw:
        kw.update(legend_kw)
    ax.legend(**kw)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Figure 1: reference_scaling.pdf ──────────────────────────────
def plot_reference_scaling(rows, out_path, D=10):
    agg = aggregate_by_k(rows)
    ks = np.array(sorted(agg.keys()))
    d1 = np.array([agg[k]["delta1_mean"] for k in ks])
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])
    d1e = np.array([agg[k]["delta1_stderr"] for k in ks])
    d2e = np.array([agg[k]["delta2_stderr"] for k in ks])
    d2s = np.array([agg[k].get(f"d2s_{D}_mean", np.nan) for k in ks])
    d2se = np.array([agg[k].get(f"d2s_{D}_stderr", 0.0) for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_with_band(ax, ks, d1, d1e, COLORS["delta1"], r"$\Delta_1$", "o")
    plot_with_band(ax, ks, d2, d2e, COLORS["delta2"], r"$\Delta_2$", "s")
    if not np.all(np.isnan(d2s)):
        plot_with_band(ax, ks, d2s, d2se, COLORS["d2s"], rf"$\Delta_2^{{({D})}}$", "^")
    add_reference_slope(ax, ks, d1, -1, "k--", r"slope $-1$")
    add_reference_slope(ax, ks, d2, -2, "k:", r"slope $-2$")
    finish(fig, ax, r"Training set size $k$", "Criterion value", out_path)


# ── Figure 2: reference_full_vs_subspace.pdf ─────────────────────
def plot_full_vs_subspace(rows, out_path, D=10):
    agg = aggregate_by_k(rows)
    ks = np.array(sorted(agg.keys()))
    d2 = np.array([agg[k]["delta2_mean"] for k in ks])
    d2e = np.array([agg[k]["delta2_stderr"] for k in ks])
    d2s = np.array([agg[k].get(f"d2s_{D}_mean", np.nan) for k in ks])
    d2se = np.array([agg[k].get(f"d2s_{D}_stderr", 0.0) for k in ks])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    plot_with_band(ax, ks, d2, d2e, COLORS["delta2"], r"$\Delta_2$ (full-space)", "s")
    if not np.all(np.isnan(d2s)):
        plot_with_band(ax, ks, d2s, d2se, COLORS["d2s"], rf"$\Delta_2^{{({D})}}$ (subspace)", "^")
    add_reference_slope(ax, ks, d2, -2, "k:", r"slope $-2$")
    finish(fig, ax, r"Training set size $k$", "Mean-squared increment (est.)", out_path)


# ── Figure 3: subspace_comparison.pdf ────────────────────────────
def plot_subspace_comparison(rows, out_path, D=10):
    strategies = [
        ("hessian", COLORS["d2s"], "^", f"top-$D$ Hessian"),
        ("random", COLORS["random"], "o", "random"),
        ("bottom", COLORS["bottom"], "v", f"bottom-$D$ Hessian"),
        ("mixed", COLORS["mixed"], "D", "mixed (top + bottom)"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for strategy, color, marker, label in strategies:
        try:
            stats = collect_nested_metric(
                rows,
                lambda row, s=strategy: row["subspace_strategy_comparison"][s][str(D)]["mean"],
            )
        except (KeyError, TypeError):
            continue
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, color, label, marker)
    finish(fig, ax, r"Training set size $k$", rf"$\Delta_2^{{({D})}}$ (est.)", out_path)


# ── Figure 4: direct_vs_proxy_curves.pdf ─────────────────────────
def plot_direct_vs_proxy(rows, out_path, D=10):
    stats_direct = collect_nested_metric(
        rows, lambda r: r["delta2_subspace"][str(D)]
    )
    stats_gm = collect_nested_metric(
        rows, lambda r: r["gm_estimator"]
    )
    stats_qmc = collect_nested_metric(
        rows, lambda r: r["quadMC_estimator"]["mean"]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ks = np.array(sorted(stats_direct.keys()))

    ys_d = np.array([stats_direct[k]["mean"] for k in ks])
    es_d = np.array([stats_direct[k]["stderr"] for k in ks])
    plot_with_band(ax, ks, ys_d, es_d, COLORS["direct"], "direct MC", "^")

    ys_q = np.array([stats_qmc[k]["mean"] for k in ks])
    es_q = np.array([stats_qmc[k]["stderr"] for k in ks])
    plot_with_band(ax, ks, ys_q, es_q, COLORS["quadMC"], "quadratic MC", "s")

    ys_g = np.array([stats_gm[k]["mean"] for k in ks])
    es_g = np.array([stats_gm[k]["stderr"] for k in ks])
    plot_with_band(ax, ks, ys_g, es_g, COLORS["gm"], "Gaussian-moment", "o")

    finish(fig, ax, r"Training set size $k$", rf"$\Delta_2^{{({D})}}$ estimator value", out_path)


# ── Figure 5: D_sweep.pdf ────────────────────────────────────────
def plot_D_sweep(rows, out_path):
    dims = available_dims(rows)
    if not dims:
        return
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, D in enumerate(dims):
        color = cmap(i / max(len(dims) - 1, 1))
        stats = collect_nested_metric(
            rows, lambda row, D=D: row["delta2_subspace"][str(D)]
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        plot_with_band(ax, ks, ys, errs, color, rf"$D={D}$", "o")
    finish(fig, ax, r"Training set size $k$", r"$\Delta_2^{(D)}$ (est.)", out_path)


# ── Figure 6: sigma_sweep.pdf ────────────────────────────────────
def plot_sigma_sweep(rows, out_path):
    sigmas = available_sigmas(rows)
    if not sigmas:
        return
    cmap = plt.cm.plasma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    for i, sigma_key in enumerate(sigmas):
        color = cmap(i / max(len(sigmas) - 1, 1))
        sigma_val = float(sigma_key)

        if "sigma_subspace_sweep" in rows[0] and sigma_key in rows[0]["sigma_subspace_sweep"]:
            stats_dir = collect_nested_metric(
                rows,
                lambda row, s=sigma_key: row["sigma_subspace_sweep"][s]["direct_mean"],
            )
            ks = np.array(sorted(stats_dir.keys()))
            ys = np.array([stats_dir[k]["mean"] for k in ks])
            errs = np.array([stats_dir[k]["stderr"] for k in ks])
            plot_with_band(ax1, ks, ys, errs, color, rf"$\sigma={sigma_val:.3f}$", "o")

            stats_gm = collect_nested_metric(
                rows,
                lambda row, s=sigma_key: row["sigma_subspace_sweep"][s]["gm_value"],
            )
            ys_g = np.array([stats_gm[k]["mean"] for k in ks])
            errs_g = np.array([stats_gm[k]["stderr"] for k in ks])
            plot_with_band(ax2, ks, ys_g, errs_g, color, rf"$\sigma={sigma_val:.3f}$", "s")
        elif "delta2_sigma_sweep" in rows[0] and sigma_key in rows[0]["delta2_sigma_sweep"]:
            stats = collect_nested_metric(
                rows,
                lambda row, s=sigma_key: row["delta2_sigma_sweep"][s]["mean"],
            )
            ks = np.array(sorted(stats.keys()))
            ys = np.array([stats[k]["mean"] for k in ks])
            errs = np.array([stats[k]["stderr"] for k in ks])
            plot_with_band(ax1, ks, ys, errs, color, rf"$\sigma={sigma_val:.3f}$", "o")

    ax1.set_xlabel(r"Training set size $k$")
    ax1.set_ylabel(r"Direct $\Delta_2^{(D)}$")
    ax1.set_title("Direct subspace MC", fontsize=9)
    ax1.legend(fontsize=6, framealpha=0.8)
    ax1.grid(True, which="both", alpha=0.25)

    ax2.set_xlabel(r"Training set size $k$")
    ax2.set_ylabel(r"GM estimator")
    ax2.set_title("Gaussian-moment proxy", fontsize=9)
    ax2.legend(fontsize=6, framealpha=0.8)
    ax2.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Figure 7: overlap.pdf ────────────────────────────────────────
def plot_overlap(rows, out_path):
    dims = sorted(int(x) for x in rows[0].get("eigenspace_drift", {}).keys())
    if not dims:
        return
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for i, D in enumerate(dims):
        color = cmap(i / max(len(dims) - 1, 1))
        stats = collect_nested_metric(
            rows, lambda row, D=D: row["eigenspace_drift"][str(D)]
        )
        ks = np.array(sorted(stats.keys()))
        ys = np.array([stats[k]["mean"] for k in ks])
        errs = np.array([stats[k]["stderr"] for k in ks])
        ax.plot(ks, ys, "o-", color=color, label=rf"$D={D}$", ms=4, lw=1.4)
        ax.fill_between(ks, np.maximum(ys - errs, 0), np.minimum(ys + errs, 1),
                         alpha=0.15, color=color)
    ax.set_xlabel(r"Training set size $k$")
    ax.set_ylabel("Mean principal cosine overlap")
    ax.set_xscale("log")
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Figure 8: cross_setting.pdf ──────────────────────────────────
def plot_cross_setting(runs, out_path, D=10):
    settings = available_settings(runs)
    if len(settings) < 2:
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        rows_all = setting_runs(runs, settings[0])
        agg = aggregate_by_k(rows_all)
        ks = np.array(sorted(agg.keys()))
        d2s = np.array([agg[k].get(f"d2s_{D}_mean", np.nan) for k in ks])
        d2se = np.array([agg[k].get(f"d2s_{D}_stderr", 0.0) for k in ks])
        plot_with_band(ax, ks, d2s, d2se, COLORS["d2s"], settings[0], "^")
        finish(fig, ax, r"Training set size $k$", rf"$\Delta_2^{{({D})}}$ (est.)", out_path)
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    cmap = plt.cm.Set1
    for ax_idx, (metric_label, metric_fn) in enumerate([
        (rf"$\Delta_2^{{({D})}}$",
         lambda row, D=D: row["delta2_subspace"][str(D)]),
        (r"$\Delta_1$",
         lambda row: row["delta1"]),
    ]):
        ax = axes[ax_idx]
        for i, sname in enumerate(settings):
            color = cmap(i / max(len(settings) - 1, 1))
            rows_s = setting_runs(runs, sname)
            stats = collect_nested_metric(rows_s, metric_fn)
            ks = np.array(sorted(stats.keys()))
            ys = np.array([stats[k]["mean"] for k in ks])
            errs = np.array([stats[k]["stderr"] for k in ks])
            plot_with_band(ax, ks, ys, errs, color, sname.replace("_", " "), "o")
        ax.set_xlabel(r"Training set size $k$")
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=6, framealpha=0.8)
        ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"  -> {out_path}")


# ── Table data: direct vs proxy agreement ────────────────────────────────────
def compute_proxy_table(rows, D=10):
    """Return table rows for direct-vs-proxy agreement."""
    table_rows = []
    from scipy.stats import pearsonr

    direct_vals = []
    gm_vals = []
    qmc_vals = []
    for row in rows:
        d = row["delta2_subspace"].get(str(D))
        g = row.get("gm_estimator")
        q = row.get("quadMC_estimator", {}).get("mean")
        if d is not None and g is not None and q is not None:
            direct_vals.append(d)
            gm_vals.append(g)
            qmc_vals.append(q)

    if len(direct_vals) >= 3:
        d_arr = np.array(direct_vals)
        g_arr = np.array(gm_vals)
        q_arr = np.array(qmc_vals)

        corr_qmc, _ = pearsonr(d_arr, q_arr)
        corr_gm, _ = pearsonr(d_arr, g_arr)
        re_qmc = np.abs(d_arr.mean() - q_arr.mean()) / max(d_arr.mean(), 1e-15)
        re_gm = np.abs(d_arr.mean() - g_arr.mean()) / max(d_arr.mean(), 1e-15)

        ks = sorted(set(r["k"] for r in rows))
        if len(ks) >= 3:
            dk = []; dq = []; dg = []
            for k_val in ks:
                krows = [r for r in rows if r["k"] == k_val]
                dk.append(np.mean([r["delta2_subspace"][str(D)] for r in krows]))
                dg.append(np.mean([r["gm_estimator"] for r in krows]))
                dq.append(np.mean([r["quadMC_estimator"]["mean"] for r in krows]))
            log_k = np.log(np.array(ks, dtype=float))
            slope_d = np.polyfit(log_k, np.log(np.maximum(np.array(dk), 1e-15)), 1)[0]
            slope_q = np.polyfit(log_k, np.log(np.maximum(np.array(dq), 1e-15)), 1)[0]
            slope_g = np.polyfit(log_k, np.log(np.maximum(np.array(dg), 1e-15)), 1)[0]
        else:
            slope_d = slope_q = slope_g = 0.0

        table_rows.append({
            "pair": "Direct vs quadMC",
            "correlation": corr_qmc,
            "relative_error": re_qmc,
            "slope_difference": slope_q - slope_d,
        })
        table_rows.append({
            "pair": "Direct vs GM",
            "correlation": corr_gm,
            "relative_error": re_gm,
            "slope_difference": slope_g - slope_d,
        })
    return table_rows


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    default_json = default_json_path()
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_json
    out_dir = _FIGURES

    if not json_path.is_file():
        print(f"Missing {json_path}; run: python code/run_experiments.py", file=sys.stderr)
        raise SystemExit(1)

    runs = load_runs(json_path)
    main_setting_name = primary_setting(runs)
    primary_rows = setting_runs(runs, main_setting_name)
    D = default_main_dim(primary_rows)

    print(f"Primary setting: {main_setting_name}, main D={D}")

    print("Figure 1: reference scaling")
    plot_reference_scaling(primary_rows, out_dir / "reference_scaling.pdf", D=D)

    print("Figure 2: full vs subspace")
    plot_full_vs_subspace(primary_rows, out_dir / "reference_full_vs_subspace.pdf", D=D)

    print("Figure 3: subspace comparison")
    plot_subspace_comparison(primary_rows, out_dir / "subspace_comparison.pdf", D=D)

    print("Figure 4: direct vs proxy")
    plot_direct_vs_proxy(primary_rows, out_dir / "direct_vs_proxy_curves.pdf", D=D)

    print("Figure 5: D sweep")
    plot_D_sweep(primary_rows, out_dir / "D_sweep.pdf")

    print("Figure 6: sigma sweep")
    plot_sigma_sweep(primary_rows, out_dir / "sigma_sweep.pdf")

    print("Figure 7: eigenspace overlap")
    plot_overlap(primary_rows, out_dir / "overlap.pdf")

    print("Figure 8: cross-setting")
    plot_cross_setting(runs, out_dir / "cross_setting.pdf", D=D)

    try:
        table = compute_proxy_table(primary_rows, D=D)
        if table:
            print("\nTable: Direct vs Proxy agreement")
            for row in table:
                print(f"  {row['pair']}: corr={row['correlation']:.3f}, "
                      f"rel_err={row['relative_error']:.3f}, "
                      f"slope_diff={row['slope_difference']:.3f}")
    except ImportError:
        print("scipy not available; skipping proxy table computation")

    print("\nAll figures written to", out_dir)


if __name__ == "__main__":
    main()
