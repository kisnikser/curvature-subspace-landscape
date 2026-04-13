"""
Merge several run_delta_k_curriculum JSONs (different seeds) into mean ± std across seeds
at each common k, write merged JSON and PDFs with fill_between bands.

  cd code
  python -m scripts.merge_delta_k_curriculum \\
      --glob .nanochat/reports/my_run_seed*.json \\
      --out-prefix .nanochat/reports/my_run_merged

Std is sample std with ddof=1 over seeds (0 if n=1).
"""
from __future__ import annotations

import argparse
import glob
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


def _mean_std(a: np.ndarray) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return m, s


def _band_positive(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper for fill; clip lower for log-scale (positive)."""
    lo = mean - std
    hi = mean + std
    eps = np.maximum(mean * 1e-6, 1e-300)
    lo = np.maximum(lo, eps)
    return lo, hi


def merge_payload(runs: list[dict], paths: list[str]) -> dict:
    """Build merged structure; only k present in every run."""
    by_k_per: list[dict[int, dict]] = []
    for r in runs:
        m = {row["k"]: row for row in r["by_k"]}
        by_k_per.append(m)

    common_ks = set(by_k_per[0].keys())
    for m in by_k_per[1:]:
        common_ks &= set(m.keys())
    ks_sorted = sorted(common_ks)

    meta0 = runs[0]["meta"]
    seeds = [r["meta"].get("seed") for r in runs]

    merged_rows: list[dict] = []
    for k in ks_sorted:
        rows_at_k = [m[k] for m in by_k_per]

        d1_vals = np.array([x["delta1"]["mean"] for x in rows_at_k])
        d1_m, d1_s = _mean_std(d1_vals)

        d2_full: list[dict] = []
        sigmas = [float(x["sigma"]) for x in rows_at_k[0]["delta2_fullspace"]]
        for sig in sigmas:
            vals = []
            for x in rows_at_k:
                ent = next(
                    e
                    for e in x["delta2_fullspace"]
                    if math.isclose(float(e["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
                )
                vals.append(ent["mean"])
            vm, vs = _mean_std(np.array(vals))
            d2_full.append({"sigma": sig, "mean": vm, "std_across_seeds": vs, "per_seed_mean": vals})

        d2_sub: list[dict] = []
        for ent0 in rows_at_k[0]["delta2_subspace_gaussian"]:
            D0 = int(ent0["D"])
            s0 = float(ent0["sigma"])
            vals = []
            for x in rows_at_k:
                ent = next(
                    e
                    for e in x["delta2_subspace_gaussian"]
                    if int(e["D"]) == D0
                    and math.isclose(float(e["sigma"]), s0, rel_tol=1e-9, abs_tol=1e-15)
                )
                vals.append(ent["gaussian_moment"])
            vm, vs = _mean_std(np.array(vals))
            d2_sub.append(
                {"D": D0, "sigma": s0, "mean": vm, "std_across_seeds": vs, "per_seed": vals}
            )

        ce_vals = np.array([x["val_loss_mean_ce"] for x in rows_at_k])
        ppl_vals = np.array([x["val_ppl"] for x in rows_at_k])
        ce_m, ce_s = _mean_std(ce_vals)
        ppl_m, ppl_s = _mean_std(ppl_vals)

        merged_rows.append(
            {
                "k": k,
                "delta1": {
                    "mean": d1_m,
                    "std_across_seeds": d1_s,
                    "per_seed_mean": d1_vals.tolist(),
                },
                "delta2_fullspace": d2_full,
                "delta2_subspace_gaussian": d2_sub,
                "val_loss_mean_ce": ce_m,
                "val_ce_std_across_seeds": ce_s,
                "val_ppl": ppl_m,
                "val_ppl_std_across_seeds": ppl_s,
                "per_seed_val_ce": ce_vals.tolist(),
                "per_seed_val_ppl": ppl_vals.tolist(),
            }
        )

    return {
        "meta": {
            "merged_from": paths,
            "seeds": seeds,
            "n_runs": len(runs),
            "k_values": ks_sorted,
            "depth": meta0.get("depth"),
            "sigmas": meta0.get("sigmas"),
            "D_list": meta0.get("D_list"),
            "notes": "Aggregated over seeds: mean and std (ddof=1) at each k common to all runs.",
        },
        "by_k": merged_rows,
    }


def plot_merged_criteria(data: dict, out: Path, font_path: Path, figsize: tuple[float, float]) -> None:
    _setup_times_font(font_path)
    meta = data["meta"]
    rows = sorted(data["by_k"], key=lambda r: r["k"])
    ks = np.array([r["k"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ (sequences consumed in training)")
    ax.set_ylabel(r"Criteria (mean $\pm$ std across seeds)")

    d1m = np.array([r["delta1"]["mean"] for r in rows])
    d1s = np.array([r["delta1"]["std_across_seeds"] for r in rows])
    lo, hi = _band_positive(d1m, d1s)
    ax.fill_between(ks, lo, hi, alpha=0.25, color="C0")
    ax.plot(ks, d1m, marker="o", linewidth=1.6, color="C0", label=r"$\Delta_1$")

    sigmas = meta.get("sigmas") or sorted(
        {float(x["sigma"]) for r in rows for x in r["delta2_fullspace"]}
    )
    for i, sig in enumerate(sigmas):
        c = f"C{(i + 1) % 10}"
        ym = []
        ys = []
        for r in rows:
            m = next(
                x
                for x in r["delta2_fullspace"]
                if math.isclose(float(x["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
            )
            ym.append(m["mean"])
            ys.append(m["std_across_seeds"])
        ym = np.array(ym)
        ys = np.array(ys)
        lo, hi = _band_positive(ym, ys)
        ax.fill_between(ks, lo, hi, alpha=0.2, color=c)
        ax.plot(
            ks,
            ym,
            marker="s",
            linewidth=1.4,
            linestyle="--",
            color=c,
            label=rf"$\Delta_2$, $\sigma={_sigma_label(float(sig))}$",
        )

    d_list = meta.get("D_list") or [1, 4, 16]
    markers = [".", "^", "v", "D", "P"]
    mi = 0
    base_ci = len(sigmas) + 2
    for D in sorted(set(d_list)):
        for sig in sigmas:
            ym = []
            ys = []
            for r in rows:
                ent = next(
                    e
                    for e in r["delta2_subspace_gaussian"]
                    if int(e["D"]) == int(D)
                    and math.isclose(float(e["sigma"]), float(sig), rel_tol=1e-9, abs_tol=1e-15)
                )
                ym.append(ent["mean"])
                ys.append(ent["std_across_seeds"])
            ym = np.array(ym)
            ys = np.array(ys)
            c = f"C{(base_ci + mi) % 10}"
            lo, hi = _band_positive(ym, ys)
            ax.fill_between(ks, lo, hi, alpha=0.2, color=c)
            mkr = markers[mi % len(markers)]
            ax.plot(
                ks,
                ym,
                marker=mkr,
                linewidth=1.2,
                linestyle="-.",
                markersize=5,
                color=c,
                label=rf"$\Delta_2^{{({D})}}$, $\sigma={_sigma_label(float(sig))}$ (GM)",
            )
            mi += 1

    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best", fontsize=5, ncol=2, framealpha=0.92)
    tag = meta.get("depth", "?")
    n = meta.get("n_runs", "?")
    ax.set_title(f"Landscape criteria vs $k$ (depth {tag}, n={n} seeds)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_merged_val(data: dict, out: Path, font_path: Path, figsize: tuple[float, float]) -> None:
    _setup_times_font(font_path)
    meta = data["meta"]
    rows = sorted(data["by_k"], key=lambda r: r["k"])
    ks = np.array([r["k"] for r in rows], dtype=float)
    ce_m = np.array([r["val_loss_mean_ce"] for r in rows])
    ce_s = np.array([r["val_ce_std_across_seeds"] for r in rows])
    ppl_m = np.array([r["val_ppl"] for r in rows])
    ppl_s = np.array([r["val_ppl_std_across_seeds"] for r in rows])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ (sequences consumed in training)")
    ax.set_ylabel("Validation CE / PPL (mean ± std across seeds)")
    ax.fill_between(ks, ce_m - ce_s, ce_m + ce_s, alpha=0.25, color="C0")
    ax.plot(ks, ce_m, marker="o", linewidth=1.8, color="C0", label="val CE (nats)")
    ax.fill_between(ks, np.maximum(ppl_m - ppl_s, 1e-6), ppl_m + ppl_s, alpha=0.25, color="C1")
    ax.plot(ks, ppl_m, marker="s", linewidth=1.8, color="C1", label="val PPL")
    ax.grid(True, which="major", alpha=0.2)
    ax.legend(loc="best", fontsize=10)
    ax.set_title(f"Validation metrics vs $k$ (depth {meta.get('depth', '?')}, n={meta.get('n_runs', '?')} seeds)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        dest="glob_pattern",
        type=str,
        default=None,
        help="Glob of JSON paths, e.g. .nanochat/reports/run_seed*.json",
    )
    ap.add_argument(
        "--jsons",
        type=str,
        default=None,
        help="Comma-separated list of JSON paths (alternative to --glob)",
    )
    ap.add_argument("--out-prefix", type=str, required=True, help="Writes <prefix>.json, <prefix>_criteria.pdf, <prefix>_val.pdf")
    ap.add_argument("--font", type=str, default=str(_default_times_font_path()))
    ap.add_argument("--figsize", type=str, default="7,4.5")
    args = ap.parse_args()

    if args.glob_pattern:
        paths = sorted(glob.glob(args.glob_pattern))
    elif args.jsons:
        paths = [p.strip() for p in args.jsons.split(",") if p.strip()]
    else:
        raise SystemExit("Provide --glob or --jsons")

    if len(paths) < 1:
        raise SystemExit("No JSON files matched")

    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))

    payload = merge_payload(runs, paths)
    prefix = Path(args.out_prefix)
    out_json = prefix.with_suffix(".json") if prefix.suffix != ".json" else prefix
    if not str(out_json).endswith(".json"):
        out_json = Path(str(out_json) + ".json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    w, h = (float(x) for x in args.figsize.split(","))
    fp = Path(args.font)
    stem = out_json.with_suffix("")
    plot_merged_criteria(payload, Path(str(stem) + "_criteria.pdf"), fp, (w, h))
    plot_merged_val(payload, Path(str(stem) + "_val.pdf"), fp, (w, h))
    print(f"Wrote {out_json}")
    print(f"Wrote {stem}_criteria.pdf")
    print(f"Wrote {stem}_val.pdf")


if __name__ == "__main__":
    main()
