"""
Wall-clock phases for Δ_2^(D) estimators (paper Sec.~\\ref{sec:algo}).

Runs optional ``--warmup`` full pipeline passes (discarded), then ``--repeats`` timed
runs (seeds ``seed``, ``seed+1``, …), aggregates mean ± sample std (ddof=1) for each
cell, writes JSON and a LaTeX table:

  rows: Direct MC, Quadratic MC, Gaussian moment
  cols: Stage I, II, III  (Stage II is N/A for Direct MC)

  cd code
  python -m scripts.benchmark_delta2_estimator_phases --model-tag d6 --model-step 3500 \\
      --warmup 1 --repeats 5 --out .nanochat/reports/delta2_estimator_phases_run.json

Produces: <out>.json and <out_stem>.tex (same directory, .tex next to .json stem)
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from scripts.delta2_subspace_estimators import (
    build_batches_first_k_sequences,
    compress_hessian_to_basis,
    compute_top_eigenvectors,
    direct_subspace_mc,
    gaussian_moment_closed_form,
    grad_flat_lm,
    quadratic_mc_estimate,
    trim_batch,
    _compute_increment,
    _flat_params,
    _set_flat_params,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _tic(device: torch.device) -> float:
    _sync(device)
    return time.perf_counter()


def _toc(device: torch.device, t0: float) -> float:
    _sync(device)
    return time.perf_counter() - t0


def _run_once(
    *,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    w_checkpoint: torch.Tensor,
    args: argparse.Namespace,
    D: int,
    msl: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, msl, split="train", device=device
    )
    x_k, y_k, x_k1, y_k1 = build_batches_first_k_sequences(iter(train_loader), args.k_sequences, device)
    w_k = w_checkpoint.clone()
    xh, yh = trim_batch(x_k, y_k, args.hess_max_seq)
    xh1, yh1 = trim_batch(x_k1, y_k1, args.hess_max_seq)

    _set_flat_params(model, w_k)
    t0 = _tic(device)
    a_k = _compute_increment(model, x_k, y_k, x_k1, y_k1, args.microbatch)
    t_stage_II_increment = _toc(device, t0)

    _set_flat_params(model, w_k)
    t0 = _tic(device)
    eigenvalues, U_full = compute_top_eigenvectors(
        model, xh, yh, D, args.eig_iters, args.eig_tol, device, dtype, args.hvp_microbatch
    )
    t_stage_I_subspace = _toc(device, t0)

    t0 = _tic(device)
    g_k = grad_flat_lm(model, x_k, y_k, args.microbatch)
    g_k1 = grad_flat_lm(model, x_k1, y_k1, args.microbatch)
    grad_diff = g_k1 - g_k
    c_full = U_full.T @ grad_diff
    t_stage_II_gradients = _toc(device, t0)

    t0 = _tic(device)
    _set_flat_params(model, w_k)
    Hk = compress_hessian_to_basis(model, xh, yh, U_full, args.hvp_microbatch)
    _set_flat_params(model, w_k)
    Hk1 = compress_hessian_to_basis(model, xh1, yh1, U_full, args.hvp_microbatch)
    B_full = Hk1 - Hk
    t_stage_II_hessian_compress = _toc(device, t0)

    stage_II_total = t_stage_II_increment + t_stage_II_gradients + t_stage_II_hessian_compress

    c_D = c_full[:D].clone()
    B_D = B_full[:D, :D].clone()
    U_D = U_full[:, :D]

    t0 = _tic(device)
    _set_flat_params(model, w_k)
    direct_subspace_mc(
        model,
        w_k,
        U_D,
        x_k,
        y_k,
        x_k1,
        y_k1,
        float(args.sigma),
        args.num_samples,
        device,
        dtype,
        args.microbatch,
    )
    t_eval_direct_mc = _toc(device, t0)

    t0 = _tic(device)
    quadratic_mc_estimate(a_k, c_D, B_D, float(args.sigma), args.num_samples, device, dtype)
    t_eval_quadratic_mc = _toc(device, t0)

    t0 = _tic(device)
    _ = gaussian_moment_closed_form(a_k, c_D, B_D, float(args.sigma))
    t_eval_gaussian_moment = _toc(device, t0)

    return {
        "stage_I_subspace": t_stage_I_subspace,
        "stage_II_increment": t_stage_II_increment,
        "stage_II_gradients": t_stage_II_gradients,
        "stage_II_hessian_compress": t_stage_II_hessian_compress,
        "stage_II_total": stage_II_total,
        "eval_direct_mc": t_eval_direct_mc,
        "eval_quadratic_mc": t_eval_quadratic_mc,
        "eval_gaussian_moment": t_eval_gaussian_moment,
        "top_hessian_eigenvalues": eigenvalues.cpu().tolist(),
    }


def _aggregate_seconds(rows: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    keys = rows[0].keys()
    mean = {}
    std = {}
    for k in keys:
        if k == "top_hessian_eigenvalues":
            continue
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        mean[k] = float(np.mean(vals))
        std[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def _matrix_mean_std(
    means: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std per table cell from per-run mean dicts (same structure)."""
    # Build per-run matrices then mean/std per cell
    mats = []
    for m in means:
        mats.append(
            np.array(
                [
                    [m["stage_I_subspace"], np.nan, m["eval_direct_mc"]],
                    [m["stage_I_subspace"], m["stage_II_total"], m["eval_quadratic_mc"]],
                    [m["stage_I_subspace"], m["stage_II_total"], m["eval_gaussian_moment"]],
                ],
                dtype=float,
            )
        )
    A = np.stack(mats, axis=0)  # (R, 3, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M_mean = np.nanmean(A, axis=0)
        M_std = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(M_mean)
    if A.shape[0] == 1:
        M_std = np.full_like(M_mean, np.nan)
    return M_mean, M_std


def _tex_cell(mean: float, std: float) -> str:
    if np.isnan(mean):
        return r"\textemdash{}"
    if np.isnan(std):
        return f"${mean:.4g}$"
    return f"${mean:.4g} \\pm {std:.3g}$"


def _write_tex(
    path: Path,
    M_mean: np.ndarray,
    M_std: np.ndarray,
    estimator_names: list[str],
    stage_headers: list[str],
) -> None:
    lines = [
        r"% Auto-generated by benchmark_delta2_estimator_phases.py",
        r"% Wall times in seconds; mean \pm sample std over repeats (ddof=1).",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Estimator & " + " & ".join(stage_headers) + r" \\",
        r"\midrule",
    ]
    for i, name in enumerate(estimator_names):
        cells = [_tex_cell(M_mean[i, j], M_std[i, j]) for j in range(3)]
        lines.append(name.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--model-step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--k-sequences", type=int, default=8)
    p.add_argument("--subspace-dim", type=int, default=10, help="D for top-Hessian basis")
    p.add_argument("--sigma", type=float, default=1e-3)
    p.add_argument("--num-samples", type=int, default=64, help="S for direct and quadratic MC")
    p.add_argument("--eig-iters", type=int, default=20)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--hess-max-seq", type=int, default=4)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--hvp-microbatch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0, help="Base seed; repeat r uses seed+r")
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Full pipeline runs before timing (discarded; reduces cold-start/GPU bias on fast phases)",
    )
    p.add_argument("--repeats", type=int, default=5, help="Number of timed runs for mean ± std")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    D = int(args.subspace_dim)

    model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.model_step)
    msl = int(args.max_seq_len or meta["model_config"]["sequence_len"])
    w_checkpoint = _flat_params(model).clone()

    for w in range(args.warmup):
        print(f"Warmup {w + 1}/{args.warmup} (seed={args.seed + w})...")
        _set_flat_params(model, w_checkpoint.clone())
        one = _run_once(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            seed=args.seed + w,
            w_checkpoint=w_checkpoint,
            args=args,
            D=D,
            msl=msl,
        )
        one.pop("top_hessian_eigenvalues")

    raw_runs: list[dict] = []
    for rep in range(args.repeats):
        print(f"Repeat {rep + 1}/{args.repeats} (seed={args.seed + rep})...")
        _set_flat_params(model, w_checkpoint.clone())
        one = _run_once(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            seed=args.seed + rep,
            w_checkpoint=w_checkpoint,
            args=args,
            D=D,
            msl=msl,
        )
        eig = one.pop("top_hessian_eigenvalues")
        raw_runs.append(one)
        if rep == 0:
            last_eig = eig

    mean_sec, std_sec = _aggregate_seconds(raw_runs)
    M_mean, M_std = _matrix_mean_std(raw_runs)

    estimator_names = ["Direct MC", "Quadratic MC", "Gaussian moment"]
    stage_headers = [r"Stage I", r"Stage II", r"Stage III"]

    results = {
        "meta": {
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "subspace_dim": D,
            "sigma": args.sigma,
            "num_samples": args.num_samples,
            "k_sequences": args.k_sequences,
            "device": str(device),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed_base": args.seed,
            "top_hessian_eigenvalues_run0": last_eig,
        },
        "seconds_mean": mean_sec,
        "seconds_std": std_sec,
        "table_mean": M_mean.tolist(),
        "table_std": M_std.tolist(),
        "repeats_raw": raw_runs,
    }

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"delta2_estimator_phases_{int(time.time())}.json")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    tex_path = out_path.with_suffix(".tex")
    _write_tex(tex_path, M_mean, M_std, estimator_names, stage_headers)

    print(f"Wrote {out_path}")
    print(f"Wrote {tex_path}")
    print(
        f"  Stage I mean: {mean_sec['stage_I_subspace']:.4f}s ± {std_sec['stage_I_subspace']:.4f}s\n"
        f"  Stage II mean (total): {mean_sec['stage_II_total']:.4f}s ± {std_sec['stage_II_total']:.4f}s"
    )


if __name__ == "__main__":
    main()
