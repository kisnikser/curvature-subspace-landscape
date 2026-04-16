"""
Collect data for Δ_2^(D) robustness plots:

  1) Heatmap: |direct - GM| / (|direct|+eps) over (σ, D)
  2) S-curve: direct & quadratic MC vs S at fixed (σ, D); GM horizontal line

Writes JSON for scripts/plot_delta2_robustness.py

  cd code
  python -m scripts.run_delta2_robustness --model-tag d6 --model-step 3500 --out .nanochat/reports/delta2_robustness.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def _default_D_list_1_to_100() -> list[int]:
    """Ten approximately uniform integers from 1 to 100 (inclusive)."""
    return sorted({int(round(x)) for x in np.linspace(1, 100, 10)})

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--model-step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--k-sequences", type=int, default=8)
    p.add_argument("--eig-iters", type=int, default=20)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--hess-max-seq", type=int, default=4)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--hvp-microbatch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--D-list",
        type=str,
        default="auto",
        help="comma-separated subspace dimensions, or 'auto' for ~10 values from 1..100 (max sets eigen basis size)",
    )
    p.add_argument("--sigma-log-min", type=float, default=1e-7)
    p.add_argument("--sigma-log-max", type=float, default=1e-1)
    p.add_argument("--sigma-num", type=int, default=10)
    p.add_argument("--heatmap-samples", type=int, default=24, help="MC samples per (sigma,D) cell")
    p.add_argument("--s-curve-sigma", type=float, default=0.02)
    p.add_argument(
        "--s-curve-D",
        type=int,
        default=45,
        help="must appear in resolved D list (default 45 matches auto grid)",
    )
    p.add_argument("--S-list", type=str, default="8,16,32,64")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.D_list.strip().lower() == "auto":
        D_list = _default_D_list_1_to_100()
    else:
        D_list = [int(x.strip()) for x in args.D_list.split(",") if x.strip()]
    if not D_list:
        raise SystemExit("D-list is empty (use 'auto' or comma-separated integers).")
    max_D = max(D_list)
    sigmas = np.logspace(
        np.log10(args.sigma_log_min), np.log10(args.sigma_log_max), args.sigma_num, dtype=np.float64
    ).tolist()
    S_list = [int(x.strip()) for x in args.S_list.split(",") if x.strip()]

    model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.model_step)
    msl = int(args.max_seq_len or meta["model_config"]["sequence_len"])

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, msl, split="train", device=device
    )
    x_k, y_k, x_k1, y_k1 = build_batches_first_k_sequences(iter(train_loader), args.k_sequences, device)
    w_k = _flat_params(model).clone()
    xh, yh = trim_batch(x_k, y_k, args.hess_max_seq)
    xh1, yh1 = trim_batch(x_k1, y_k1, args.hess_max_seq)

    t0 = time.time()
    _set_flat_params(model, w_k)
    a_k = _compute_increment(model, x_k, y_k, x_k1, y_k1, args.microbatch)

    print(f"Top-{max_D} Hessian eigenvectors...")
    _set_flat_params(model, w_k)
    eigenvalues, U_full = compute_top_eigenvectors(
        model, xh, yh, max_D, args.eig_iters, args.eig_tol, device, dtype, args.hvp_microbatch
    )

    g_k = grad_flat_lm(model, x_k, y_k, args.microbatch)
    g_k1 = grad_flat_lm(model, x_k1, y_k1, args.microbatch)
    grad_diff = g_k1 - g_k
    c_full = U_full.T @ grad_diff

    _set_flat_params(model, w_k)
    Hk = compress_hessian_to_basis(model, xh, yh, U_full, args.hvp_microbatch)
    _set_flat_params(model, w_k)
    Hk1 = compress_hessian_to_basis(model, xh1, yh1, U_full, args.hvp_microbatch)
    B_full = Hk1 - Hk

    eps = 1e-12
    heatmap = {
        "sigmas": sigmas,
        "D_list": D_list,
        "rel_err_direct_gm": [],
        "direct_mean": [],
        "gm": [],
    }
    print("Heatmap (sigma x D)...")
    for si, sigma in enumerate(sigmas):
        heatmap["rel_err_direct_gm"].append([])
        heatmap["direct_mean"].append([])
        heatmap["gm"].append([])
        for D in D_list:
            c_k = c_full[:D].clone()
            B_k = B_full[:D, :D].clone()
            gm = gaussian_moment_closed_form(a_k, c_k, B_k, float(sigma))
            U_D = U_full[:, :D]
            _set_flat_params(model, w_k)
            direct = direct_subspace_mc(
                model,
                w_k,
                U_D,
                x_k,
                y_k,
                x_k1,
                y_k1,
                float(sigma),
                args.heatmap_samples,
                device,
                dtype,
                args.microbatch,
            )
            dm = direct["mean"]
            rel = abs(dm - gm) / (abs(dm) + eps)
            heatmap["rel_err_direct_gm"][-1].append(rel)
            heatmap["direct_mean"][-1].append(dm)
            heatmap["gm"][-1].append(gm)
            print(f"  sigma={sigma:.2e} D={D} direct={dm:.4e} gm={gm:.4e} rel={rel:.4e}")

    s_curve = {
        "sigma": args.s_curve_sigma,
        "D": args.s_curve_D,
        "S_list": S_list,
        "gm": None,
        "direct_mean": [],
        "direct_stderr": [],
        "qmc_mean": [],
        "qmc_stderr": [],
    }
    D_sc = args.s_curve_D
    if D_sc not in D_list:
        raise SystemExit(f"--s-curve-D={D_sc} must be in --D-list {D_list}")
    c_sc = c_full[:D_sc].clone()
    B_sc = B_full[:D_sc, :D_sc].clone()
    U_sc = U_full[:, :D_sc]
    gm_sc = gaussian_moment_closed_form(a_k, c_sc, B_sc, args.s_curve_sigma)
    s_curve["gm"] = gm_sc

    print(f"S-curve at sigma={args.s_curve_sigma}, D={D_sc}...")
    for S in S_list:
        _set_flat_params(model, w_k)
        d = direct_subspace_mc(
            model,
            w_k,
            U_sc,
            x_k,
            y_k,
            x_k1,
            y_k1,
            args.s_curve_sigma,
            S,
            device,
            dtype,
            args.microbatch,
        )
        q = quadratic_mc_estimate(a_k, c_sc, B_sc, args.s_curve_sigma, S, device, dtype)
        s_curve["direct_mean"].append(d["mean"])
        s_curve["direct_stderr"].append(d["stderr"])
        s_curve["qmc_mean"].append(q["mean"])
        s_curve["qmc_stderr"].append(q["stderr"])
        print(f"  S={S} direct={d['mean']:.4e} qmc={q['mean']:.4e}")

    results = {
        "meta": {
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "k_sequences": args.k_sequences,
            "heatmap_samples": args.heatmap_samples,
            "seconds": time.time() - t0,
            "top_hessian_eigenvalues": eigenvalues.cpu().tolist(),
        },
        "heatmap": heatmap,
        "s_curve": s_curve,
    }

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"delta2_robustness_{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
