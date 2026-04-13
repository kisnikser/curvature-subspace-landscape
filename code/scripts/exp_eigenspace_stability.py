"""
Experiment 4: eigenspace stability of H^(k) vs H^(k+1).

For a fixed checkpoint w_k*, computes the top-D eigenvectors of the empirical
Hessian H^(k)(w_k*) and H^(k+1)(w_k*) for varying dataset sizes k, then measures
the principal-angle overlap between the two D-dimensional subspaces.

Metric (subspace overlap):
  overlap(D) = (1/D) * ||U_D^{(k)T} U_D^{(k+1)}||_F^2  in [0, 1]
  = 1  =>  subspaces are identical  (Assumption 1 fully satisfied)
  = 0  =>  subspaces are orthogonal

Also reports: grassmann_dist = sqrt(D - ||U_D^{(k)T} U_D^{(k+1)}||_F^2),
which equals ||sin(principal angles)||_F and is a proper metric on Gr(D,N).

Run for multiple k values and multiple subspace dimensions D to produce a
(k x D) heatmap and per-k line plots.

Usage (from code/):
  python -m scripts.exp_eigenspace_stability \\
      --model-tag d6 --model-step 3500 \\
      --k-values 2,4,8,16,32 \\
      --subspace-dims 1,5,10,45 \\
      --seeds 0,1,2 \\
      --out .nanochat/reports/eigenspace_stability.json

Output JSON structure:
  {
    "meta": { ... },
    "results": [
      {
        "k": 8, "D": 10,
        "seeds": [
          {"seed": 0, "overlap": 0.97, "grassmann_dist": 0.17,
           "eigenvalues_k": [...], "eigenvalues_k1": [...]},
          ...
        ],
        "overlap_mean": 0.97, "overlap_std": 0.01,
        "grassmann_dist_mean": 0.17, "grassmann_dist_std": 0.02
      },
      ...
    ]
  }

Plot: python -m scripts.plot_eigenspace_stability --json <output.json>
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from itertools import product
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from scripts.delta2_subspace_estimators import (
    build_batches_first_k_sequences,
    compute_top_eigenvectors,
    trim_batch,
    _flat_params,
    _set_flat_params,
)


# ---------------------------------------------------------------------------
# Subspace overlap metrics
# ---------------------------------------------------------------------------


def subspace_overlap(U1: torch.Tensor, U2: torch.Tensor) -> tuple[float, float]:
    """
    Compute overlap and Grassmann distance between two D-dim subspaces.

    Args:
        U1: (N, D) orthonormal basis for subspace 1
        U2: (N, D) orthonormal basis for subspace 2

    Returns:
        overlap: (1/D) * ||U1^T U2||_F^2  in [0, 1]
        grassmann_dist: sqrt(D - ||U1^T U2||_F^2)  >= 0
    """
    D = U1.shape[1]
    # Cross-Gram matrix: (D, D)
    G = U1.T @ U2  # (D, D)
    fro_sq = float(torch.sum(G * G).item())
    overlap = fro_sq / D
    grassmann_dist = float((max(D - fro_sq, 0.0)) ** 0.5)
    return overlap, grassmann_dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def _sdp_math(device: torch.device):
    if device.type == "cuda":
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def _run_one_cell(
    *,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    w_checkpoint: torch.Tensor,
    k: int,
    D: int,
    msl: int,
    device_batch_size: int,
    eig_iters: int,
    eig_tol: float,
    hess_max_seq: int,
    hvp_microbatch: int,
) -> dict:
    """Compute eigenspace overlap between H^(k) and H^(k+1) at w_checkpoint."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Load k+1 sequences from training data
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_batch_size, msl, split="train", device=device
    )
    x_k, y_k, x_k1, y_k1 = build_batches_first_k_sequences(iter(train_loader), k, device)

    # Trim for Hessian computation
    xh_k, yh_k = trim_batch(x_k, y_k, hess_max_seq)
    xh_k1, yh_k1 = trim_batch(x_k1, y_k1, hess_max_seq)

    # Compute top-D eigenvectors of H^(k)(w*)
    _set_flat_params(model, w_checkpoint.clone())
    eigs_k, U_k = compute_top_eigenvectors(
        model, xh_k, yh_k, D, eig_iters, eig_tol, device, dtype, hvp_microbatch
    )

    # Compute top-D eigenvectors of H^(k+1)(w*)
    _set_flat_params(model, w_checkpoint.clone())
    eigs_k1, U_k1 = compute_top_eigenvectors(
        model, xh_k1, yh_k1, D, eig_iters, eig_tol, device, dtype, hvp_microbatch
    )

    # Restore checkpoint weights
    _set_flat_params(model, w_checkpoint)

    overlap, grassmann_dist = subspace_overlap(U_k, U_k1)

    return {
        "seed": seed,
        "overlap": overlap,
        "grassmann_dist": grassmann_dist,
        "eigenvalues_k": eigs_k.cpu().tolist(),
        "eigenvalues_k1": eigs_k1.cpu().tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eigenspace stability of H^(k) vs H^(k+1) across k and D (Experiment 4)."
    )
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-batch-size", type=int, default=16, help="must be > max(k-values)")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--k-values",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated k values (dataset sizes)",
    )
    parser.add_argument(
        "--subspace-dims",
        type=str,
        default="1,5,10,45",
        help="Comma-separated subspace dimensions D to evaluate",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--eig-iters", type=int, default=30)
    parser.add_argument("--eig-tol", type=float, default=1e-4)
    parser.add_argument("--hess-max-seq", type=int, default=4)
    parser.add_argument("--hvp-microbatch", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    k_values = [int(s.strip()) for s in args.k_values.split(",") if s.strip()]
    subspace_dims = [int(s.strip()) for s in args.subspace_dims.split(",") if s.strip()]

    if not k_values:
        raise SystemExit("--k-values must contain at least one value")
    if not subspace_dims:
        raise SystemExit("--subspace-dims must contain at least one value")
    max_D = max(subspace_dims)
    max_k = max(k_values)

    model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.model_step)
    msl = int(args.max_seq_len or meta["model_config"]["sequence_len"])
    w_checkpoint = _flat_params(model).clone()

    results_list: list[dict] = []
    total_cells = len(k_values) * len(subspace_dims)
    cell_idx = 0

    for k, D in product(k_values, subspace_dims):
        cell_idx += 1
        print(f"[{cell_idx}/{total_cells}] k={k}, D={D} ...")

        if args.device_batch_size <= k:
            print(f"  WARNING: device_batch_size ({args.device_batch_size}) <= k ({k}); skipping.")
            continue

        seed_results: list[dict] = []
        for seed in seeds:
            t0 = time.time()
            res = _run_one_cell(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dtype=dtype,
                seed=seed,
                w_checkpoint=w_checkpoint,
                k=k,
                D=D,
                msl=msl,
                device_batch_size=args.device_batch_size,
                eig_iters=args.eig_iters,
                eig_tol=args.eig_tol,
                hess_max_seq=args.hess_max_seq,
                hvp_microbatch=args.hvp_microbatch,
            )
            elapsed = time.time() - t0
            print(f"  seed={seed}: overlap={res['overlap']:.4f}, grassmann_dist={res['grassmann_dist']:.4f}  ({elapsed:.1f}s)")
            seed_results.append(res)

        overlaps = [r["overlap"] for r in seed_results]
        gdists = [r["grassmann_dist"] for r in seed_results]
        cell = {
            "k": k,
            "D": D,
            "seeds": seed_results,
            "overlap_mean": statistics.fmean(overlaps),
            "overlap_std": statistics.stdev(overlaps) if len(overlaps) > 1 else 0.0,
            "grassmann_dist_mean": statistics.fmean(gdists),
            "grassmann_dist_std": statistics.stdev(gdists) if len(gdists) > 1 else 0.0,
        }
        results_list.append(cell)

    output = {
        "meta": {
            "model_tag": args.model_tag or "auto",
            "model_step": args.model_step or "auto",
            "k_values": k_values,
            "subspace_dims": subspace_dims,
            "seeds": seeds,
            "eig_iters": args.eig_iters,
            "eig_tol": args.eig_tol,
            "hess_max_seq": args.hess_max_seq,
        },
        "results": results_list,
    }

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"eigenspace_stability_{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
