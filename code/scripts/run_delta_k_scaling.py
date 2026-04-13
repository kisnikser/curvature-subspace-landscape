"""
Single-run sweep at a **fixed** pretrained checkpoint: Δ_1, full-space Δ_2 (per σ),
Δ_2^(D) via Gaussian closed form (per σ, D), and validation CE / PPL vs nested subset size k
on **one** train batch (no online training).

For **training along the data stream** and evaluating at each k, see
``scripts.run_delta_k_curriculum`` instead.

Uses one fixed train batch (B >= max_k + 1) so L_k, L_{k+1} are nested for increasing k.
JSON layout supports later merging multiple runs for mean ± std over seeds.

  cd code
  python -m scripts.run_delta_k_scaling --model-tag d6 --model-step 3500 \\
      --device-batch-size 256 --k-grid 8,16,32,64,128 \\
      --out .nanochat/reports/delta_k_scaling_d6_3500.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

from scripts.delta_criteria_k import compute_delta1, compute_delta2_fullspace
from scripts.delta2_subspace_estimators import (
    _compute_increment,
    _flat_params,
    _mean_ce,
    _set_flat_params,
    compress_hessian_to_basis,
    compute_top_eigenvectors,
    gaussian_moment_closed_form,
    grad_flat_lm,
    trim_batch,
)


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


@torch.no_grad()
def _eval_validation(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    msl: int,
    batch_size: int,
    microbatch: int,
    num_batches: int,
) -> dict[str, float]:
    """Mean token CE (nats) and perplexity exp(CE) on val split."""
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, msl, split="val", device=device
    )
    it = iter(loader)
    total = torch.zeros((), device=device, dtype=torch.float64)
    ntok = 0
    for _ in range(num_batches):
        x, y = next(it)
        nb = x.numel()
        total = total + _mean_ce(model, x, y, microbatch).double() * nb
        ntok += nb
    mean_ce = float((total / max(ntok, 1)).item())
    return {"val_loss_mean_ce": mean_ce, "val_ppl": math.exp(mean_ce)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--model-step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=256, help="B; must exceed max k by +1")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument(
        "--k-grid",
        type=str,
        default="8,16,32,64,128",
        help="Comma-separated k values (nested subsets from one batch)",
    )
    p.add_argument(
        "--sigmas",
        type=str,
        default="1e-4,1e-3,1e-2",
        help="Comma-separated σ for full-space Δ_2 and subspace Δ_2^(D)",
    )
    p.add_argument("--D-list", type=str, default="1,4,16", help="Subspace dimensions for Δ_2^(D)")
    p.add_argument("--delta1-eps", type=float, default=0.02)
    p.add_argument("--delta1-directions", type=int, default=50)
    p.add_argument("--num-samples-delta2", type=int, default=32, help="MC samples for full-space Δ_2")
    p.add_argument("--eig-iters", type=int, default=30)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--hess-max-seq", type=int, default=4)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--hvp-microbatch", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    k_list = _parse_int_list(args.k_grid)
    sigmas = _parse_float_list(args.sigmas)
    d_list = _parse_int_list(args.D_list)
    if not k_list:
        raise SystemExit("empty --k-grid")
    if not sigmas:
        raise SystemExit("empty --sigmas")
    if not d_list:
        raise SystemExit("empty --D-list")
    max_k = max(k_list)
    d_max = max(d_list)
    if args.device_batch_size <= max_k:
        raise SystemExit(f"Need --device-batch-size > max k ({max_k})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    t_load = time.time()
    model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.model_step)
    msl = int(args.max_seq_len or meta["model_config"]["sequence_len"])

    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, msl, split="train", device=device
    )
    x_full, y_full = next(iter(train_loader))
    if x_full.shape[0] < max_k + 1:
        raise SystemExit(f"Batch rows {x_full.shape[0]} < max_k+1={max_k+1}")

    w_checkpoint = _flat_params(model).clone()
    _set_flat_params(model, w_checkpoint)

    val_stats = _eval_validation(
        model,
        tokenizer,
        device,
        msl,
        args.val_batch_size,
        args.microbatch,
        args.val_batches,
    )

    rows: list[dict] = []
    for k in sorted(set(k_list)):
        t_k = time.time()
        x_k, y_k = x_full[:k].clone(), y_full[:k].clone()
        x_k1, y_k1 = x_full[: k + 1].clone(), y_full[: k + 1].clone()

        _set_flat_params(model, w_checkpoint.clone())
        w_k = _flat_params(model).clone()

        d1 = compute_delta1(
            model,
            w_k,
            x_k,
            y_k,
            x_k1,
            y_k1,
            args.delta1_eps,
            args.delta1_directions,
            device,
            dtype,
            args.microbatch,
        )

        d2_by_sigma: list[dict] = []
        for sig in sigmas:
            _set_flat_params(model, w_checkpoint.clone())
            w_k2 = _flat_params(model).clone()
            st = compute_delta2_fullspace(
                model,
                w_k2,
                x_k,
                y_k,
                x_k1,
                y_k1,
                sig,
                args.num_samples_delta2,
                device,
                dtype,
                args.microbatch,
            )
            d2_by_sigma.append({"sigma": sig, **st})

        _set_flat_params(model, w_checkpoint.clone())
        w_k = _flat_params(model).clone()
        xh, yh = trim_batch(x_k, y_k, args.hess_max_seq)
        xh1, yh1 = trim_batch(x_k1, y_k1, args.hess_max_seq)

        a_k = _compute_increment(model, x_k, y_k, x_k1, y_k1, args.microbatch)

        _set_flat_params(model, w_k)
        _, U_full = compute_top_eigenvectors(
            model,
            xh,
            yh,
            d_max,
            args.eig_iters,
            args.eig_tol,
            device,
            dtype,
            args.hvp_microbatch,
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

        d2d_entries: list[dict] = []
        for D in sorted(set(d_list)):
            if D > d_max:
                continue
            c_D = c_full[:D].clone()
            B_D = B_full[:D, :D].clone()
            for sig in sigmas:
                gm = gaussian_moment_closed_form(a_k, c_D, B_D, float(sig))
                d2d_entries.append({"D": D, "sigma": sig, "gaussian_moment": float(gm)})

        rows.append(
            {
                "k": k,
                "seconds": time.time() - t_k,
                "delta1": d1,
                "delta2_fullspace": d2_by_sigma,
                "delta2_subspace_gaussian": d2d_entries,
            }
        )

    results = {
        "meta": {
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "seed": args.seed,
            "device": str(device),
            "device_batch_size": args.device_batch_size,
            "k_grid": sorted(set(k_list)),
            "sigmas": sigmas,
            "D_list": sorted(set(d_list)),
            "delta1_eps": args.delta1_eps,
            "delta1_directions": args.delta1_directions,
            "num_samples_delta2_fullspace": args.num_samples_delta2,
            "eig_top_dim": d_max,
            "hess_max_seq": args.hess_max_seq,
            "seconds_load_and_val": time.time() - t_load,
            "notes": (
                "Nested k from one train batch; model checkpoint fixed. "
                "val_loss / val_ppl are independent of k (same weights). "
                "delta2_subspace_gaussian is closed-form E of quadratic surrogate for Δ_2^(D)."
            ),
        },
        "validation": val_stats,
        "by_k": rows,
    }

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"delta_k_scaling_{int(time.time())}.json")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
