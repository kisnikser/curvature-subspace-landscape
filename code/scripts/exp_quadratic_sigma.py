"""
Experiment 1: quadratic Taylor vs true loss increment for isotropic Gaussian perturbations z ~ N(0, sigma^2 I).

At parameters w (e.g. a trained checkpoint), on a fixed batch (x, y):
  - L0 = L(w)
  - Delta_true = L(w + z) - L(w)
  - Delta_quad = g^T z + 0.5 * z^T H z   with g = grad L(w), H = Hessian of L at w

For each sigma, sample several z with z ~ N(0, sigma^2 I) in the full flat parameter space,
then report relative error |Delta_true - Delta_quad| / (|Delta_true| + eps) and related stats.

Usage (single GPU recommended; full-model HVP is memory-heavy):
  cd code
  python -m scripts.exp_quadratic_sigma --model-tag d6 --sigmas 1e-4,3e-4,1e-3 --num-samples 8
  # log10-uniform σ from 1e-7 to 1e-1 (25 points by default):
  python -m scripts.exp_quadratic_sigma --model-tag d6 --sigma-log-min 1e-7 --sigma-log-max 1e-1

Outputs JSON to code/.nanochat/reports/ by default (see --out).
Use --seeds 0,1,2 to average MC stats across seeds; JSON includes aggregated_by_sigma
(mean ± std of rel_err_mean over seeds). Plot: python -m scripts.plot_quadratic_sigma --json ...

Note: nanochat GPT uses bf16 matmuls; second derivatives run with autocast disabled and float32 loss for stability.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


def _autocast_disabled(device: torch.device):
    """No autocast (stable second derivatives); CPU has no cuda.autocast."""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def _sdp_math_only(device: torch.device):
    """SDPA math kernel: Flash/MemEfficient have no 2nd derivatives (HVP)."""
    if device.type == "cuda":
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def _flat_params_float(model: torch.nn.Module) -> torch.Tensor:
    """Float32 flat copy for perturbations and Taylor (stable vs bf16 params)."""
    return torch.cat([p.detach().reshape(-1).float() for p in model.parameters() if p.requires_grad])


def _set_flat_params_float(model: torch.nn.Module, flat_f32: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.data.copy_(flat_f32[idx : idx + n].to(dtype=p.dtype).view_as(p))
        idx += n


def _loss_scalar(model, x, y) -> torch.Tensor:
    """Mean CE loss (same as training)."""
    out = model(x, y)
    return out if out.ndim == 0 else out.mean()


def quadratic_delta(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Delta_quad = g(w)^T z + 0.5 z^T H(w) z using one HVP.
    z must be same device/dtype as flat parameters (float32 recommended).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    with _sdp_math_only(device):
        with _autocast_disabled(device):
            loss = _loss_scalar(model, x, y).float()
        loss.backward(create_graph=True)
        grad_flat = torch.cat([p.grad.reshape(-1) for p in params])
        linear = grad_flat.detach().dot(z)
        zt = (grad_flat * z).sum()
        hvp_tup = torch.autograd.grad(zt, params, retain_graph=False, allow_unused=False)
        hvp_flat = torch.cat([h.reshape(-1) for h in hvp_tup])
    return linear + 0.5 * z.dot(hvp_flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None, help="override; default from checkpoint meta")
    parser.add_argument(
        "--sigmas",
        type=str,
        default="1e-4,3e-4,1e-3,3e-3,1e-2",
        help="comma-separated sigma (ignored if --sigma-log-min and --sigma-log-max are set)",
    )
    parser.add_argument(
        "--sigma-log-min",
        type=float,
        default=None,
        help="with --sigma-log-max: log10-uniform grid from this σ (e.g. 1e-7)",
    )
    parser.add_argument(
        "--sigma-log-max",
        type=float,
        default=None,
        help="with --sigma-log-min: log10-uniform grid up to this σ (e.g. 1e-1)",
    )
    parser.add_argument(
        "--sigma-num",
        type=int,
        default=25,
        help="number of σ points on the log grid (when log min/max are set)",
    )
    parser.add_argument("--num-samples", type=int, default=8, help="Monte Carlo samples per sigma")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="comma-separated RNG seeds (each: new z draws; same checkpoint and batch)",
    )
    parser.add_argument("--out", type=str, default=None, help="JSON path (default: base_dir/reports/quadratic_sigma_<ts>.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("Need at least one seed in --seeds")

    model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=args.model_step)
    if args.max_seq_len is not None:
        msl = args.max_seq_len
    else:
        msl = int(meta["model_config"]["sequence_len"])

    # One fixed batch (same split as training loader uses by default)
    train_loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, msl, split="train", device=device
    )
    x, y = next(train_loader)

    if args.sigma_log_min is not None and args.sigma_log_max is not None:
        lo = min(args.sigma_log_min, args.sigma_log_max)
        hi = max(args.sigma_log_min, args.sigma_log_max)
        if lo <= 0 or hi <= 0:
            raise SystemExit("--sigma-log-min and --sigma-log-max must be positive")
        if args.sigma_num < 2:
            raise SystemExit("--sigma-num must be at least 2")
        sigmas = np.logspace(np.log10(lo), np.log10(hi), args.sigma_num, dtype=np.float64).tolist()
    elif args.sigma_log_min is not None or args.sigma_log_max is not None:
        raise SystemExit("Set both --sigma-log-min and --sigma-log-max, or neither")
    else:
        sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    w0 = _flat_params_float(model)
    dim = w0.numel()

    with torch.no_grad():
        with _autocast_disabled(device):
            l0_scalar = _loss_scalar(model, x, y).float().item()

    meta = {
        "model_tag": args.model_tag or "auto",
        "step": args.model_step or "auto",
        "flat_dim": dim,
        "batch_shape": list(x.shape),
        "max_seq_len": msl,
        "sigmas": sigmas,
        "num_samples": args.num_samples,
        "seeds": seeds,
    }
    if args.sigma_log_min is not None:
        meta["sigma_log_grid"] = {
            "min": min(args.sigma_log_min, args.sigma_log_max),
            "max": max(args.sigma_log_min, args.sigma_log_max),
            "num": args.sigma_num,
            "spacing": "log10_uniform",
        }

    results = {
        "meta": meta,
        "by_seed": [],
        "aggregated_by_sigma": [],
    }

    eps = 1e-12
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        by_sigma = []
        for sigma in sigmas:
            rel_errors = []
            abs_errors = []
            true_deltas = []
            quad_deltas = []
            t0 = time.time()
            for _ in range(args.num_samples):
                z = torch.randn(dim, device=device, dtype=torch.float32)
                z.mul_(sigma)
                _set_flat_params_float(model, w0 + z)
                model.zero_grad(set_to_none=True)
                with torch.no_grad():
                    with _autocast_disabled(device):
                        l1 = _loss_scalar(model, x, y).float().item()
                delta_true = l1 - l0_scalar

                _set_flat_params_float(model, w0)
                d_quad = quadratic_delta(model, x, y, z, device).item()
                model.zero_grad(set_to_none=True)
                abs_err = abs(delta_true - d_quad)
                rel = abs_err / (abs(delta_true) + eps)
                rel_errors.append(rel)
                abs_errors.append(abs_err)
                true_deltas.append(delta_true)
                quad_deltas.append(d_quad)

            by_sigma.append(
                {
                    "sigma": sigma,
                    "seconds": time.time() - t0,
                    "rel_err_mean": float(sum(rel_errors) / len(rel_errors)),
                    "rel_err_max": float(max(rel_errors)),
                    "abs_err_mean": float(sum(abs_errors) / len(abs_errors)),
                    "delta_true_mean": float(sum(true_deltas) / len(true_deltas)),
                    "delta_quad_mean": float(sum(quad_deltas) / len(quad_deltas)),
                }
            )

        results["by_seed"].append({"seed": seed, "by_sigma": by_sigma})

    # Mean line: average over seeds; band: std across seeds (per sigma).
    for j, sigma in enumerate(sigmas):
        rel_means = [results["by_seed"][i]["by_sigma"][j]["rel_err_mean"] for i in range(len(seeds))]
        abs_means = [results["by_seed"][i]["by_sigma"][j]["abs_err_mean"] for i in range(len(seeds))]
        m_rel = statistics.fmean(rel_means)
        m_abs = statistics.fmean(abs_means)
        if len(rel_means) > 1:
            s_rel = statistics.stdev(rel_means)
            s_abs = statistics.stdev(abs_means)
        else:
            s_rel = 0.0
            s_abs = 0.0
        results["aggregated_by_sigma"].append(
            {
                "sigma": sigma,
                "rel_err_mean_mean": m_rel,
                "rel_err_mean_std": s_rel,
                "abs_err_mean_mean": m_abs,
                "abs_err_mean_std": s_abs,
                "n_seeds": len(seeds),
            }
        )

    base = get_base_dir()
    out = args.out
    if out is None:
        rep = os.path.join(base, "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"quadratic_sigma_{int(time.time())}.json")
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
