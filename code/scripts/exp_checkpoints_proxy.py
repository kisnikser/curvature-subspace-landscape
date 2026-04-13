"""
Experiment 3: quadratic proxy validity across multiple training checkpoints.

Repeats the sigma-sweep from exp_quadratic_sigma.py for several model steps,
producing a single JSON that can be plotted as a multi-line figure (one line per
checkpoint) to verify that the valid proxy regime (sigma <= ~1e-3) is robust
across training stages.

Usage (from code/):
  python -m scripts.exp_checkpoints_proxy \\
      --model-tag d6 \\
      --model-steps 500,1500,3500 \\
      --sigma-log-min 1e-7 --sigma-log-max 1e-1 --sigma-num 25 \\
      --num-samples 8 --seeds 0,1,2

Output JSON structure:
  {
    "meta": { ... },
    "by_step": [
      {
        "step": 500,
        "aggregated_by_sigma": [
          {"sigma": ..., "rel_err_mean_mean": ..., "rel_err_mean_std": ...},
          ...
        ]
      },
      ...
    ]
  }

Plot: python -m scripts.plot_checkpoints_proxy --json <output.json>
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


# ---------------------------------------------------------------------------
# Helpers (identical to exp_quadratic_sigma.py)
# ---------------------------------------------------------------------------


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def _sdp_math_only(device: torch.device):
    if device.type == "cuda":
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def _flat_params_float(model: torch.nn.Module) -> torch.Tensor:
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
    out = model(x, y)
    return out if out.ndim == 0 else out.mean()


def _quadratic_delta(model, x, y, z, device) -> float:
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
    return (linear + 0.5 * z.dot(hvp_flat)).item()


def _run_sigma_sweep(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    w0: torch.Tensor,
    sigmas: list[float],
    num_samples: int,
    seeds: list[int],
    device: torch.device,
    eps: float = 1e-12,
) -> list[dict]:
    """Run multi-seed sigma sweep; returns aggregated_by_sigma list."""
    dim = w0.numel()

    with torch.no_grad():
        with _autocast_disabled(device):
            l0 = _loss_scalar(model, x, y).float().item()

    by_seed_results: list[list[dict]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        by_sigma = []
        for sigma in sigmas:
            rel_errors = []
            for _ in range(num_samples):
                z = torch.randn(dim, device=device, dtype=torch.float32).mul_(sigma)
                _set_flat_params_float(model, w0 + z)
                with torch.no_grad():
                    with _autocast_disabled(device):
                        l1 = _loss_scalar(model, x, y).float().item()
                delta_true = l1 - l0
                _set_flat_params_float(model, w0)
                d_quad = _quadratic_delta(model, x, y, z, device)
                model.zero_grad(set_to_none=True)
                rel = abs(delta_true - d_quad) / (abs(delta_true) + eps)
                rel_errors.append(rel)
            by_sigma.append({"sigma": sigma, "rel_err_mean": float(sum(rel_errors) / len(rel_errors))})

        by_seed_results.append(by_sigma)

    # Aggregate over seeds: mean ± std per sigma
    aggregated = []
    for j, sigma in enumerate(sigmas):
        rel_means = [by_seed_results[i][j]["rel_err_mean"] for i in range(len(seeds))]
        m = statistics.fmean(rel_means)
        s = statistics.stdev(rel_means) if len(rel_means) > 1 else 0.0
        aggregated.append({"sigma": sigma, "rel_err_mean_mean": m, "rel_err_mean_std": s})

    # Restore model parameters to w0
    _set_flat_params_float(model, w0)
    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quadratic proxy validity across training checkpoints (Experiment 3)."
    )
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument(
        "--model-steps",
        type=str,
        required=True,
        help="Comma-separated training steps to evaluate, e.g. 500,1500,3500",
    )
    parser.add_argument("--device-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--sigma-log-min", type=float, default=1e-7)
    parser.add_argument("--sigma-log-max", type=float, default=1e-1)
    parser.add_argument("--sigma-num", type=int, default=25)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    steps = [int(s.strip()) for s in args.model_steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("--model-steps must contain at least one step")

    sigmas: list[float] = np.logspace(
        np.log10(args.sigma_log_min), np.log10(args.sigma_log_max), args.sigma_num
    ).tolist()

    results: dict = {
        "meta": {
            "model_tag": args.model_tag or "auto",
            "steps": steps,
            "sigmas": sigmas,
            "num_samples": args.num_samples,
            "seeds": seeds,
            "sigma_log_min": args.sigma_log_min,
            "sigma_log_max": args.sigma_log_max,
        },
        "by_step": [],
    }

    for step in steps:
        print(f"\n=== step {step} ===")
        model, tokenizer, meta = load_model("base", device, "eval", model_tag=args.model_tag, step=step)
        msl = int(args.max_seq_len or meta["model_config"]["sequence_len"])

        train_loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, msl, split="train", device=device
        )
        x, y = next(train_loader)
        w0 = _flat_params_float(model)

        t0 = time.time()
        aggregated = _run_sigma_sweep(
            model=model,
            x=x,
            y=y,
            w0=w0,
            sigmas=sigmas,
            num_samples=args.num_samples,
            seeds=seeds,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s")

        results["by_step"].append({"step": step, "aggregated_by_sigma": aggregated})

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"checkpoints_proxy_{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
