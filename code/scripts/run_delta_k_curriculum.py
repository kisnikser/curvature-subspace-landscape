"""
Curriculum run: train from init, consuming the train dataloader in batches (sequences per row).
After each optimizer step, if ``sequences_seen >= k_nominal + 1`` for the next grid
checkpoint ``k_nominal`` (``--k-grid log`` or ``linear``), we evaluate (weights have been trained on *at least* ``k_nominal``
sequences — typically a bit more right after a batch). Criteria use the **first** ``k_nominal``
and ``k_nominal+1`` rows from the sequence buffer for ``\\mathcal L_k`` / ``\\mathcal L_{k+1}``.

  - Δ_1, full-space Δ_2 (MC, per σ), Δ_2^(D) (Gaussian closed form, per D, σ)
  - validation mean CE and PPL

  cd code
  python -m scripts.run_delta_k_curriculum --depth 6 --k-max 10000 --n-k-points 20 \\
      --train-batch-size 16 --out .nanochat/reports/delta_k_curriculum.json

After each checkpoint, the JSON is written atomically (same path) and ``<stem>_criteria.pdf`` /
``<stem>_val.pdf`` are refreshed so you can monitor progress. Use ``--no-plots`` to skip PDF updates
and only flush JSON (faster). This script uses a single GPU; extra GPUs can run separate seeds in
parallel as separate processes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from nanochat.common import COMPUTE_DTYPE, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

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
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically (same path readers always see a complete file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _log_spaced_int_ks(k_min: int, k_max: int, n: int) -> list[int]:
    """Unique sorted ints, log-spaced between k_min and k_max (inclusive)."""
    lo = math.log10(max(k_min, 1))
    hi = math.log10(max(k_max, k_min))
    raw = np.logspace(lo, hi, num=n)
    ks = sorted({max(1, int(round(x))) for x in raw})
    ks = [k for k in ks if k_min <= k <= k_max]
    if not ks:
        ks = [k_min]
    return ks


def _linear_spaced_int_ks(k_min: int, k_max: int, n: int) -> list[int]:
    """Unique sorted ints, evenly spaced on the linear axis (inclusive endpoints).

    Step between consecutive targets is approximately ``(k_max - k_min) / max(n - 1, 1)``.
    """
    if n <= 0:
        return []
    if k_max < k_min:
        raise ValueError(f"k_max ({k_max}) < k_min ({k_min})")
    raw = np.linspace(float(k_min), float(k_max), num=n)
    ks = sorted({max(1, int(round(x))) for x in raw})
    ks = [k for k in ks if k_min <= k <= k_max]
    if not ks:
        ks = [k_min]
    return ks


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


def _stack_buffer(buf_x: list[torch.Tensor], buf_y: list[torch.Tensor], k: int, device: torch.device):
    """buf_* are CPU (1,T) tensors; return (k,T) on device."""
    if k <= 0 or k > len(buf_x):
        raise ValueError(f"bad k={k}, buffer len={len(buf_x)}")
    x = torch.cat(buf_x[:k], dim=0).to(device=device, dtype=torch.long)
    y = torch.cat(buf_y[:k], dim=0).to(device=device, dtype=torch.long)
    return x, y


def _evaluate_at_k(
    model: torch.nn.Module,
    *,
    k_nominal: int,
    w_checkpoint: torch.Tensor,
    x_k: torch.Tensor,
    y_k: torch.Tensor,
    x_k1: torch.Tensor,
    y_k1: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sigmas: list[float],
    d_list: list[int],
    d_max: int,
    delta1_eps: float,
    delta1_directions: int,
    num_samples_delta2: int,
    eig_iters: int,
    eig_tol: float,
    hess_max_seq: int,
    grad_max_seq: int,
    microbatch: int,
    hvp_microbatch: int,
) -> dict:
    """All metrics at fixed weights w (caller sets model to w_checkpoint)."""
    _set_flat_params(model, w_checkpoint.clone())
    w_k = _flat_params(model).clone()

    print0(f"eval k={k_nominal}: Δ₁ …", flush=True)
    d1 = compute_delta1(
        model,
        w_k,
        x_k,
        y_k,
        x_k1,
        y_k1,
        delta1_eps,
        delta1_directions,
        device,
        dtype,
        microbatch,
    )
    print0(f"eval k={k_nominal}: Δ₁ done", flush=True)

    d2_by_sigma: list[dict] = []
    for sig in sigmas:
        print0(f"eval k={k_nominal}: Δ₂ full-space MC σ={sig} …", flush=True)
        _set_flat_params(model, w_checkpoint.clone())
        w2 = _flat_params(model).clone()
        st = compute_delta2_fullspace(
            model,
            w2,
            x_k,
            y_k,
            x_k1,
            y_k1,
            sig,
            num_samples_delta2,
            device,
            dtype,
            microbatch,
        )
        d2_by_sigma.append({"sigma": sig, **st})
        print0(f"eval k={k_nominal}: Δ₂ full-space MC σ={sig} done", flush=True)

    _set_flat_params(model, w_checkpoint.clone())
    w_k = _flat_params(model).clone()
    xh, yh = trim_batch(x_k, y_k, hess_max_seq)
    xh1, yh1 = trim_batch(x_k1, y_k1, hess_max_seq)

    print0(f"eval k={k_nominal}: loss increment a_k …", flush=True)
    a_k = _compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch)
    print0(f"eval k={k_nominal}: loss increment done", flush=True)

    print0(
        f"eval k={k_nominal}: top-{d_max} eigenvectors (Hessian rows={xh.shape[0]}, "
        f"iters={eig_iters}) …",
        flush=True,
    )
    _set_flat_params(model, w_k)
    _, U_full = compute_top_eigenvectors(
        model,
        xh,
        yh,
        d_max,
        eig_iters,
        eig_tol,
        device,
        dtype,
        hvp_microbatch,
    )
    print0(f"eval k={k_nominal}: eigenvectors done", flush=True)

    nk = int(x_k.shape[0])
    nk1 = int(x_k1.shape[0])
    chunk = grad_max_seq if grad_max_seq > 0 else None
    print0(
        f"eval k={k_nominal}: grads for L_k, L_{{k+1}} ({nk} and {nk1} seq rows; "
        f"grad chunk rows={chunk or 'all'}) …",
        flush=True,
    )
    g_k = grad_flat_lm(model, x_k, y_k, microbatch, max_chunk_rows=chunk)
    g_k1 = grad_flat_lm(model, x_k1, y_k1, microbatch, max_chunk_rows=chunk)
    grad_diff = g_k1 - g_k
    c_full = U_full.T @ grad_diff
    print0(f"eval k={k_nominal}: full grads + projection done", flush=True)

    print0(f"eval k={k_nominal}: compress Hessian to basis (2×) …", flush=True)
    _set_flat_params(model, w_k)
    Hk = compress_hessian_to_basis(model, xh, yh, U_full, hvp_microbatch)
    _set_flat_params(model, w_k)
    Hk1 = compress_hessian_to_basis(model, xh1, yh1, U_full, hvp_microbatch)
    B_full = Hk1 - Hk
    print0(f"eval k={k_nominal}: basis Hessian done", flush=True)

    d2d_entries: list[dict] = []
    for D in sorted(set(d_list)):
        if D > d_max:
            continue
        c_D = c_full[:D].clone()
        B_D = B_full[:D, :D].clone()
        for sig in sigmas:
            gm = gaussian_moment_closed_form(a_k, c_D, B_D, float(sig))
            d2d_entries.append({"D": D, "sigma": sig, "gaussian_moment": float(gm)})

    print0(f"eval k={k_nominal}: subspace Gaussian moments done", flush=True)
    _set_flat_params(model, w_checkpoint.clone())
    return {
        "delta1": d1,
        "delta2_fullspace": d2_by_sigma,
        "delta2_subspace_gaussian": d2d_entries,
    }


def _train_batch(model, optimizer, x: torch.Tensor, y: torch.Tensor, scaler) -> float:
    """One optimizer step on batch (B, T)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(x, y)
    lf = loss.detach().float().item()
    loss.backward()
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return lf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=auto)")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--aspect-ratio", type=int, default=64)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--window-pattern", type=str, default="L")
    p.add_argument(
        "--k-min",
        type=int,
        default=1,
        help="left end of k grid; with --k-grid log, small k_min crowds the left end",
    )
    p.add_argument("--k-max", type=int, default=10000)
    p.add_argument("--n-k-points", type=int, default=20, help="number of k checkpoints (see --k-grid)")
    p.add_argument(
        "--k-grid",
        type=str,
        choices=("log", "linear"),
        default="log",
        help="log: log-spaced k; linear: evenly spaced in k (step ≈ (k_max−k_min)/max(n_k_points−1,1))",
    )
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="sequences (rows) per optimizer step; k checkpoints fire when seq_seen passes k+1",
    )
    p.add_argument("--sigmas", type=str, default="1e-4,1e-3,1e-2")
    p.add_argument("--D-list", type=str, default="1,4,16")
    p.add_argument("--delta1-eps", type=float, default=0.02)
    p.add_argument("--delta1-directions", type=int, default=50)
    p.add_argument("--num-samples-delta2", type=int, default=32)
    p.add_argument("--eig-iters", type=int, default=30)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--hess-max-seq", type=int, default=4)
    p.add_argument(
        "--grad-max-seq",
        type=int,
        default=0,
        help="if >0, compute grad_flat_lm in row chunks of this size (exact ∇ of mean CE; slower); "
        "0=single pass over all rows (can OOM for large k)",
    )
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--hvp-microbatch", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="Stem for live-updated PDFs (<stem>_criteria.pdf, <stem>_val.pdf); default: same as --out",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not refresh PDFs after each checkpoint (JSON still written)",
    )
    p.add_argument(
        "--plot-figsize",
        type=str,
        default="7,4.5",
        help="Figure size for incremental plots (only if --no-plots not set)",
    )
    p.add_argument(
        "--plot-font",
        type=str,
        default=None,
        help="Path to Times New Roman .ttf for plots; default: code-old/Times New Roman.ttf",
    )
    p.add_argument(
        "--progress-every-seq",
        type=int,
        default=256,
        help="Print curriculum line every this many sequences seen (0=off): seq_seen, next_eval_k",
    )
    args = p.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    if ddp_world_size != 1:
        raise SystemExit("Use single-process run (world_size=1) for this script.")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sigmas = _parse_float_list(args.sigmas)
    d_list = _parse_int_list(args.D_list)
    d_max = max(d_list)
    if args.k_grid == "log":
        k_targets = _log_spaced_int_ks(args.k_min, args.k_max, args.n_k_points)
    else:
        k_targets = _linear_spaced_int_ks(args.k_min, args.k_max, args.n_k_points)
    if not k_targets:
        raise SystemExit("empty k grid")

    out_path = (
        Path(args.out)
        if args.out
        else Path(os.path.join(get_base_dir(), "reports", f"delta_k_curriculum_{int(time.time())}.json"))
    )
    plot_prefix = Path(args.plot_prefix) if args.plot_prefix else (out_path.parent / out_path.stem)
    _repo_root = Path(__file__).resolve().parents[2]
    plot_font_path = Path(args.plot_font) if args.plot_font else (_repo_root / "code-old" / "Times New Roman.ttf")
    plot_figsize = tuple(float(x.strip()) for x in args.plot_figsize.split(","))

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    base_dim = args.depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=COMPUTE_DTYPE)
    model.train()

    optimizer = model.setup_optimizer()
    scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

    msl = args.max_seq_len
    meta: dict = {
        "depth": args.depth,
        "max_seq_len": msl,
        "k_min": args.k_min,
        "k_max": args.k_max,
        "k_grid": args.k_grid,
        "k_targets": k_targets,
        "train_batch_size": args.train_batch_size,
        "seed": args.seed,
        "sigmas": sigmas,
        "D_list": d_list,
        "model_config": asdict(config),
        "total_checkpoints": len(k_targets),
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "json_path": str(out_path.resolve()),
        "plot_prefix": str(plot_prefix),
        "hess_max_seq": args.hess_max_seq,
        "grad_max_seq": args.grad_max_seq,
        "notes": (
            "Trained from init; at nominal grid k, seq_seen >= k+1 so we can form L_k and L_{k+1} from "
            "the first k and k+1 buffered sequences; weights may reflect training on more than k "
            "sequences (batch overshoot). Field sequences_trained_at_eval records seq_seen. "
            "JSON and PDFs refresh after each checkpoint."
        ),
    }

    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.train_batch_size, msl, split="train", device=device, resume_state_dict=None
    )
    train_iter = iter(train_loader)

    buf_x: list[torch.Tensor] = []
    buf_y: list[torch.Tensor] = []
    seq_seen = 0
    dtype = COMPUTE_DTYPE
    rows_out: list[dict] = []
    next_target_idx = 0
    t0 = time.time()

    def flush_results(*, plot: bool) -> None:
        meta["progress"] = {
            "checkpoints_completed": len(rows_out),
            "total_checkpoints": len(k_targets),
            "seq_seen": seq_seen,
            "last_k_evaluated": rows_out[-1]["k"] if rows_out else None,
        }
        meta["last_json_write_at"] = datetime.now(timezone.utc).isoformat()
        payload = {"meta": meta, "by_k": rows_out}
        _atomic_write_json(out_path, payload)
        if rows_out:
            print0(f"Saved checkpoint {len(rows_out)}/{len(k_targets)} → {out_path}")
        else:
            print0(f"Started run (0 checkpoints yet) → {out_path}")
        if plot and (not args.no_plots) and rows_out:
            from scripts.plot_delta_k_curriculum import plot_criteria, plot_val

            plot_criteria(payload, Path(str(plot_prefix) + "_criteria.pdf"), plot_font_path, plot_figsize)
            plot_val(payload, Path(str(plot_prefix) + "_val.pdf"), plot_font_path, plot_figsize)
            print0(f"Updated plots: {plot_prefix}_criteria.pdf, {plot_prefix}_val.pdf")

    flush_results(plot=False)

    def run_checkpoints_for_current_batch() -> None:
        """Fire all k_nominal checkpoints satisfied by seq_seen with prefix data in buffer."""
        nonlocal next_target_idx, t0
        while next_target_idx < len(k_targets):
            k_t = k_targets[next_target_idx]
            if seq_seen < k_t + 1:
                break
            if len(buf_x) < k_t + 1:
                raise RuntimeError(f"buffer len {len(buf_x)} < k_t+1={k_t+1} despite seq_seen={seq_seen}")

            print0(
                f"Evaluating at k={k_t} (checkpoint {next_target_idx + 1}/{len(k_targets)}), "
                f"seq_seen={seq_seen}"
            )

            x_k, y_k = _stack_buffer(buf_x, buf_y, k_t, device)
            x_k1, y_k1 = _stack_buffer(buf_x, buf_y, k_t + 1, device)

            w_ckpt = _flat_params(model).detach().clone()

            val_stats = _eval_validation(
                model, tokenizer, device, msl, args.val_batch_size, args.microbatch, args.val_batches
            )
            print0(f"eval k={k_t}: validation CE done", flush=True)

            ev = _evaluate_at_k(
                model,
                k_nominal=k_t,
                w_checkpoint=w_ckpt,
                x_k=x_k,
                y_k=y_k,
                x_k1=x_k1,
                y_k1=y_k1,
                device=device,
                dtype=dtype,
                sigmas=sigmas,
                d_list=d_list,
                d_max=d_max,
                delta1_eps=args.delta1_eps,
                delta1_directions=args.delta1_directions,
                num_samples_delta2=args.num_samples_delta2,
                eig_iters=args.eig_iters,
                eig_tol=args.eig_tol,
                hess_max_seq=args.hess_max_seq,
                grad_max_seq=args.grad_max_seq,
                microbatch=args.microbatch,
                hvp_microbatch=args.hvp_microbatch,
            )
            model.train()
            model.zero_grad(set_to_none=True)

            rows_out.append(
                {
                    "k": k_t,
                    "sequences_trained_at_eval": seq_seen,
                    "sequences_in_buffer_for_increment": k_t + 1,
                    **val_stats,
                    **ev,
                    "seconds_eval": time.time() - t0,
                }
            )
            t0 = time.time()
            next_target_idx += 1
            flush_results(plot=True)

    try:
        while next_target_idx < len(k_targets):
            x, y, st = next(train_iter)
            prev_seen = seq_seen
            _train_batch(model, optimizer, x, y, scaler)
            for i in range(x.shape[0]):
                buf_x.append(x[i : i + 1].detach().cpu().clone())
                buf_y.append(y[i : i + 1].detach().cpu().clone())
            seq_seen += x.shape[0]
            pe = args.progress_every_seq
            if pe > 0:
                b0 = prev_seen // pe
                b1 = seq_seen // pe
                if b1 != b0 or prev_seen == 0:
                    if next_target_idx < len(k_targets):
                        k_need = k_targets[next_target_idx]
                        print0(
                            f"Training: seq_seen={seq_seen}  next_eval_k={k_need}  "
                            f"(need seq_seen ≥ {k_need + 1})"
                        )
                    else:
                        print0(f"Training: seq_seen={seq_seen}  (all k checkpoints evaluated)")
            run_checkpoints_for_current_batch()
    except StopIteration:
        meta["status"] = "partial"
        meta["stop_reason"] = "train_iterator_exhausted"
        print0(
            f"Warning: train iterator exhausted at seq_seen={seq_seen}; "
            f"evaluated {next_target_idx}/{len(k_targets)} checkpoints."
        )
    else:
        meta["status"] = "complete"

    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    flush_results(plot=False)

    print0(f"Final JSON: {out_path}")
    compute_cleanup()


if __name__ == "__main__":
    main()
