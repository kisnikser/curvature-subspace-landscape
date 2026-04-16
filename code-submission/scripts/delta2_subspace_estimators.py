"""
Estimate the subspace mean-squared criterion Δ_2^(D) in three ways (paper Sec. algo):

  1) Direct subspace MC — E[ (L_{k+1}(w*+U z) - L_k(w*+U z))^2 ],  z ~ N(0, σ² I_D)
  2) Quadratic MC — same expectation for the surrogate (a + c^T z + z^T B z / 2)^2
  3) Gaussian-moment closed form — exact E of the quadratic surrogate under Gaussian z

Coefficients (at w*): a = L_{k+1}(w*)-L_k(w*), c = U^T(g^{(k+1)}-g^{(k)}),
B = U^T (H^{(k+1)}-H^{(k)}) U.

Ref: code-old/run_experiments.py (gaussian_moment_estimator, quadratic_mc_estimator),
     code-old/criteria.py (compute_delta2_subspace).

Example:
  cd code
  python -m scripts.delta2_subspace_estimators --model-tag d6 --model-step 3500 \\
    --k-sequences 8 --subspace-dim 5 --sigma 0.02 --num-samples 64
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from nanochat.common import get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit


# ---------------------------------------------------------------------------
# Pure estimators (match code-old)
# ---------------------------------------------------------------------------


def _summarize_samples(values: list[float]) -> dict:
    t = torch.tensor(values, dtype=torch.float64)
    mean = float(t.mean().item()) if values else 0.0
    std = float(t.std(unbiased=True).item()) if len(values) > 1 else 0.0
    stderr = std / (len(values) ** 0.5) if values else 0.0
    return {"mean": mean, "std": std, "stderr": stderr, "num_samples": len(values)}


def gaussian_moment_closed_form(
    a_k: float,
    c_k: torch.Tensor,
    B_k: torch.Tensor,
    sigma: float,
) -> float:
    """
    E[(a + c^T z + 0.5 z^T B z)^2] for z ~ N(0, σ² I), with B symmetric.
    """
    tr_B = torch.trace(B_k).item()
    tr_B2 = torch.trace(B_k @ B_k).item()
    c_norm_sq = torch.dot(c_k, c_k).item()
    return (
        a_k**2
        + a_k * sigma**2 * tr_B
        + sigma**2 * c_norm_sq
        + (sigma**4 / 4.0) * (2.0 * tr_B2 + tr_B**2)
    )


def quadratic_mc_estimate(
    a_k: float,
    c_k: torch.Tensor,
    B_k: torch.Tensor,
    sigma: float,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Monte Carlo for E[(a + c^T z + 0.5 z^T B z)^2], z ~ N(0, σ² I_D)."""
    D = c_k.numel()
    sq_vals: list[float] = []
    for _ in range(num_samples):
        z = torch.randn(D, device=device, dtype=dtype) * sigma
        val = a_k + torch.dot(c_k, z).item() + 0.5 * torch.dot(z, B_k @ z).item()
        sq_vals.append(val**2)
    return _summarize_samples(sq_vals)


def direct_subspace_mc(
    model: torch.nn.Module,
    w_k: torch.Tensor,
    U_D: torch.Tensor,
    x_k: torch.Tensor,
    y_k: torch.Tensor,
    x_k1: torch.Tensor,
    y_k1: torch.Tensor,
    sigma: float,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    microbatch: int,
) -> dict:
    """Δ_2^(D) via direct MC (true increment, no quadratic surrogate)."""
    D = U_D.shape[1]
    sq: list[float] = []
    for _ in range(num_samples):
        z = torch.randn(D, device=device, dtype=dtype) * sigma
        w_sample = w_k + U_D @ z
        _set_flat_params(model, w_sample)
        diff = _compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch)
        sq.append(diff**2)
    _set_flat_params(model, w_k)
    return _summarize_samples(sq)


# ---------------------------------------------------------------------------
# Model helpers (nanochat LM)
# ---------------------------------------------------------------------------


def _autocast_disabled(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return contextlib.nullcontext()


def _sdp_math(device: torch.device):
    if device.type == "cuda":
        return sdpa_kernel(SDPBackend.MATH)
    return contextlib.nullcontext()


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])


def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n


def _mean_ce(model, x: torch.Tensor, y: torch.Tensor, microbatch: int) -> torch.Tensor:
    """Mean CE over all tokens (nanochat `forward` returns scalar mean when targets given)."""
    model.eval()
    total = torch.zeros((), device=x.device, dtype=torch.float32)
    ntok = 0
    for s in range(0, x.shape[0], microbatch):
        xb, yb = x[s : s + microbatch], y[s : s + microbatch]
        with _autocast_disabled(x.device):
            loss = model(xb, yb)
        nb = xb.numel()
        total = total + loss.float() * nb
        ntok += nb
    return total / max(ntok, 1)


def _compute_loss_scalar(model, x: torch.Tensor, y: torch.Tensor, microbatch: int) -> float:
    with torch.no_grad():
        return _mean_ce(model, x, y, microbatch).item()


def _compute_increment(
    model: torch.nn.Module,
    x_k: torch.Tensor,
    y_k: torch.Tensor,
    x_k1: torch.Tensor,
    y_k1: torch.Tensor,
    microbatch: int,
) -> float:
    lk = _compute_loss_scalar(model, x_k, y_k, microbatch)
    lk1 = _compute_loss_scalar(model, x_k1, y_k1, microbatch)
    return lk1 - lk


def grad_flat_lm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    microbatch: int,
    *,
    max_chunk_rows: int | None = None,
) -> torch.Tensor:
    """Gradient of global mean CE (``_mean_ce`` over all tokens in ``x``) w.r.t. parameters.

    If ``max_chunk_rows`` is set and ``x.shape[0] > max_chunk_rows``, splits ``x`` into row blocks,
    backprops each block's mean CE, and combines as ``(sum_i n_i * grad_i) / N`` with ``N`` the total
    token count—same gradient as one full forward, lower peak memory.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    ntot = int(x.numel())
    if ntot == 0:
        raise ValueError("grad_flat_lm: empty x")

    use_chunks = (
        max_chunk_rows is not None and max_chunk_rows > 0 and int(x.shape[0]) > max_chunk_rows
    )

    if not use_chunks:
        model.zero_grad(set_to_none=True)
        with _sdp_math(next(model.parameters()).device):
            with _autocast_disabled(x.device):
                loss = _mean_ce(model, x, y, microbatch)
            loss.backward()
        return torch.cat([p.grad.reshape(-1) for p in params])

    grad_acc: torch.Tensor | None = None
    for s in range(0, int(x.shape[0]), max_chunk_rows):
        xe = x[s : s + max_chunk_rows]
        ye = y[s : s + max_chunk_rows]
        model.zero_grad(set_to_none=True)
        with _sdp_math(next(model.parameters()).device):
            with _autocast_disabled(x.device):
                loss = _mean_ce(model, xe, ye, microbatch)
            loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in params])
        nc = int(xe.numel())
        grad_acc = g * nc if grad_acc is None else grad_acc + g * nc
    assert grad_acc is not None
    return grad_acc / float(ntot)


def hvp_lm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    vec: torch.Tensor,
    microbatch: int,
) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    with _sdp_math(next(model.parameters()).device):
        with _autocast_disabled(x.device):
            loss = _mean_ce(model, x, y, microbatch)
        loss.backward(create_graph=True)
        g = torch.cat([p.grad.reshape(-1) for p in params])
        z = (g * vec).sum()
        hvp_tup = torch.autograd.grad(z, params, retain_graph=False, allow_unused=False)
    return torch.cat([h.reshape(-1) for h in hvp_tup])


def compress_hessian_to_basis(
    model, x, y, basis: torch.Tensor, microbatch: int
) -> torch.Tensor:
    """U^T H U for symmetric H in LM Hessian; basis (N, D) orthonormal columns."""
    D = basis.shape[1]
    Hc = torch.zeros((D, D), device=basis.device, dtype=basis.dtype)
    for j in range(D):
        hvp = hvp_lm(model, x, y, basis[:, j], microbatch)
        Hc[:, j] = basis.T @ hvp
    return 0.5 * (Hc + Hc.T)


def power_iteration(matvec_fn, dim: int, num_iters: int, tol: float, device, dtype):
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)
    eigval = torch.tensor(0.0, device=device, dtype=dtype)
    for _ in range(num_iters):
        w = matvec_fn(v)
        nw = w.norm()
        if nw < 1e-12:
            break
        v_next = w / nw
        eigval_next = torch.dot(v_next, w)
        if torch.abs(eigval_next - eigval) < tol * torch.abs(eigval_next).clamp(min=1e-12):
            return eigval_next, v_next
        v, eigval = v_next, eigval_next
    return eigval, v


def compute_top_eigenvectors(
    model,
    x,
    y,
    D: int,
    num_iters: int,
    tol: float,
    device,
    dtype,
    hvp_microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    eigenvalues: list[float] = []
    eigenvectors: list[torch.Tensor] = []

    def base_matvec(v):
        return hvp_lm(model, x, y, v, hvp_microbatch)

    for i in range(D):
        if i == 0:
            matvec_fn = base_matvec
        else:
            U_prev = torch.stack(eigenvectors, dim=1)
            lam_prev = torch.tensor(eigenvalues, device=device, dtype=dtype)

            def deflated_matvec(v, U=U_prev, lam=lam_prev):
                Hv = base_matvec(v)
                for j in range(U.shape[1]):
                    uj = U[:, j]
                    Hv = Hv - lam[j] * torch.dot(uj, v) * uj
                return Hv

            matvec_fn = deflated_matvec

        eigval, eigvec = power_iteration(matvec_fn, dim, num_iters, tol, device, dtype)
        eigenvalues.append(float(eigval.item()))
        eigenvectors.append(eigvec)

    eigs = torch.tensor(eigenvalues, device=device, dtype=dtype)
    U_D = torch.stack(eigenvectors, dim=1)
    return eigs, U_D


def trim_batch(x: torch.Tensor, y: torch.Tensor, max_seq: int):
    if x.shape[0] <= max_seq:
        return x, y
    idx = torch.linspace(0, x.shape[0] - 1, steps=max_seq, device=x.device)
    idx = torch.round(idx).long()
    idx = torch.unique_consecutive(idx)
    if idx.numel() < max_seq:
        pad = torch.arange(x.shape[0] - (max_seq - idx.numel()), x.shape[0], device=x.device)
        idx = torch.unique(torch.cat([idx, pad], dim=0), sorted=True)[-max_seq:]
    return x[idx].contiguous(), y[idx].contiguous()


def build_batches_first_k_sequences(
    loader_iter, k: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One loader batch (B>=k+1); return (x[:k],y[:k]), (x[:k+1],y[:k+1])."""
    x, y = next(loader_iter)
    if x.shape[0] < k + 1:
        raise SystemExit(f"Need batch size >= k+1 = {k+1}, got {x.shape[0]}")
    x_k, y_k = x[:k].to(device), y[:k].to(device)
    x_k1, y_k1 = x[: k + 1].to(device), y[: k + 1].to(device)
    return x_k, y_k, x_k1, y_k1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", type=str, default=None)
    p.add_argument("--model-step", type=int, default=None)
    p.add_argument("--device-batch-size", type=int, default=16, help="must be > k (k-sequences)")
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--k-sequences", type=int, default=8, help="use first k vs k+1 sequences for L_k vs L_{k+1}")
    p.add_argument("--subspace-dim", type=int, default=5, help="D for top-Hessian subspace")
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--num-samples", type=int, default=64, help="MC samples for direct + quadratic MC")
    p.add_argument("--eig-iters", type=int, default=30)
    p.add_argument("--eig-tol", type=float, default=1e-4)
    p.add_argument("--hess-max-seq", type=int, default=4, help="max sequences for Hessian / power iter")
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument(
        "--hvp-microbatch",
        type=int,
        default=2,
        help="microbatch rows for HVP / Hessian compress (keep small for memory)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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

    print("Computing top-%d Hessian eigenvectors (HVP power iter)..." % args.subspace_dim)
    _set_flat_params(model, w_k)
    eigenvalues, U_full = compute_top_eigenvectors(
        model,
        xh,
        yh,
        args.subspace_dim,
        args.eig_iters,
        args.eig_tol,
        device,
        dtype,
        args.hvp_microbatch,
    )
    U_D = U_full

    g_k = grad_flat_lm(model, x_k, y_k, args.microbatch)
    g_k1 = grad_flat_lm(model, x_k1, y_k1, args.microbatch)
    grad_diff = g_k1 - g_k
    c_k = U_D.T @ grad_diff

    _set_flat_params(model, w_k)
    Hk = compress_hessian_to_basis(model, xh, yh, U_D, args.hvp_microbatch)
    _set_flat_params(model, w_k)
    Hk1 = compress_hessian_to_basis(model, xh1, yh1, U_D, args.hvp_microbatch)
    B_k = Hk1 - Hk

    gm = gaussian_moment_closed_form(a_k, c_k, B_k, args.sigma)
    qmc = quadratic_mc_estimate(a_k, c_k, B_k, args.sigma, args.num_samples, device, dtype)

    _set_flat_params(model, w_k)
    direct = direct_subspace_mc(
        model,
        w_k,
        U_D,
        x_k,
        y_k,
        x_k1,
        y_k1,
        args.sigma,
        args.num_samples,
        device,
        dtype,
        args.microbatch,
    )

    results = {
        "meta": {
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "k_sequences": args.k_sequences,
            "subspace_dim": args.subspace_dim,
            "sigma": args.sigma,
            "num_samples_mc": args.num_samples,
            "seconds": time.time() - t0,
            "top_hessian_eigenvalues": eigenvalues.cpu().tolist(),
        },
        "coefficients": {
            "a_k": a_k,
            "c_norm": float(torch.linalg.norm(c_k).item()),
            "B_fro": float(torch.linalg.norm(B_k).item()),
        },
        "delta2_subspace": {
            "direct_mc": direct,
            "quadratic_mc": qmc,
            "gaussian_moment_closed_form": gm,
        },
        "notes": {
            "direct": "True Δ_2^(D): MC of squared increment L_{k+1}-L_k along w*+U z",
            "quadratic_mc": "Surrogate: MC of (a + c^T z + z^T B z/2)^2",
            "gm": "Closed-form E of surrogate under z~N(0,σ²I); should match quadratic_MC as num_samples→∞",
        },
    }

    out = args.out
    if out is None:
        rep = os.path.join(get_base_dir(), "reports")
        os.makedirs(rep, exist_ok=True)
        out = os.path.join(rep, f"delta2_subspace_estimators_{int(time.time())}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return obj
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        return obj

    with open(out, "w", encoding="utf-8") as f:
        json.dump(_json_safe(results), f, indent=2)

    print(f"Wrote {out}")
    print(f"  direct MC mean:     {direct['mean']:.8e}")
    print(f"  quadratic MC mean:  {qmc['mean']:.8e}")
    print(f"  Gaussian moment:    {gm:.8e}")


if __name__ == "__main__":
    main()
