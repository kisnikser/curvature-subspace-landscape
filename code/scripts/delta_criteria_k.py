"""
Δ_1 (one-point) and full-space Δ_2 (isotropic Gaussian) for nanochat LM.

Uses the same loss / increment helpers as scripts.delta2_subspace_estimators.
"""
from __future__ import annotations

import torch

from scripts.delta2_subspace_estimators import (
    _compute_increment,
    _set_flat_params,
    _summarize_samples,
)


def compute_delta1(
    model: torch.nn.Module,
    w_k: torch.Tensor,
    x_k: torch.Tensor,
    y_k: torch.Tensor,
    x_k1: torch.Tensor,
    y_k1: torch.Tensor,
    eps: float,
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
    microbatch: int,
) -> dict:
    """
    Δ_1 ≈ (1/M) sum_j |L_{k+1}(w*+ε d_j) - L_k(w*+ε d_j)|, d_j ~ uniform on sphere.
    """
    dim = w_k.numel()
    diffs: list[float] = []
    for _ in range(num_directions):
        d = torch.randn(dim, device=device, dtype=dtype)
        d = d / (d.norm() + 1e-12)
        _set_flat_params(model, w_k + float(eps) * d)
        diffs.append(abs(_compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch)))
    _set_flat_params(model, w_k)
    return _summarize_samples(diffs)


def compute_delta2_fullspace(
    model: torch.nn.Module,
    w_k: torch.Tensor,
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
    """
    Full-space Δ_2 = E[ (L_{k+1}(w) - L_k(w))^2 ],  w ~ N(w*, σ² I_N).
    """
    dim = w_k.numel()
    sq: list[float] = []
    for _ in range(num_samples):
        noise = torch.randn(dim, device=device, dtype=dtype) * float(sigma)
        _set_flat_params(model, w_k + noise)
        diff = _compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch)
        sq.append(diff**2)
    _set_flat_params(model, w_k)
    return _summarize_samples(sq)
