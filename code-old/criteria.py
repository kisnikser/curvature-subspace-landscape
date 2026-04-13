"""
Convergence criteria for loss landscape analysis.

Delta_1: absolute one-point criterion
Delta_2: mean-squared criterion  
Delta_2^(D): mean-squared criterion in principal curvature subspace
"""
import torch
import torch.nn.functional as F


def _get_flat_params(model):
    """Get flattened parameter vector."""
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


def _set_flat_params(model, flat_params):
    """Set model parameters from flattened vector."""
    idx = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(flat_params[idx:idx + numel].view_as(p))
            idx += numel


def _summarize_samples(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = tensor.mean().item() if values else 0.0
    std = tensor.std(unbiased=True).item() if len(values) > 1 else 0.0
    stderr = std / (len(values) ** 0.5) if values else 0.0
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "num_samples": len(values),
    }


def compute_loss(model, x, y, microbatch: int = 512):
    """
    Mean cross-entropy over all tokens (causal LM).
    Micro-batches along sequence dimension 0 to limit activation memory when k is large.
    """
    model.eval()
    with torch.no_grad():
        b = x.shape[0]
        if b <= microbatch:
            logits, _ = model(x, y)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item()
        total, ntok = 0.0, 0
        for s in range(0, b, microbatch):
            xb = x[s : s + microbatch]
            yb = y[s : s + microbatch]
            logits, _ = model(xb, yb)
            part = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), yb.reshape(-1), reduction="sum"
            )
            total += part.item()
            ntok += xb.numel()
        return total / ntok


def compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch: int = 512) -> float:
    """Return L_{k+1}(w) - L_k(w) at the model's current parameters."""
    loss_k = compute_loss(model, x_k, y_k, microbatch=microbatch)
    loss_k1 = compute_loss(model, x_k1, y_k1, microbatch=microbatch)
    return loss_k1 - loss_k


def compute_delta1(
    model,
    w_k,
    x_k,
    y_k,
    x_k1,
    y_k1,
    eps,
    num_directions,
    device,
    dtype,
    microbatch: int = 512,
    return_details: bool = False,
):
    """
    Compute absolute one-point criterion Delta_1.
    
    Delta_1 = (1/M) * sum_j |L_{k+1}(w_k + eps*d_j) - L_k(w_k + eps*d_j)|
    
    where d_j are random unit directions.
    
    Args:
        model: neural network model
        w_k: parameters at minimum of L_k (flat tensor)
        x_k, y_k: data for computing L_k (k samples)
        x_k1, y_k1: data for computing L_{k+1} (k+1 samples)
        eps: perturbation radius
        num_directions: number of random directions M
        device, dtype: torch device and dtype
    
    Returns:
        delta1: estimated Delta_1 value
    """
    dim = w_k.shape[0]
    diffs = []
    
    for _ in range(num_directions):
        d = torch.randn(dim, device=device, dtype=dtype)
        d = d / d.norm()
        
        w_perturbed = w_k + eps * d
        _set_flat_params(model, w_perturbed)
        
        diff = abs(compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch=microbatch))
        diffs.append(diff)
    
    _set_flat_params(model, w_k)
    
    stats = _summarize_samples(diffs)
    if return_details:
        return stats
    return stats["mean"]


def compute_delta2(
    model,
    w_k,
    x_k,
    y_k,
    x_k1,
    y_k1,
    sigma,
    num_samples,
    device,
    dtype,
    microbatch: int = 512,
    return_details: bool = False,
):
    """
    Compute mean-squared criterion Delta_2.
    
    Delta_2 = E_q[ (L_{k+1}(w) - L_k(w))^2 ]
    
    where q(w) = N(w_k, sigma^2 * I).
    
    Args:
        model: neural network model
        w_k: parameters at minimum of L_k (flat tensor)
        x_k, y_k: data for computing L_k (k samples)
        x_k1, y_k1: data for computing L_{k+1} (k+1 samples)
        sigma: standard deviation for Gaussian sampling
        num_samples: number of Monte Carlo samples
        device, dtype: torch device and dtype
    
    Returns:
        delta2: estimated Delta_2 value
    """
    dim = w_k.shape[0]
    sq_diffs = []
    
    for _ in range(num_samples):
        noise = torch.randn(dim, device=device, dtype=dtype) * sigma
        w_sample = w_k + noise
        _set_flat_params(model, w_sample)
        
        diff = compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch=microbatch)
        sq_diffs.append(diff ** 2)
    
    _set_flat_params(model, w_k)
    
    stats = _summarize_samples(sq_diffs)
    if return_details:
        return stats
    return stats["mean"]


def compute_delta2_subspace(
    model,
    w_k,
    U_D,
    x_k,
    y_k,
    x_k1,
    y_k1,
    sigma,
    num_samples,
    device,
    dtype,
    microbatch: int = 512,
    return_details: bool = False,
):
    """
    Compute mean-squared criterion in principal curvature subspace Delta_2^(D).
    
    Delta_2^(D) = E_q[ (L_{k+1}(w_k + U_D*z) - L_k(w_k + U_D*z))^2 ]
    
    where q(z) = N(0, sigma^2 * I_D) and U_D contains top-D eigenvectors.
    
    Args:
        model: neural network model
        w_k: parameters at minimum of L_k (flat tensor)
        U_D: matrix of top-D eigenvectors (N x D)
        x_k, y_k: data for computing L_k (k samples)
        x_k1, y_k1: data for computing L_{k+1} (k+1 samples)
        sigma: standard deviation for Gaussian sampling in subspace
        num_samples: number of Monte Carlo samples
        device, dtype: torch device and dtype
    
    Returns:
        delta2_D: estimated Delta_2^(D) value
    """
    D = U_D.shape[1]
    sq_diffs = []
    
    for _ in range(num_samples):
        z = torch.randn(D, device=device, dtype=dtype) * sigma
        w_sample = w_k + U_D @ z
        _set_flat_params(model, w_sample)
        
        diff = compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch=microbatch)
        sq_diffs.append(diff ** 2)
    
    _set_flat_params(model, w_k)
    
    stats = _summarize_samples(sq_diffs)
    if return_details:
        return stats
    return stats["mean"]


def quadratic_form_expectation(
    a_k: float,
    c_k: torch.Tensor,
    B_k: torch.Tensor,
    num_samples: int,
    sigma: float,
    device,
    dtype,
) -> dict[str, float]:
    """
    Monte Carlo diagnostics for the quadratic subspace approximation.

    Returns the component energies and the total quadratic expectation under
    z ~ N(0, sigma^2 I).
    """
    D = c_k.numel()
    value_terms = []
    linear_terms = []
    curvature_terms = []
    totals = []

    for _ in range(num_samples):
        z = torch.randn(D, device=device, dtype=dtype) * sigma
        linear = torch.dot(c_k, z)
        curvature = 0.5 * torch.dot(z, B_k @ z)
        total = a_k + linear + curvature
        value_terms.append(float(a_k ** 2))
        linear_terms.append(float(linear.item() ** 2))
        curvature_terms.append(float(curvature.item() ** 2))
        totals.append(float(total.item() ** 2))

    return {
        "value_gap_sq": _summarize_samples(value_terms)["mean"],
        "linear_energy": _summarize_samples(linear_terms)["mean"],
        "curvature_energy": _summarize_samples(curvature_terms)["mean"],
        "quadratic_total": _summarize_samples(totals)["mean"],
    }
