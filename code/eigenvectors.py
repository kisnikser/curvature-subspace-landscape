"""
Compute top-D eigenvectors of the Hessian matrix.

Uses power iteration with deflation to find the largest eigenvalues
and corresponding eigenvectors.
"""
import torch
import torch.nn.functional as F


def _flatten_tensors(tensors, params):
    return torch.cat([
        (
            tensor.contiguous().view(-1)
            if tensor is not None
            else torch.zeros_like(param).view(-1)
        )
        for tensor, param in zip(tensors, params)
    ])


def compute_gradient_vector_lm(model, x, y, microbatch: int | None = None):
    """Return the gradient of mean LM loss as a flat vector."""
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)

    if microbatch is None or x.shape[0] <= microbatch:
        logits, _ = model(x, y)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return _flatten_tensors(grads, params)

    total_tokens = 0
    grad_accum = [torch.zeros_like(p) for p in params]
    for start in range(0, x.shape[0], microbatch):
        xb = x[start : start + microbatch]
        yb = y[start : start + microbatch]
        logits, _ = model(xb, yb)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            yb.reshape(-1),
            reduction="sum",
        )
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        for i, grad in enumerate(grads):
            if grad is not None:
                grad_accum[i] = grad_accum[i] + grad
        total_tokens += xb.numel()

    grad_accum = [grad / max(total_tokens, 1) for grad in grad_accum]
    return _flatten_tensors(grad_accum, params)


def hessian_vector_product_lm(model, x, y, vec):
    """
    Hessian-vector product for causal LM loss (nanoGPT-style forward).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    logits, _ = model(x, y)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grad_flat = _flatten_tensors(grads, params)
    
    vjp = (grad_flat * vec).sum()
    
    hvp_grads = torch.autograd.grad(vjp, params, allow_unused=True)
    hvp_flat = _flatten_tensors(hvp_grads, params)
    
    return hvp_flat


def power_iteration(matvec_fn, dim, num_iters=50, tol=1e-4, device=None, dtype=None):
    """
    Power iteration to find largest eigenvalue and eigenvector.
    
    Args:
        matvec_fn: function computing matrix-vector product
        dim: dimension of the space
        num_iters: maximum iterations
        tol: convergence tolerance
        device, dtype: torch device and dtype
    
    Returns:
        eigenvalue: largest eigenvalue (scalar)
        eigenvector: corresponding unit eigenvector
    """
    v = torch.randn(dim, device=device, dtype=dtype)
    v = v / v.norm()
    eigval = torch.tensor(0.0, device=device, dtype=dtype)
    
    for _ in range(num_iters):
        w = matvec_fn(v)
        norm_w = w.norm()
        if norm_w < 1e-12:
            break
        v_next = w / norm_w
        eigval_next = torch.dot(v_next, w)
        
        if torch.abs(eigval_next - eigval) < tol * torch.abs(eigval_next).clamp(min=1e-12):
            return eigval_next, v_next
        
        v = v_next
        eigval = eigval_next
    
    return eigval, v


def compute_top_eigenvectors(model, x, y, D, num_iters=50, tol=1e-4, device=None, dtype=None):
    """
    Compute top-D eigenvectors of the Hessian using power iteration with deflation.
    
    Args:
        model: neural network
        x, y: input data and targets
        D: number of top eigenvectors to compute
        num_iters: iterations per eigenvector
        tol: convergence tolerance
        device, dtype: torch device and dtype
    
    Returns:
        eigenvalues: tensor of shape (D,) with top eigenvalues
        U_D: matrix of shape (N, D) with eigenvectors as columns
    """
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    
    if device is None:
        device = next(model.parameters()).device
    if dtype is None:
        dtype = next(model.parameters()).dtype
    
    eigenvalues = []
    eigenvectors = []
    
    def base_matvec(v):
        return hessian_vector_product_lm(model, x, y, v)
    
    for i in range(D):
        if i == 0:
            matvec_fn = base_matvec
        else:
            U_prev = torch.stack(eigenvectors, dim=1)
            eigs_prev = torch.tensor(eigenvalues, device=device, dtype=dtype)
            
            def deflated_matvec(v, U=U_prev, lam=eigs_prev):
                Hv = base_matvec(v)
                for j in range(U.shape[1]):
                    u_j = U[:, j]
                    Hv = Hv - lam[j] * torch.dot(u_j, v) * u_j
                return Hv
            
            matvec_fn = deflated_matvec
        
        eigval, eigvec = power_iteration(
            matvec_fn, dim, num_iters=num_iters, tol=tol, device=device, dtype=dtype
        )
        
        eigenvalues.append(eigval.item())
        eigenvectors.append(eigvec)
    
    eigenvalues = torch.tensor(eigenvalues, device=device, dtype=dtype)
    U_D = torch.stack(eigenvectors, dim=1)
    
    return eigenvalues, U_D


def compress_hessian_to_basis(model, x, y, basis):
    """Return U^T H U for the LM Hessian and a given column-orthonormal basis U."""
    D = basis.shape[1]
    compressed = torch.zeros((D, D), device=basis.device, dtype=basis.dtype)
    for j in range(D):
        hvp = hessian_vector_product_lm(model, x, y, basis[:, j])
        compressed[:, j] = basis.transpose(0, 1) @ hvp
    return 0.5 * (compressed + compressed.transpose(0, 1))


def subspace_overlap(U_a, U_b, D_values: list[int]) -> dict[str, float]:
    """Mean principal cosine overlap for requested subspace dimensions."""
    overlap = {}
    for D in D_values:
        Ua = U_a[:, :D]
        Ub = U_b[:, :D]
        singular_values = torch.linalg.svdvals(Ua.transpose(0, 1) @ Ub)
        overlap[str(D)] = singular_values.mean().item()
    return overlap
