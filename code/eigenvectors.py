"""
Compute top-D eigenvectors of the Hessian matrix.

Uses power iteration with deflation to find the largest eigenvalues
and corresponding eigenvectors.
"""
import torch
import torch.nn.functional as F


def hessian_vector_product_lm(model, x, y, vec):
    """
    Hessian-vector product for causal LM loss (nanoGPT-style forward).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    logits, _ = model(x, y)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grad_flat = torch.cat([
        (g.view(-1) if g is not None else torch.zeros_like(p).view(-1))
        for g, p in zip(grads, params)
    ])
    
    vjp = (grad_flat * vec).sum()
    
    hvp_grads = torch.autograd.grad(vjp, params, allow_unused=True)
    hvp_flat = torch.cat([
        (g.contiguous().view(-1) if g is not None else torch.zeros_like(p).view(-1))
        for g, p in zip(hvp_grads, params)
    ])
    
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
