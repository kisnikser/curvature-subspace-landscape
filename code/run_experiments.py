"""
Run the A-level landscape experiment program around a NanoGPT-style causal LM.

This runner supports:
- multiple named settings via a run matrix
- reviewer-facing ablations (D, random-vs-Hessian, frozen-vs-recomputed subspaces)
- sigma sweeps for locality checks
- theory-alignment diagnostics for the quadratic approximation
"""
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_code_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_code_root))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from gpt.model import GPT, GPTConfig
from criteria import (
    compute_delta1,
    compute_delta2,
    compute_delta2_subspace,
    compute_increment,
    quadratic_form_expectation,
    _get_flat_params,
    _set_flat_params,
)
from eigenvectors import (
    compute_gradient_vector_lm,
    compute_top_eigenvectors,
    compress_hessian_to_basis,
    subspace_overlap,
)
from shared.text_data import (
    TextChunkDataset,
    build_char_vocab,
    encode,
    ensure_text_corpus,
    load_text_corpus,
)


def gpt_from_conf(conf) -> GPT:
    m = conf.model
    cfg = GPTConfig(
        block_size=int(m.block_size),
        vocab_size=int(m.vocab_size),
        n_layer=int(m.n_layer),
        n_head=int(m.n_head),
        n_embd=int(m.n_embd),
        dropout=float(m.dropout),
        bias=bool(m.bias),
    )
    return GPT(cfg)


def train_model(model, train_loader, lr, weight_decay, betas, num_epochs, device):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=tuple(betas)
    )

    for _ in range(num_epochs):
        total_loss = 0.0
        ntok = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.numel()
            ntok += x.numel()

        avg_loss = total_loss / max(ntok, 1)

    model.eval()
    return model, avg_loss


def get_subset_data(dataset, indices):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
    x, y = next(iter(loader))
    return x, y


def trim_for_hessian(x, y, max_seq: int):
    if x.shape[0] <= max_seq:
        return x, y
    # Sample across the full nested subset so H_k and H_{k+1} do not collapse
    # to the same prefix once k exceeds max_seq.
    idx = torch.linspace(0, x.shape[0] - 1, steps=max_seq, device=x.device)
    idx = torch.round(idx).to(torch.long)
    idx = torch.unique_consecutive(idx)
    if idx.numel() < max_seq:
        pad = torch.arange(x.shape[0] - (max_seq - idx.numel()), x.shape[0], device=x.device)
        idx = torch.unique(torch.cat([idx, pad], dim=0), sorted=True)
        idx = idx[-max_seq:]
    return x[idx].contiguous(), y[idx].contiguous()


def default_text_path(conf) -> Path:
    corpus_name = str(getattr(conf.data, "corpus_name", "tiny_shakespeare"))
    if getattr(conf.data, "text_path", None):
        return Path(conf.data.text_path)
    filename = {
        "tiny_shakespeare": "tinyshakespeare.txt",
        "wikitext2": "wikitext2_train.txt",
    }.get(corpus_name, f"{corpus_name}.txt")
    return _repo_root / "code" / "data" / filename


def build_run_matrix(conf):
    run_matrix = list(getattr(conf, "run_matrix", []))
    if not run_matrix:
        default_name = str(getattr(conf.data, "corpus_name", getattr(conf.data, "name", "default")))
        return [(default_name, True, "single-setting default", conf)]

    base_container = OmegaConf.to_container(conf, resolve=False)
    settings = []
    for idx, preset in enumerate(run_matrix):
        preset_container = OmegaConf.to_container(preset, resolve=False)
        name = str(preset_container.pop("name"))
        primary = bool(preset_container.pop("primary", idx == 0))
        description = str(preset_container.pop("description", ""))
        merged = OmegaConf.merge(
            OmegaConf.create(base_container),
            OmegaConf.create(preset_container),
        )
        settings.append((name, primary, description, merged))
    return settings


def random_orthonormal_basis(dim: int, D: int, seed: int, device, dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mat = torch.randn((dim, D), generator=generator, dtype=dtype)
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q.to(device=device, dtype=dtype)


def summarize_alignment(true_values, approx_values):
    true_t = torch.tensor(true_values, dtype=torch.float64)
    approx_t = torch.tensor(approx_values, dtype=torch.float64)
    rel_error = (approx_t.mean() - true_t.mean()).abs() / true_t.mean().clamp(min=1e-12)
    if true_t.numel() > 1:
        corr = torch.corrcoef(torch.stack([true_t, approx_t]))[0, 1].item()
    else:
        corr = 1.0
    return {
        "true_mean": true_t.mean().item(),
        "quadratic_mean": approx_t.mean().item(),
        "relative_mean_error": rel_error.item(),
        "squared_increment_mse": torch.mean((true_t - approx_t) ** 2).item(),
        "correlation": corr,
        "num_samples": int(true_t.numel()),
    }


def compute_quadratic_alignment(
    model,
    w_k,
    basis,
    x_k,
    y_k,
    x_k1,
    y_k1,
    sigma,
    num_samples,
    a_k,
    c_k,
    B_k,
    device,
    dtype,
    microbatch,
):
    true_sq = []
    quadratic_sq = []

    for _ in range(num_samples):
        z = torch.randn(basis.shape[1], device=device, dtype=dtype) * sigma
        w_sample = w_k + basis @ z
        _set_flat_params(model, w_sample)
        true_increment = compute_increment(
            model, x_k, y_k, x_k1, y_k1, microbatch=microbatch
        )
        quadratic_increment = (
            a_k + torch.dot(c_k, z).item() + 0.5 * torch.dot(z, B_k @ z).item()
        )
        true_sq.append(true_increment ** 2)
        quadratic_sq.append(quadratic_increment ** 2)

    _set_flat_params(model, w_k)
    return summarize_alignment(true_sq, quadratic_sq)


def run_single_experiment(conf, setting_name, primary, description, seed, device, dtype):
    torch.manual_seed(seed)

    text_path = default_text_path(conf)
    ensure_text_corpus(
        text_path,
        corpus_name=str(getattr(conf.data, "corpus_name", "tiny_shakespeare")),
        url_override=getattr(conf.data, "corpus_url", None),
    )
    text = load_text_corpus(text_path)
    stoi, _chars = build_char_vocab(text)
    data_tensor = encode(text, stoi)
    conf.model.vocab_size = len(stoi)

    block_size = int(conf.model.block_size)
    stride = int(conf.data.stride)
    dataset = TextChunkDataset(data_tensor, block_size=block_size, stride=stride)

    n_total = len(dataset)
    sample_sizes = list(conf.experiment.sample_sizes)
    max_k = max(sample_sizes)
    if max_k + 1 > n_total:
        raise ValueError(
            f"Need at least max(k)+1 = {max_k + 1} text chunks; have {n_total}. "
            "Use a longer corpus or smaller block_size / stride."
        )

    perm = torch.randperm(n_total).tolist()
    mb = int(conf.experiment.loss_microbatch)
    grad_mb = int(getattr(conf.experiment, "gradient_microbatch", mb))
    hmax = int(conf.experiment.hessian_max_sequences)
    main_sigma = float(conf.experiment.delta2_sigma)
    sigma_values = sorted(set(float(s) for s in conf.experiment.sigma_values))
    subspace_dims = list(conf.experiment.subspace_dims)
    max_D = max(subspace_dims)
    overlap_dims = list(getattr(conf.experiment, "overlap_dims", subspace_dims))
    main_D = int(conf.experiment.main_subspace_dim)
    if main_D not in subspace_dims:
        raise ValueError(f"main_subspace_dim={main_D} must be included in subspace_dims")

    results = []
    frozen_hessian_basis = None
    frozen_hessian_anchor_k = None

    for k in sample_sizes:
        print(f"  Processing k={k}")

        model = gpt_from_conf(conf)
        model.to(device)

        indices_k = perm[:k]
        train_subset = Subset(dataset, indices_k)
        train_loader = DataLoader(
            train_subset,
            batch_size=min(int(conf.data.batch_size), k),
            shuffle=True,
        )

        betas = list(conf.experiment.betas)
        model, final_loss = train_model(
            model,
            train_loader,
            lr=float(conf.experiment.lr),
            weight_decay=float(conf.experiment.weight_decay),
            betas=betas,
            num_epochs=int(conf.experiment.train_epochs),
            device=device,
        )
        print(f"    Training loss (token CE): {final_loss:.6f}")

        w_k = _get_flat_params(model).clone()

        x_k, y_k = get_subset_data(dataset, indices_k)
        x_k = x_k.to(device)
        y_k = y_k.to(device)

        indices_k1 = perm[: k + 1]
        x_k1, y_k1 = get_subset_data(dataset, indices_k1)
        x_k1 = x_k1.to(device)
        y_k1 = y_k1.to(device)

        xh, yh = trim_for_hessian(x_k, y_k, hmax)
        xh1, yh1 = trim_for_hessian(x_k1, y_k1, hmax)

        _set_flat_params(model, w_k)
        delta1_stats = compute_delta1(
            model,
            w_k,
            x_k,
            y_k,
            x_k1,
            y_k1,
            eps=float(conf.experiment.delta1_eps),
            num_directions=int(conf.experiment.delta1_num_directions),
            device=device,
            dtype=dtype,
            microbatch=mb,
            return_details=True,
        )

        delta2_sigma_sweep = {}
        delta2_stats = None
        for sigma in sigma_values:
            _set_flat_params(model, w_k)
            stats = compute_delta2(
                model,
                w_k,
                x_k,
                y_k,
                x_k1,
                y_k1,
                sigma=sigma,
                num_samples=int(conf.experiment.delta2_num_samples),
                device=device,
                dtype=dtype,
                microbatch=mb,
                return_details=True,
            )
            delta2_sigma_sweep[f"{sigma:.4f}"] = stats
            if abs(sigma - main_sigma) < 1e-12:
                delta2_stats = stats

        if delta2_stats is None:
            raise RuntimeError("Main sigma was not included in sigma_values")

        _set_flat_params(model, w_k)
        eigenvalues, U_full = compute_top_eigenvectors(
            model,
            xh,
            yh,
            max_D,
            num_iters=int(conf.experiment.top_eigenvec_iters),
            tol=float(conf.experiment.top_eigenvec_tol),
            device=device,
            dtype=dtype,
        )
        if frozen_hessian_basis is None:
            frozen_hessian_basis = U_full.clone()
            frozen_hessian_anchor_k = k

        _set_flat_params(model, w_k)
        eigenvalues_k1, U_full_k1 = compute_top_eigenvectors(
            model,
            xh1,
            yh1,
            max_D,
            num_iters=int(conf.experiment.top_eigenvec_iters),
            tol=float(conf.experiment.top_eigenvec_tol),
            device=device,
            dtype=dtype,
        )

        delta2_subspace = {}
        delta2_subspace_stats = {}
        strategy_comparison = {"hessian": {}, "random": {}}
        refresh_comparison = {"recompute": {}, "freeze": {}}
        random_basis = random_orthonormal_basis(
            U_full.shape[0],
            max_D,
            seed=seed * 10_000 + k,
            device=device,
            dtype=dtype,
        )
        for D in subspace_dims:
            U_D = U_full[:, :D]
            hessian_stats = compute_delta2_subspace(
                model,
                w_k,
                U_D,
                x_k,
                y_k,
                x_k1,
                y_k1,
                sigma=main_sigma,
                num_samples=int(conf.experiment.delta2_num_samples),
                device=device,
                dtype=dtype,
                microbatch=mb,
                return_details=True,
            )
            delta2_subspace[str(D)] = hessian_stats["mean"]
            delta2_subspace_stats[str(D)] = hessian_stats
            strategy_comparison["hessian"][str(D)] = hessian_stats
            refresh_comparison["recompute"][str(D)] = hessian_stats

            random_stats = compute_delta2_subspace(
                model,
                w_k,
                random_basis[:, :D],
                x_k,
                y_k,
                x_k1,
                y_k1,
                sigma=main_sigma,
                num_samples=int(conf.experiment.delta2_num_samples),
                device=device,
                dtype=dtype,
                microbatch=mb,
                return_details=True,
            )
            strategy_comparison["random"][str(D)] = random_stats

            frozen_stats = compute_delta2_subspace(
                model,
                w_k,
                frozen_hessian_basis[:, :D],
                x_k,
                y_k,
                x_k1,
                y_k1,
                sigma=main_sigma,
                num_samples=int(conf.experiment.delta2_num_samples),
                device=device,
                dtype=dtype,
                microbatch=mb,
                return_details=True,
            )
            refresh_comparison["freeze"][str(D)] = frozen_stats

        _set_flat_params(model, w_k)
        a_k = compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch=mb)
        grad_k = compute_gradient_vector_lm(model, x_k, y_k, microbatch=grad_mb)
        grad_k1 = compute_gradient_vector_lm(model, x_k1, y_k1, microbatch=grad_mb)
        grad_diff = grad_k1 - grad_k
        U_main = U_full[:, :main_D]
        c_k = U_main.transpose(0, 1) @ grad_diff
        Hk_compressed = compress_hessian_to_basis(model, xh, yh, U_main)
        Hk1_compressed = compress_hessian_to_basis(model, xh1, yh1, U_main)
        B_k = Hk1_compressed - Hk_compressed
        term_contributions = quadratic_form_expectation(
            a_k=a_k,
            c_k=c_k,
            B_k=B_k,
            num_samples=int(conf.experiment.alignment_num_samples),
            sigma=main_sigma,
            device=device,
            dtype=dtype,
        )
        quadratic_alignment = compute_quadratic_alignment(
            model=model,
            w_k=w_k,
            basis=U_main,
            x_k=x_k,
            y_k=y_k,
            x_k1=x_k1,
            y_k1=y_k1,
            sigma=main_sigma,
            num_samples=int(conf.experiment.alignment_num_samples),
            a_k=a_k,
            c_k=c_k,
            B_k=B_k,
            device=device,
            dtype=dtype,
            microbatch=mb,
        )
        eigenspace_drift = subspace_overlap(U_full, U_full_k1, overlap_dims)

        row = {
            "setting_name": setting_name,
            "setting_primary": primary,
            "setting_description": description,
            "corpus_name": str(getattr(conf.data, "corpus_name", "tiny_shakespeare")),
            "text_path": str(text_path),
            "seed": seed,
            "k": k,
            "delta1": delta1_stats["mean"],
            "delta1_stats": delta1_stats,
            "delta2": delta2_stats["mean"],
            "delta2_stats": delta2_stats,
            "delta2_sigma_sweep": delta2_sigma_sweep,
            "delta2_subspace": delta2_subspace,
            "delta2_subspace_stats": delta2_subspace_stats,
            "subspace_strategy_comparison": strategy_comparison,
            "subspace_refresh_comparison": refresh_comparison,
            "top_eigenvalues": eigenvalues[:max_D].tolist(),
            "top_eigenvalues_k1": eigenvalues_k1[:max_D].tolist(),
            "eigenspace_drift": eigenspace_drift,
            "quadratic_alignment": quadratic_alignment,
            "term_contributions": term_contributions,
            "value_gap_at_wk": a_k,
            "gradient_gap_norm": grad_diff.norm().item(),
            "compressed_hessian_gap_fro": torch.linalg.norm(B_k).item(),
            "frozen_hessian_anchor_k": frozen_hessian_anchor_k,
            "hessian_sequences_used": int(xh.shape[0]),
            "hessian_sequences_used_k1": int(xh1.shape[0]),
            "final_loss": final_loss,
            "model_summary": {
                "block_size": int(conf.model.block_size),
                "n_layer": int(conf.model.n_layer),
                "n_head": int(conf.model.n_head),
                "n_embd": int(conf.model.n_embd),
            },
        }
        results.append(row)

        print(
            "    "
            f"Delta_1={delta1_stats['mean']:.6f}, "
            f"Delta_2={delta2_stats['mean']:.8f}, "
            f"Delta_2^(D={main_D})={delta2_subspace[str(main_D)]:.8f}"
        )

    return results


def main(conf=None):
    if conf is None:
        conf_path = Path(__file__).parent / "config.yaml"
        conf = OmegaConf.load(conf_path)

    OmegaConf.resolve(conf)

    device = torch.device(
        conf.experiment.device if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32

    all_results = []
    base_seed = conf.common.seed
    for setting_name, primary, description, setting_conf in build_run_matrix(conf):
        print(f"Running setting {setting_name}")
        setting_conf.data.text_path = str(default_text_path(setting_conf))
        for s in range(int(setting_conf.experiment.num_seeds)):
            seed = base_seed + s
            print(f"  Running experiment with seed {seed}")
            results = run_single_experiment(
                setting_conf,
                setting_name=setting_name,
                primary=primary,
                description=description,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            all_results.extend(results)

    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "landscape_experiments.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
