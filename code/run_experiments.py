"""
Run landscape convergence experiments with a NanoGPT-style causal LM.

Measures Delta_1, Delta_2, Delta_2^(D) as a function of training-set size k (number of text chunks).
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
    _get_flat_params,
    _set_flat_params,
)
from eigenvectors import compute_top_eigenvectors
from shared.text_data import (
    TextChunkDataset,
    build_char_vocab,
    encode,
    ensure_tiny_shakespeare,
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
    return x[:max_seq].contiguous(), y[:max_seq].contiguous()


def run_single_experiment(conf, seed, device, dtype):
    torch.manual_seed(seed)

    text_path = Path(conf.data.text_path)
    ensure_tiny_shakespeare(text_path)
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
    hmax = int(conf.experiment.hessian_max_sequences)

    results = []

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

        _set_flat_params(model, w_k)
        delta1 = compute_delta1(
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
        )

        delta2 = compute_delta2(
            model,
            w_k,
            x_k,
            y_k,
            x_k1,
            y_k1,
            sigma=float(conf.experiment.delta2_sigma),
            num_samples=int(conf.experiment.delta2_num_samples),
            device=device,
            dtype=dtype,
            microbatch=mb,
        )

        subspace_dims = list(conf.experiment.subspace_dims)
        max_D = max(subspace_dims)

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

        delta2_subspace = {}
        for D in subspace_dims:
            U_D = U_full[:, :D]
            delta2_D = compute_delta2_subspace(
                model,
                w_k,
                U_D,
                x_k,
                y_k,
                x_k1,
                y_k1,
                sigma=float(conf.experiment.delta2_sigma),
                num_samples=int(conf.experiment.delta2_num_samples),
                device=device,
                dtype=dtype,
                microbatch=mb,
            )
            delta2_subspace[D] = delta2_D

        row = {
            "seed": seed,
            "k": k,
            "delta1": delta1,
            "delta2": delta2,
            "delta2_subspace": {str(a): b for a, b in delta2_subspace.items()},
            "top_eigenvalues": eigenvalues[:max_D].tolist(),
            "final_loss": final_loss,
        }
        results.append(row)

        print(f"    Delta_1={delta1:.6f}, Delta_2={delta2:.8f}")

    return results


def main(conf=None):
    if conf is None:
        conf_path = Path(__file__).parent / "config.yaml"
        conf = OmegaConf.load(conf_path)

    if getattr(conf.data, "text_path", None) is None:
        conf.data.text_path = str(_repo_root / "data" / "tinyshakespeare.txt")

    OmegaConf.resolve(conf)

    device = torch.device(
        conf.experiment.device if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float32

    all_results = []
    base_seed = conf.common.seed

    for s in range(int(conf.experiment.num_seeds)):
        seed = base_seed + s
        print(f"Running experiment with seed {seed}")
        results = run_single_experiment(conf, seed, device, dtype)
        all_results.extend(results)

    out_dir = _repo_root / conf.common.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "landscape_experiments.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
