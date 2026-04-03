"""
Focused experiment: proxy validity as a function of sigma.

For each (setting, seed, k), retrains the model and evaluates the
direct subspace criterion and GM/quadMC proxy across a fine sigma grid.
Produces a JSON file for plotting ratio(proxy/direct) vs sigma at each k.

Usage:
    python run_proxy_sigma_study.py [--setting NAME] [--seed S] [--gpu GPU_ID]
"""

import sys, json, argparse
from pathlib import Path

_code_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_code_root))

import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from gpt.model import GPT, GPTConfig
from criteria import (
    compute_delta2_subspace, compute_increment,
    _get_flat_params, _set_flat_params,
)
from eigenvectors import (
    compute_gradient_vector_lm, compute_top_eigenvectors,
    compress_hessian_to_basis,
)
from shared.text_data import (
    TextChunkDataset, build_char_vocab, encode,
    ensure_text_corpus, load_text_corpus,
)
from run_experiments import (
    gpt_from_conf, get_subset_data, trim_for_hessian,
    split_train_validation_indices, default_text_path,
    gaussian_moment_estimator, quadratic_mc_estimator,
    train_model, build_run_matrix,
)

FINE_SIGMAS = [0.0001, 0.0003, 0.001, 0.002, 0.003, 0.005, 0.007,
               0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]


def run_proxy_study(conf, setting_name, seed, device, dtype):
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
    validation_sequences = int(getattr(conf.experiment, "validation_sequences", 0))
    train_pool, _ = split_train_validation_indices(
        n_total=n_total, max_k=max_k, validation_sequences=validation_sequences
    )

    mb = int(conf.experiment.loss_microbatch)
    hmax = int(conf.experiment.hessian_max_sequences)
    ns = int(conf.experiment.delta2_num_samples)
    main_D = int(conf.experiment.main_subspace_dim)

    results = []

    for k in sample_sizes:
        print(f"  k={k}", flush=True)

        model = gpt_from_conf(conf)
        model.to(device)

        indices_k = train_pool[:k]
        train_subset = Subset(dataset, indices_k)
        train_loader = DataLoader(
            train_subset,
            batch_size=min(int(conf.data.batch_size), k),
            shuffle=True,
        )
        betas = list(conf.experiment.betas)
        model, final_loss = train_model(
            model, train_loader,
            lr=float(conf.experiment.lr),
            weight_decay=float(conf.experiment.weight_decay),
            betas=betas,
            num_epochs=int(conf.experiment.train_epochs),
            device=device,
        )
        print(f"    loss={final_loss:.6f}", flush=True)

        w_k = _get_flat_params(model).clone()

        x_k, y_k = get_subset_data(dataset, indices_k)
        x_k, y_k = x_k.to(device), y_k.to(device)

        indices_k1 = train_pool[:k + 1]
        x_k1, y_k1 = get_subset_data(dataset, indices_k1)
        x_k1, y_k1 = x_k1.to(device), y_k1.to(device)

        xh, yh = trim_for_hessian(x_k, y_k, hmax)
        xh1, yh1 = trim_for_hessian(x_k1, y_k1, hmax)

        _set_flat_params(model, w_k)
        eigenvalues, U_full = compute_top_eigenvectors(
            model, xh, yh, main_D,
            num_iters=int(conf.experiment.top_eigenvec_iters),
            tol=float(conf.experiment.top_eigenvec_tol),
            device=device, dtype=dtype,
        )
        U_main = U_full[:, :main_D]

        _set_flat_params(model, w_k)
        a_k = compute_increment(model, x_k, y_k, x_k1, y_k1, microbatch=mb)

        grad_mb = int(getattr(conf.experiment, "gradient_microbatch", mb))
        grad_k = compute_gradient_vector_lm(model, xh, yh, microbatch=grad_mb)
        grad_k1 = compute_gradient_vector_lm(model, xh1, yh1, microbatch=grad_mb)
        grad_diff = grad_k1 - grad_k
        c_k = U_main.T @ grad_diff

        Hk_compressed = compress_hessian_to_basis(model, xh, yh, U_main)
        Hk1_compressed = compress_hessian_to_basis(model, xh1, yh1, U_main)
        B_k = Hk1_compressed - Hk_compressed

        sigma_data = {}
        for sigma in FINE_SIGMAS:
            _set_flat_params(model, w_k)
            direct_stats = compute_delta2_subspace(
                model, w_k, U_main, x_k, y_k, x_k1, y_k1,
                sigma=sigma, num_samples=ns,
                device=device, dtype=dtype, microbatch=mb, return_details=True,
            )
            gm_val = gaussian_moment_estimator(a_k, c_k, B_k, sigma)
            qmc = quadratic_mc_estimator(
                a_k, c_k, B_k, sigma,
                num_samples=int(conf.experiment.alignment_num_samples),
                device=device, dtype=dtype,
            )
            sigma_data[f"{sigma:.6f}"] = {
                "sigma": sigma,
                "direct_mean": direct_stats["mean"],
                "direct_std": direct_stats.get("std", 0),
                "gm_value": gm_val,
                "quadMC_mean": qmc["mean"],
                "ratio_gm": gm_val / max(direct_stats["mean"], 1e-30),
                "ratio_qmc": qmc["mean"] / max(direct_stats["mean"], 1e-30),
            }
            print(f"    sigma={sigma:.4f}: direct={direct_stats['mean']:.2e}  "
                  f"GM={gm_val:.2e}  ratio={gm_val / max(direct_stats['mean'], 1e-30):.2f}",
                  flush=True)

        results.append({
            "setting_name": setting_name,
            "seed": seed,
            "k": k,
            "a_k": a_k,
            "c_k_norm": c_k.norm().item(),
            "B_k_fro": torch.linalg.norm(B_k).item(),
            "tr_B_k": torch.trace(B_k).item(),
            "sigma_sweep": sigma_data,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    conf = OmegaConf.load(Path(__file__).resolve().parent / "config.yaml")

    out_dir = Path(__file__).resolve().parent / "output" / "landscape"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    base_seed = int(conf.common.seed)
    for setting_name, primary, description, setting_conf in build_run_matrix(conf):
        if args.setting and setting_name != args.setting:
            continue
        setting_conf.data.text_path = str(default_text_path(setting_conf))
        for s in range(int(setting_conf.experiment.num_seeds)):
            seed = base_seed + s
            if args.seed is not None and seed != args.seed:
                continue
            print(f"=== {setting_name} seed={seed} gpu={args.gpu} ===", flush=True)
            res = run_proxy_study(setting_conf, setting_name, seed, device, dtype)
            all_results.extend(res)

    suffix = ""
    if args.setting:
        suffix += f"_{args.setting}"
    if args.seed is not None:
        suffix += f"_s{args.seed}"
    out_path = out_dir / f"proxy_sigma_study{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {len(all_results)} entries to {out_path}", flush=True)


if __name__ == "__main__":
    main()
