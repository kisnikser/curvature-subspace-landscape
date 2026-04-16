# Curvature-Aligned Probing for Local Loss-Landscape Stabilization

Anonymized code release. This package reproduces every figure and table in
the main text of the submission.

## Layout

```
code-submission/
├── nanochat/                    # model, dataloader, checkpoint manager
├── scripts/                     # core library
│   ├── delta_criteria_k.py      # Δ_1, Δ_2 definitions and estimators
│   └── delta2_subspace_estimators.py   # Δ_2^(D): Direct MC, Quadratic MC, Gaussian moment
├── experiments/
│   ├── fig2_delta_k_decay/      # Fig. 2: decay of Δ_1, Δ_2, Δ_2^(D) with k
│   ├── fig3_subspace_ratio/     # Fig. 3: Δ_2^(D) / Δ_2 across k, D, σ
│   ├── fig4_quadratic_sigma/    # Fig. 4: quadratic-proxy validity vs σ
│   ├── fig5_6_robustness/       # Figs. 5, 6: estimator S-curve + error heatmap
│   └── tab2_estimator_phases/   # Table 2: wall-clock phases per estimator
└── configs/                     # one bash script per figure/table
```

Each `experiments/<name>/run.py` writes a JSON log; `plot.py` consumes that
JSON and writes a PDF. Bash invocations that match the paper setup live in
`configs/`.

## Figure → script map

| Asset | Runner | Plotter | Config |
|---|---|---|---|
| Fig. 2 (Δ_k decay) | `experiments/fig2_delta_k_decay/run.py` | `.../plot.py` | `configs/run_fig2_fig3.sh` |
| Fig. 3 (subspace ratio) | same JSON as Fig. 2 | `experiments/fig3_subspace_ratio/plot.py` | `configs/run_fig2_fig3.sh` |
| Fig. 4 (quadratic σ) | `experiments/fig4_quadratic_sigma/run.py` | `.../plot.py` | `configs/run_fig4.sh` |
| Fig. 5 (S-curve) | `experiments/fig5_6_robustness/run.py` | `.../plot.py` | `configs/run_fig5_fig6.sh` |
| Fig. 6 (heatmap) | same JSON as Fig. 5 | `.../plot.py` | `configs/run_fig5_fig6.sh` |
| Table 2 (phases) | `experiments/tab2_estimator_phases/run.py` | (prints LaTeX) | `configs/run_tab2.sh` |

## Setup

Requires Python ≥ 3.10 and PyTorch ≥ 2.2 with CUDA. Install dependencies:

```bash
pip install -e .
```

or, with `uv`:

```bash
uv sync
```

The experiments load the `nanochat` depth-6 checkpoint at step 3500. Place
the checkpoint under the directory returned by `nanochat.common.get_base_dir()`
(by default `~/.cache/nanochat/`), or set `NANOCHAT_BASE_DIR` accordingly.

## Reproducing the paper

From `code-submission/`:

```bash
bash configs/run_fig2_fig3.sh     # Figs. 2, 3
bash configs/run_fig4.sh          # Fig. 4
bash configs/run_fig5_fig6.sh     # Figs. 5, 6
bash configs/run_tab2.sh          # Table 2
```

Outputs are written under `runs/<fig>/`. Override with
`OUTDIR=<path> bash configs/...`. Seeds for Fig. 2 / Fig. 3 default to
`SEEDS="0 1 2 3 4 5 6 7"`.

All runs are single-GPU; multi-seed runs for Fig. 2 / Fig. 3 can be
parallelized across GPUs by looping over seeds with different
`CUDA_VISIBLE_DEVICES`.

## Notes

- Precision is float32 throughout; autocast is disabled and the SDPA math
  kernel is used for numerical stability, as described in the paper.
- Subspace construction uses deflated power iteration on Hessian–vector
  products. Any HVP-based eigensolver (Lanczos, LOBPCG) is a drop-in
  replacement.
- All scripts accept `--help` for the full CLI.
