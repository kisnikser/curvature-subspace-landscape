# Code

PyTorch utilities for landscape increment metrics (`Δ₁`, `Δ₂`, `Δ₂⁽ᴰ⁾`) on
NanoGPT-style causal language models, plus plotting and reviewer-facing
diagnostics for the paper.

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires CUDA if you keep `experiment.device: cuda` in `config.yaml`; set `cpu` for a CPU-only run.

## Run

From the **repository root** (imports assume `code` on `PYTHONPATH` via the scripts):

```bash
python code/run_experiments.py
python code/plot_scaling_from_json.py
```

- **JSON** is written to `code/output/landscape/landscape_experiments.json` (path from `config.yaml` → `common.output_dir`).
- **Primary and ablation PDFs** are written to `paper/figures/` for inclusion in `main.tex`.

Optional: `python code/plot_scaling_from_json.py /path/to/custom.json`

## Config

`config.yaml` now contains:

- a shared base configuration
- a `run_matrix` with named settings
- reviewer-facing ablation controls (`sigma_values`, `main_subspace_dim`, `overlap_dims`)

If `data.text_path` is null, the runner downloads the requested corpus to `code/data/`.

## Layout

| File / dir | Role |
|------------|------|
| `run_experiments.py` | Training loops, Δ metrics, ablations, JSON export |
| `plot_scaling_from_json.py` | Main and ablation figures from JSON |
| `config.yaml` | Base config plus named run matrix |
| `data/` | Versioned input corpora used by the run matrix |
| `gpt/` | NanoGPT-style model |
| `criteria.py`, `eigenvectors.py` | Loss increments and top Hessian directions |
| `shared/` | Text data loading |

## Current Program

The default run matrix includes:

- `tiny_shakespeare_reference` for clean proof-of-concept scaling curves
- `wikitext2_medium` as a stronger validation setting

The JSON log stores:

- seed-aggregatable `Δ₁`, `Δ₂`, `Δ₂⁽ᴰ⁾`
- sigma sweeps
- Hessian vs random subspace comparisons
- recomputed vs frozen subspace comparisons
- eigenspace drift metrics
- quadratic-proxy alignment diagnostics
