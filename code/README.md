# Code

PyTorch utilities for landscape increment metrics (Δ₁, Δ₂, Δ₂⁽ᴰ⁾) on a small GPT-style LM, plus plotting for the paper.

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

- **JSON** is written to `output/landscape/landscape_experiments.json` (path from `config.yaml` → `common.output_dir`).
- **Scaling PDFs** are written to `paper/figures/` for inclusion in `main.tex`.

Optional: `python code/plot_scaling_from_json.py /path/to/custom.json`

## Config

`config.yaml` — model size, data path, sample sizes `k`, subspace dimensions `D`, seeds, and optimizer settings. If `data.text_path` is null, Tiny Shakespeare is downloaded under `data/tinyshakespeare.txt` when needed.

## Layout

| File / dir | Role |
|------------|------|
| `run_experiments.py` | Training loops, Δ metrics, JSON export |
| `plot_scaling_from_json.py` | Log–log figures from JSON |
| `config.yaml` | Experiment hyperparameters |
| `gpt/` | NanoGPT-style model |
| `criteria.py`, `eigenvectors.py` | Loss increments and top Hessian directions |
| `shared/` | Text data loading |
