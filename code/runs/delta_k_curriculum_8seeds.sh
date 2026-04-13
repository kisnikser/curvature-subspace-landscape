#!/usr/bin/env bash
# Launch 8 parallel ``run_delta_k_curriculum`` jobs (one seed + one GPU each).
#
# Live plots: each run refreshes ``<plot-prefix>_criteria.pdf`` and ``<plot-prefix>_val.pdf``
# after every k checkpoint. Do not pass ``--no-plots``.
#
# Usage (from repo ``code/`` directory):
#   bash runs/delta_k_curriculum_8seeds.sh
# Or override stems / GPUs:
#   DELTA_K_OUT_STEM=.nanochat/reports/my_run DELTA_K_GPUS=0,1,2,3,4,5,6,7 bash runs/delta_k_curriculum_8seeds.sh
#
# After all seeds finish, optionally runs merge (mean ± std, fill_between). Disable with SKIP_MERGE=1.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Prefix for outputs: ``{OUT_STEM}_seed{S}.json``, PDFs, and ``.log`` files in the same directory.
DELTA_K_OUT_STEM="${DELTA_K_OUT_STEM:-.nanochat/reports/delta_k_curriculum_d6_8seeds}"

# Eight seeds on eight devices (CUDA_VISIBLE_DEVICES per process).
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7}"
DELTA_K_GPUS="${DELTA_K_GPUS:-0,1,2,3,4,5,6,7}"

# Set SKIP_MERGE=1 to only produce per-seed JSON/PDFs.
SKIP_MERGE="${SKIP_MERGE:-0}"

# k grid: ``linear`` = evenly spaced in k (step ≈ (k_max−k_min)/(n_k_points−1), e.g. ~526 for 1…10000, n=20).
# Use ``log`` for log-spaced k. Override: DELTA_K_GRID=log
DELTA_K_GRID="${DELTA_K_GRID:-linear}"
DELTA_K_MIN="${DELTA_K_MIN:-1}"

# Row-chunk size for ``grad_flat_lm`` (exact ∇ of mean CE; full k rows in one pass → OOM at k≈500+).
# Override: DELTA_K_GRAD_MAX_SEQ=128
DELTA_K_GRAD_MAX_SEQ="${DELTA_K_GRAD_MAX_SEQ:-64}"

MERGE_FLAGS=( )
if [[ "$SKIP_MERGE" != "1" ]]; then
  MERGE_FLAGS+=(--merge-after)
fi

# Training / curriculum args (edit to match your experiment). Plots use defaults from run_delta_k_curriculum
# (--plot-figsize 7,4.5, Times font). Add e.g. ``--plot-figsize 8,5`` here if needed.
exec python -m scripts.launch_delta_k_curriculum_seeds \
  --out-stem "$DELTA_K_OUT_STEM" \
  --seeds "$SEEDS" \
  --gpus "$DELTA_K_GPUS" \
  "${MERGE_FLAGS[@]}" \
  -- \
  --depth 6 \
  --k-grid "${DELTA_K_GRID}" \
  --k-min "${DELTA_K_MIN}" \
  --k-max 10000 \
  --n-k-points 20 \
  --train-batch-size 16 \
  --grad-max-seq "${DELTA_K_GRAD_MAX_SEQ}" \
  --microbatch 1 \
  --hvp-microbatch 1
