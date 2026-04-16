#!/usr/bin/env bash
# Figure 4: quadratic-proxy relative error vs perturbation scale sigma.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-runs/fig4}
mkdir -p "$OUTDIR"

python -m experiments.fig4_quadratic_sigma.run \
    --model-tag d6 --model-step 3500 \
    --out "$OUTDIR/quadratic_sigma.json"

python -m experiments.fig4_quadratic_sigma.plot \
    --input "$OUTDIR/quadratic_sigma.json" \
    --out "$OUTDIR/fig4_quadratic_sigma.pdf"
