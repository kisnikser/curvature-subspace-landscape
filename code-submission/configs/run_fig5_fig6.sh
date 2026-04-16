#!/usr/bin/env bash
# Figures 5 and 6: Direct / Quadratic MC convergence toward the Gaussian-moment
# closed form (S-curve), and the relative-error heatmap over (sigma, D).
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-runs/fig5_fig6}
mkdir -p "$OUTDIR"

python -m experiments.fig5_6_robustness.run \
    --model-tag d6 --model-step 3500 \
    --out "$OUTDIR/delta2_robustness.json"

python -m experiments.fig5_6_robustness.plot \
    --input "$OUTDIR/delta2_robustness.json" \
    --out-scurve "$OUTDIR/fig5_scurve.pdf" \
    --out-heatmap "$OUTDIR/fig6_heatmap.pdf"
