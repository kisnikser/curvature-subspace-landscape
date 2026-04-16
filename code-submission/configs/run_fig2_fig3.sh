#!/usr/bin/env bash
# Figures 2 and 3: decay of Delta_1 / Delta_2 / Delta_2^{(D)} with k and the
# subspace-to-full-space ratio Delta_2^{(D)} / Delta_2.
# One JSON per seed; seeds are then merged and plotted.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-runs/fig2_fig3}
mkdir -p "$OUTDIR"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}

for S in $SEEDS; do
    python -m experiments.fig2_delta_k_decay.run \
        --depth 6 \
        --k-max 10000 --n-k-points 20 \
        --sigmas "1e-4,1e-3,1e-2" \
        --D-list "1,4,16" \
        --seed "$S" \
        --out "$OUTDIR/delta_k_seed${S}.json"
done

python -m experiments.fig2_delta_k_decay.merge \
    --inputs "$OUTDIR"/delta_k_seed*.json \
    --out "$OUTDIR/delta_k_merged.json"

# Figure 2: the decay plot.
python -m experiments.fig2_delta_k_decay.plot \
    --input "$OUTDIR/delta_k_merged.json" \
    --out "$OUTDIR/fig2_delta_k_decay.pdf"

# Figure 3: the subspace-to-full-space ratio plot.
python -m experiments.fig3_subspace_ratio.plot \
    --input "$OUTDIR/delta_k_merged.json" \
    --mode gm_vs_full_ratio \
    --out "$OUTDIR/fig3_subspace_ratio.pdf"
