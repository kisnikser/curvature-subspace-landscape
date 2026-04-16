#!/usr/bin/env bash
# Table 2: wall-clock phases for the three Delta_2^{(D)} estimators.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR=${OUTDIR:-runs/tab2}
mkdir -p "$OUTDIR"

python -m experiments.tab2_estimator_phases.run \
    --model-tag d6 --model-step 3500 \
    --warmup 1 --repeats 5 \
    --out "$OUTDIR/estimator_phases.json"
