#!/bin/bash
set -e
WD=/home/jovyan/shares/SR008.fs2/nkiselev/sandbox/curvature-subspace-landscape/code
PY=/home/jovyan/nkiselev/.conda/nkiselev-kandinsky-cuda12.8-transformers/bin/python
LOGDIR=$WD/output/landscape

mkdir -p $LOGDIR

CUDA_VISIBLE_DEVICES=0 $PY -u $WD/run_proxy_sigma_study.py --setting tiny_shakespeare_reference --seed 42 --gpu 0 > $LOGDIR/proxy_log_ts42.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u $WD/run_proxy_sigma_study.py --setting tiny_shakespeare_reference --seed 43 --gpu 0 > $LOGDIR/proxy_log_ts43.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u $WD/run_proxy_sigma_study.py --setting tiny_shakespeare_reference --seed 44 --gpu 0 > $LOGDIR/proxy_log_ts44.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -u $WD/run_proxy_sigma_study.py --setting wikitext2_medium --seed 42 --gpu 0 > $LOGDIR/proxy_log_wt42.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY -u $WD/run_proxy_sigma_study.py --setting wikitext2_medium --seed 43 --gpu 0 > $LOGDIR/proxy_log_wt43.txt 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY -u $WD/run_proxy_sigma_study.py --setting wikitext2_medium --seed 44 --gpu 0 > $LOGDIR/proxy_log_wt44.txt 2>&1 &

echo "Launched 6 processes: $(jobs -p)"
wait
echo "All proxy study jobs completed"
