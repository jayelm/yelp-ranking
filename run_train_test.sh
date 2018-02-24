#!/usr/bin/env bash

set -e

for k in 0 1 2 3 4; do
    python3 run_trueskill_mp.py --fast_m_acc \
        --mfile "dataset_processed/matches.pkl.$k.train" \
        --bfile "dataset_processed/businesses.pkl.$k.train" \
        --save "results/mp_{num_samples}.$k.npy"
done
