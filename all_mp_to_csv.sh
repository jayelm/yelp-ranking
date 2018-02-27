#!/usr/bin/env bash

set -e

for k in 0 1 2 3 4; do
    python3 mp_to_csv.py \
        --mfile "dataset_processed/matches.pkl.$k.train" \
        --bfile "dataset_processed/businesses.pkl.$k.train" \
        "results/mp_10.$k.npy"
done
