#!/bin/sh

python stag_hunt_find_all.py \
    --initial-config="./configs/stag-hunt-gw-no-fruit-prediction.json" \
    --default-likelihood=200.0 \
    --project-dir="./sync-results/stag-hunt-gw/find-all" \
    --runs-per-stage=1 \
    --max-stages=20