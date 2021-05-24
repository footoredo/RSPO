#!/bin/sh

python escalation_find_all.py \
    --initial-config="./configs/escalation-gw-1-prediction.json" \
    --default-likelihood=60.0 \
    --project-dir="./sync-results/escalation-gw/find-all-50-other-play" \
    --runs-per-stage=1 \
    --max-stages=20