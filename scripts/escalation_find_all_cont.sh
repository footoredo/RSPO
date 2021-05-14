#!/bin/sh

python escalation_find_all.py \
    --initial-config="./configs/escalation-gw-find-all-stage-6-cont.json" \
    --default-likelihood=60.0 \
    --project-dir="./sync-results/escalation-gw/find-all-50-other-play-cont" \
    --runs-per-stage=1 \
    --max-stages=20