#!/bin/sh

python find_all.py \
    --initial-config="./configs/half-cheetah-0-test.json" \
    --default-likelihood=1000.0 \
    --project-dir="./sync-results/half-cheetah/find-all-test" \
    --agents="0" \
    --keywords "normal" "reversed" "front_upright" "back_upright" \
    --runs-per-stage=1 \
    --max-stages=20
