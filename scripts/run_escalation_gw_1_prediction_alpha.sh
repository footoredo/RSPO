#!/bin/sh

python ma_main.py --config="./configs/escalation-gw-1-prediction-no-stop.json" \
  --prediction-reward-alpha=$1 \
  --save-dir="./sync-results/escalation-gw/1-prediction-no-stop-alpha-$1" \
  --use-wandb \
  --wandb-group="1-prediction-no-stop-alpha" \
  --seed=$2