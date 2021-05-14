#!/bin/sh

python ma_main.py --config="./configs/stag-hunt-gw-no-fruit-prediction.json" \
  --exploration-reward-alpha=$1 \
  --save-dir="./sync-results/stag-hunt-gw/no-fruit-prediction-$1" \
  --seed=$2