#!/bin/sh

python ma_main.py --algo=ppo --env-name=stag-hunt-gw --parallel-limit=32 --num-processes=32 --num-agents=2 --num-steps=1600 --episode-steps=50 --num-env-steps=40960000 --no-cuda --seed=$1 --num-mini-batch=1 --ppo-epoch=4 --reseed-step=-1 --reseed-z=7 --lr=1e-3 --value-loss-coef=1 --entropy-coef=0.01 --gamma=0.99 --gae-lambda=0.95 --save-interval=100 --save-dir="./sync-results/stag-hunt-gw/first" --load-dir="./results/stag-hunt-gw/2020-12-09T17:22:49.027992" --load-step=32 --direction=0 --guided-updates=0 --no-play --no-train-in-turn
