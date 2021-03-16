#!/bin/bash

for i in {1..32}
do
  /home/footoredo/anaconda3/envs/py3.8/bin/python /home/footoredo/PycharmProjects/pytorch-a2c-ppo-acktr-gail-footoredo/ma_main.py --algo=ppo --env-name=simple --num-processes=8 --num-agents=1 --num-steps=1024 --episode-steps=32 --num-env-steps=8192 --no-cuda --seed=5123 --num-mini-batch=8192 --ppo-epoch=1 --reseed-step=6553600 --reseed-z=1 --lr=3e-4 --value-loss-coef=0.7 --entropy-coef=0. --gamma=0.995 --gae-lambda=0.99 --no-grad-norm-clip --reward-normalization --save-interval=100 --save-dir=/tmp/ppo-simple/ --load-dir=./results/ppo-simple/2020-11-20T09:32:31.027288 --load-step=$i --load --task=grad-$i
done