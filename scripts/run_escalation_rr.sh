#!/bin/sh

r=$(( RANDOM % 10 ))
env_config="./env-configs/escalation-gw-rr/-0.$r.json"

echo $r

python ma_main.py --config="./configs/escalation-gw-rr.json" --env-config=$env_config --seed=$1