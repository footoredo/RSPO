#!/bin/bash

for i in $(seq 1 $1)
do
    ./scripts/run_stag_hunt_gw_no_fruit.sh $RANDOM
done
