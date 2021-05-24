#!/bin/bash

for i in {0..9}
do
    z=0.$i
    ./scripts/run_batch.sh run_stag_hunt_gw_no_fruit_prediction_alpha.sh 3 $z
done
