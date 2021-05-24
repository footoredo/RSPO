#!/bin/bash

for i in {5..9}
do
    z=0.$i
    ./scripts/run_batch.sh run_escalation_gw_1_prediction_alpha.sh 5 $z
done
