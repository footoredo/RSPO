#!/bin/bash

for i in {0..9}
do
    z=1.$i
    ./scripts/run_batch.sh run_escalation_gw_1_prediction_alpha.sh 5 $z
done
