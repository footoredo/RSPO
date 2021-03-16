#!/bin/bash

for i in $(seq 1 $2)
do
    ./run.sh $RANDOM $1 $3
done
