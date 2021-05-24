#!/bin/bash

script_file=$1
shift
repeat_num=$1
shift
rest_args=$@

for i in $(seq 1 $repeat_num)
do
    ./scripts/$script_file $rest_args $RANDOM
done
