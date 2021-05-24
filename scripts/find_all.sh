#!/bin/bash

initial_config=$1
project_dir=$2
num_stages=$3
num_runs=$4
default_likelihood=$5

project_dir=$project_dir/`python generate_timestamp.py`

echo $project_dir

for (( i=0 ; i<$num_stages ; i++ ))
do
  python find_all_shell.py --initial-config=$initial_config --default-likelihood=$default_likelihood \
      --project-dir=$project_dir --current-stage=$i
  echo Stage-$i
  config_file=`python generate_timestamp.py`
  cp $project_dir/stage-$i/config.json ./configs/$config_file.json
  ./scripts/run_batch.sh run_config.sh $num_runs $config_file > /dev/null 2> /dev/null
#  ./scripts/run_batch.sh run_config.sh $num_runs $config_file
  rm ./configs/$config_file.json
done

python find_all_shell.py --initial-config=$initial_config --default-likelihood=$default_likelihood \
      --project-dir=$project_dir --current-stage=$num_stages