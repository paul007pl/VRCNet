#!/bin/bash

partition=innova
job_name=benchmark
gpus=$2
g=$((${gpus}<8?${gpus}:8))

srun -u --partition=${partition} --job-name=${job_name} -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
     python train.py --config cfgs/$1.yaml
