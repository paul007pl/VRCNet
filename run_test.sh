#!/bin/bash

partition=vi_irdc_v100_32g
job_name=benchmark
gpus=1
g=$((${gpus}<8?${gpus}:8))

srun -u --partition=${partition} --job-name=${job_name} -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
     python test.py --config cfgs/$1.yaml
