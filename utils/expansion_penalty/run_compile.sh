partition=ips_share
job_name=compile
gpus=1
g=$((${gpus}<8?${gpus}:8))


srun -u --partition=${partition} --job-name=${job_name} \
    -n1 --gres=gpu:${gpus} --ntasks-per-node=1 \
    python3 setup.py install
