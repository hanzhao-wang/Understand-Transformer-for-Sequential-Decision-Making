#!/bin/bash
#SBATCH -J JOB_NAME
#SBATCH -A ACCOUNT_NAME
#SBATCH -p JOB_QUEUE
#SBATCH -N 1 --ntasks-per-node=2 --cpus-per-task=2 --gres=gpu:2
#SBATCH --time=20:00:00

script_path=$(realpath "$0")
parent_dir=$(dirname $script_path)
pparent_dir=$(dirname $parent_dir)
ppparent_dir=$(dirname $pparent_dir)

export PYTHONPATH="${PYTHONPATH}:${ppparent_dir}/src"

OMP_NUM_THREADS=1 python -W "ignore" -m torch.distributed.launch \
    --nproc_per_node=2  "${parent_dir}/NV_cts.py" \
    --quinine_config_path "${parent_dir}/4env_4d.yaml" 

