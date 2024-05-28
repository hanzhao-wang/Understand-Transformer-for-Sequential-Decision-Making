#!/bin/bash
#SBATCH -J JOB_NAME
#SBATCH -A ACCOUNT_NAME
#SBATCH -p JOB_QUEUE
#SBATCH -N 1 
#SBATCH -n 1 --cpus-per-task=4
#SBATCH -t 30:00:00
#SBATCH --mem=128G

source activate OMGPT
# quite silly. but it works.
script_path=$(realpath "$0")
parent_dir=$(dirname $script_path)
pparent_dir=$(dirname $parent_dir)
ppparent_dir=$(dirname $pparent_dir)

export PYTHONPATH="${PYTHONPATH}:${ppparent_dir}/src"

python "${ppparent_dir}/src/data_generator.py"