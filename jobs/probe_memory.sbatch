#!/bin/bash
#SBATCH --job-name=oct_probe
#SBATCH --account=wilms0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/probe_memory.py
