#!/bin/bash
#SBATCH --job-name=oct_pilot
#SBATCH --account=wilms0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
source ~/.oct_paths
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python scripts/train.py --out ~/oct-runs/pilot_w2z1 --design w2z1 --batch-size 2 --epochs 1 --max-steps 50 --workers 4
