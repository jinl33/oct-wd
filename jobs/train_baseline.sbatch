#!/bin/bash
#SBATCH --job-name=oct_base
#SBATCH --account=wilms0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
source ~/.oct_paths
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
OUT=$HOME/oct-runs/baseline_w2z1
RESUME=""
if [ -f "$OUT/latest.pt" ]; then RESUME="--resume"; fi
python scripts/train.py --out "$OUT" --design w2z1 --batch-size 4 --epochs "${EPOCHS:-50}" --workers 4 $RESUME
