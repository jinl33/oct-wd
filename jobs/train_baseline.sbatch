#!/bin/bash
#SBATCH --job-name=oct_train
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
DESIGN=${DESIGN:-w2z1}
SLO=${SLO:-0}
OUT=${OUT:-$HOME/oct-runs/${DESIGN}_slo${SLO}}
python scripts/train.py --out "$OUT" --design "$DESIGN" --slo-size "$SLO" --batch-size "${BATCH:-4}" --accum "${ACCUM:-1}" --epochs "${EPOCHS:-200}" --workers 4 --auto-resume
