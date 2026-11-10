#!/bin/bash
#SBATCH --job-name=oct_down
#SBATCH --account=wilms0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
source ~/.oct_paths
RUN=${RUN:-$HOME/oct-runs/w2z1_slo0}
for TASK in ${TASKS:-male righteye}; do
    python scripts/downstream.py --run "$RUN" --task "$TASK" --mode "${MODE:-probe}" $EXTRA
done
