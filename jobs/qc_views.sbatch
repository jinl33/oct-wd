#!/bin/bash
#SBATCH --job-name=oct_qc
#SBATCH --account=wilms0
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
source ~/.oct_paths
python scripts/qc_views.py
