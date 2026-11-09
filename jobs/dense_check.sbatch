#!/bin/bash
#SBATCH --job-name=oct_dense
#SBATCH --account=wilms0
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
module load python/3.10.4
source ~/oct-env/bin/activate
source ~/.oct_paths
python scripts/stack_views.py --n-slices 512 --out ~/oct-qc/dense512
python scripts/stack_views.py --n-slices 97 --out ~/oct-qc/dense97
python scripts/export_nifti.py --n-slices 512 --n-views 2
python scripts/export_nifti.py --n-slices 25 --n-views 4
