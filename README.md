# oct-wd

Wavelet diffusion pretraining for retinal OCT.

## Setup
    git clone https://github.com/jinl33/oct-wd ~/oct-wd
    echo 'export OCT_ROOT=/path/to/OCT-AI' > ~/.oct_paths

## Run from the repo folder
    module load python/3.10.4; source ~/oct-env/bin/activate; source ~/.oct_paths

| Command | What it does | Reads OCT data? |
|---|---|---|
| `python tests/test_pipeline.py` | End-to-end test on invented data | No |
| `python scripts/check_cohort.py` | Cohort and slice-geometry summary, counts and ranges only | Yes |
| `sbatch jobs/probe_memory.sbatch` | GPU memory on random tensors | No |
| `sbatch jobs/qc_views.sbatch` | Staircase numbers, plus QC images in `~/oct-qc` | Yes |
| `sbatch jobs/slo_alignment.sbatch` | Whether OCT and SLO line up, numbers only | Yes |
| `sbatch jobs/train_pilot.sbatch` | Short training run | Yes |

QC images and checkpoints stay on Armis2. Printed logs hold counts and losses only, but read them before sharing.

## Wavelet designs
| Name | Levels within B-scan / across slices | Channels | Grid for 32 x 512 x 512 |
|---|---|---|---|
| `w2z1` | 2 / 1 | 32 | 16 x 128 x 128 |
| `w1z1` | 1 / 1 | 8 | 16 x 256 x 256 |
| `w2z2` | 2 / 2 (the brain layout) | 64 | 8 x 128 x 128, rejected by this U-Net |

## Differences from the brain pipeline
Same as brain: U-Net, 1000-step linear DDPM, sample prediction, MSE, AdamW at 1e-5, mixed precision.

Different on purpose:
- Percentiles use real pixels only. Padding happens afterwards, with -1.
- Anything larger than the target stops with an error instead of being center-cropped.
- A broken volume stops with an error instead of being replaced with noise.
- Slices are stacked by number. The split is by patient.
- Checkpoints are written every epoch.
