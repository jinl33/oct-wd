# oct-wd

Wavelet diffusion pretraining for retinal OCT.

Start with `docs/HANDOVER.md`: machines, setup, the run plan, decisions and results so far.

## Setup on Armis2
    git clone https://github.com/jinl33/oct-wd ~/oct-wd
    echo 'export OCT_ROOT=/path/to/OCT-AI' > ~/.oct_paths

## Setup on a lab computer (RTX 5090 or RTX 6000, Ubuntu, no SLURM)
    python3 -m venv ~/oct-env && source ~/oct-env/bin/activate
    pip install -r requirements.txt
    echo 'export OCT_ROOT=/path/to/OCT-AI' > ~/.oct_paths
    python tests/test_pipeline.py

## Run from the repo folder
Armis2:

    module load python/3.10.4; source ~/oct-env/bin/activate; source ~/.oct_paths

Lab computer, in the background with a log in `~/oct-logs`:

    bash jobs/lab_run.sh NAME scripts/train.py --out ~/oct-runs/w2z1_slo0 --design w2z1 --epochs 200 --auto-resume

| Command | What it does | Reads OCT data? |
|---|---|---|
| `python tests/test_pipeline.py` | End-to-end test on invented data | No |
| `python scripts/check_cohort.py` | Cohort and slice-geometry summary, counts and ranges only | Yes |
| `sbatch jobs/probe_memory.sbatch` | GPU memory on random tensors | No |
| `sbatch jobs/qc_views.sbatch` | Staircase numbers, plus QC images in `~/oct-qc` | Yes |
| `sbatch jobs/stack_views.sbatch` | Slices through the stack for random scans, smooth slope vs sudden jumps, images in `~/oct-qc` | Yes |
| `sbatch jobs/dense_check.sbatch` | The same check on the 512- and 97-slice scans, also thinned to 25 slices, plus NIfTI files in `~/oct-qc/nifti` for ITK-SNAP | Yes |
| `sbatch jobs/slo_alignment.sbatch` | Whether OCT and SLO line up, numbers only | Yes |
| `sbatch jobs/train_pilot.sbatch` | Short training run | Yes |
| `sbatch jobs/train_baseline.sbatch` | Pretraining; resubmit to continue. `DESIGN`, `SLO` (0 or a size such as 256), `BATCH`, `ACCUM`, `EPOCHS`, `OUT` | Yes |
| `sbatch jobs/sample.sbatch` | Denoising numbers, plus generated and real images in the run folder; `RUN` picks the run | Yes |
| `sbatch jobs/downstream.sbatch` | Encoder-only features plus a classification or regression head, scored on the test split; `RUN`, `TASKS`, `MODE` (probe or finetune), `EXTRA` | Yes |
| `bash jobs/lab_run.sh NAME scripts/...` | Any of the scripts above on a lab computer, in the background | Depends on the script |
| `bash jobs/lab_run.sh NAME jobs/lab_evaluate.sh RUN [RUN ...]` | For each finished run, one after another: sampling, then probe and finetune heads, pretrained and random init. `TASKS`, `MODES`, `SAMPLE_ARGS`, `DOWN_ARGS` | Yes |

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
- Slices are stacked by number. The split is by patient: 70% train, 10% validation, 20% test. Pretraining never reads the test part.
- Downstream heads use the encoder and bottleneck only; the decoder is dropped to save memory.
- Optional SLO conditioner: a small 2D encoder turns the SLO into 64 tokens that the U-Net reads by cross-attention.
- No registration, within or between subjects, and no resampling to a physical grid.
- Checkpoints are written every epoch.
