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
| `bash jobs/lab_run.sh NAME jobs/lab_label_fraction.sh RUN [RUN ...]` | Heads trained on part of the training patients' labels (`--label-fraction`), several random subsets each, pretrained and random init; ends with a summary table. `TASKS` (default righteye), `MODES` (default probe), `FRACTIONS` (default 0.1 0.25 0.5 1), `SEEDS` (default 0 to 4), `DOWN_ARGS` | Yes |
| `bash jobs/lab_run.sh NAME jobs/lab_next_runs.sh RUNS_FOLDER` | One after another: memory probe and 200-epoch training for `w2z1_adapter` (SLO adapter, band scaling 0.5), then for `w2z1_sqrt24` (24 slices, slice axis halved twice). A failed step does not stop the next. `EPOCHS` | Yes |
| `bash jobs/lab_run.sh NAME jobs/lab_slo_runs.sh RUNS_FOLDER` | One after another: memory probe and 200-epoch training for `w2z1_xattn` (SLO with 256 tokens and extra cross-attention at levels 0 to 2), then for `w2z1_joint` (SLO as its own slice after the B-scans). Both on 24 slices with band scaling 0.5. `SLO_RUNS` (default `xattn joint`; `joint_step` trains `w2z1_joint_step`, the joint model with its own SLO noise step), `EPOCHS` | Yes |
| `bash jobs/lab_run.sh NAME scripts/scan_report.py --run RUN` | Each validation and test scan noised to steps 50, 200 and 500 and restored in one model call: PSNR, SSIM and the noisy input's PSNR, split by cohort columns (`--group`, default `male righteye`). Writes plots, a per-scan table without IDs, and views of the lowest-PSNR scans to `RUN/scan_report` | Yes |
| `python scripts/summarize_downstream.py --run RUN [RUN ...]` | One table of every head result in a run: mean, min and max over subsets | No, reads `results.json` only |

QC images and checkpoints stay on Armis2. Printed logs hold counts and losses only, but read them before sharing.

## New data, start to finish
On a lab computer, from the repo folder, with `OCT_ROOT` in `~/.oct_paths` pointing at the new image tree and cohort file:

1. `bash jobs/lab_run.sh cohort scripts/check_cohort.py`: counts, slice numbering and scan-line rows. Fix anything it reports first.
2. `bash jobs/lab_run.sh probe scripts/probe_memory.py --designs w2z1 --target 24 512 512 --z-down 0 0 1 1 --max-batch 8`: the largest batch that fits (add the SLO flags of step 3 when they are used).
3. Pretrain, 200 epochs (the current best setup; run plan in `docs/HANDOVER.md`):

       bash jobs/lab_run.sh pretrain scripts/train.py --out RUNS/w2z1_sqrt24 --design w2z1 --band-scale 0.5 --batch-size 4 --epochs 200 --auto-resume --keep-slices 24 --target 24 512 512 --z-down 0 0 1 1

   Add `--slo-size 256 --slo-tokens 16 --slo-xattn 0 1 2` for the SLO model, or `--slo-slice raster --target 32 512 512` for the joint model (add `--slo-own-step` to give the SLO its own noise step). Restart the same command after a stop.
4. `bash jobs/lab_run.sh eval jobs/lab_evaluate.sh RUNS/w2z1_sqrt24`: generated samples and their numbers, then probe and finetune heads for each task in `TASKS`, pretrained and from random init.
5. Heads for a new label: any cohort column, `--task COLUMN` with `--kind binary` (0/1) or `--kind regression` (for example age). `jobs/lab_label_fraction.sh` repeats them on 10 to 100% of the labels.

Distillation is not part of this pipeline yet: the heads use the pretrained encoder directly (see Open items in `docs/HANDOVER.md`).

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
- Optional SLO adapter (`train.py --slo-adapter`, with `--slo-size`): the same SLO encoder's feature maps from its second stage on are sampled along each B-scan's line on the SLO (from `metadata.csv`), averaged over each U-Net cell, repeated along depth, and added to the encoder at every level through 1×1 convolutions that start at zero. A pooled SLO vector is added to the time embedding. Padded slices get no SLO features. Cross-attention stays. A new adapter run starts out identical to the cross-attention model, and the heads read the adapted encoder.
- Optional 24 slices (`--keep-slices 24 --target 24 512 512 --z-down 0 0 1 1`): keeps the first 24 of the 25 B-scans and halves the slice axis only at the last two U-Net downsamplings, so 12 wavelet slices become 6 and then 3. Padding falls from 24% to 3% of the volume. `--z-down` takes one 0/1 per downsampling; the weights keep the same names.
- Optional band scaling (`train.py --band-scale`): each wavelet band is divided by its root mean square on the training scans before noise is added. `--band-scale 0.5` divides by the square root of it instead, a milder version that leaves the coarse image more weight. The divisors are saved in the checkpoint; `sample.py` multiplies them back, still clipping predictions to [-1, 1] in the original units, and `downstream.py` divides by them before the encoder.
- No registration, within or between subjects, and no resampling to a physical grid.
- Checkpoints are written every epoch.
