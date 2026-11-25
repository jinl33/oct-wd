# OCT wavelet diffusion: handover

Status as of 2026-11-10. Written so that someone else (Hayagreev, or whoever picks this up) can run every step without asking.

## The data rule

The OCT scans are patient data under IRB, and use is audited.

- Scans, QC images, NIfTI files, generated images and checkpoints stay on Michigan systems: Armis2 and the HITS-cleared lab computers. Never Great Lakes, never a personal laptop, never an external AI service.
- Image viewers in VS Code (PNG preview, NiiVue) draw the image on the machine running the VS Code window. Open images only from the Michigan Medicine virtual machine.
- Logs print counts, losses and metrics only. Read a log before sharing it.
- GitHub holds code only. Armis2 and the lab computers only pull from it.

## Machines

| Machine | GPU | How jobs run | Notes |
|---|---|---|---|
| Armis2 | Titan V class, 12 GB | `sbatch jobs/*.sbatch`, account `wilms0`, partition `gpu` | `w1z1` does not fit here |
| Lab computer 10.156.33.27 | RTX 5090, 32 GB | `bash jobs/lab_run.sh` | Ubuntu, no SLURM. Reach it by SSH from the Michigan Medicine VM |
| Lab computer 10.156.33.24 | RTX 6000, 96 GB | `bash jobs/lab_run.sh` | Hayagreev's until the MIDL deadline; ask first |

The same commands run on both lab computers. `jobs/lab_run.sh NAME scripts/X.py ...` activates `~/oct-env`, reads `~/.oct_paths`, writes the GPU state and all output to `~/oct-logs/NAME-<time>.log`, and keeps running after you log out. Check `nvidia-smi` first so you don't share a GPU with someone else's run.

### Setup on a lab computer

    git clone https://github.com/jinl33/oct-wd ~/oct-wd
    cd ~/oct-wd
    python3 -m venv ~/oct-env && source ~/oct-env/bin/activate
    pip install -r requirements.txt
    echo 'export OCT_ROOT=/path/to/OCT-AI' > ~/.oct_paths
    python tests/test_pipeline.py

The test uses invented data only and must end with `ALL TESTS PASSED`. If the repository becomes private, copy it from Armis2 with `scp` run on the lab computer, rather than signing in to GitHub there. `OCT_ROOT` can point at the Turbo mount, or at a local copy on the lab computer, which is cleared to hold the data and may train faster.

### Data on a lab computer

The scans live on the sensitive Turbo volume `umms-wilms2`. Armis2 mounts it at `/nfs/turbo/umms-wilms2`. On the RTX 5090 (RDLU0080), `/nfs/turbo` is an automount whose map (`/etc/auto.turbo`) HITS manages with Ansible, and `umms-wilms2` is not in it. Don't edit that file; HITS overwrites it.

Mount the volume over SMB instead. This is Hayagreev's method, using the SMB name pattern in ARC's Turbo documentation. The file and folder modes keep the mounted data readable only by you on this shared machine. The mount asks for your UMICH password and lasts until reboot.

    sudo apt install cifs-utils
    sudo mkdir -p /mnt/umms-wilms2
    sudo mount -t cifs -o username=UNIQNAME,domain=UMROOT,sec=ntlmssp,uid=$(id -u),gid=$(id -g),file_mode=0600,dir_mode=0700,vers=3.0 //umms-wilms2-win.turbo.storage.umich.edu/umms-wilms2 /mnt/umms-wilms2
    echo 'export OCT_ROOT=/mnt/umms-wilms2/OCT_AI' > ~/.oct_paths

The home disk on the 5090 is nearly full. Put the environment, runs and logs on the large hard disk at `/data`, which is root-owned, so create your own private folder there:

    sudo mkdir -p /data/UNIQNAME && sudo chown $(id -u):$(id -g) /data/UNIQNAME && chmod 700 /data/UNIQNAME
    mkdir -p /data/UNIQNAME/oct-runs /data/UNIQNAME/oct-logs /data/UNIQNAME/tmp
    ln -sfn /data/UNIQNAME/oct-logs ~/oct-logs

A reboot drops the mount and stops running jobs. Mount again with the command above, then start each unfinished run with its original command; `--auto-resume` continues from the last epoch. Don't store your UMICH password on the machine to make the mount permanent: everyone with sudo there could read it. The permanent fix is for HITS to add `umms-wilms2` to the automount; ask through Hayagreev.

Keep the scans on the Turbo mount rather than copying them. That leaves one copy to account for, and Hayagreev found `/data` slow.

## What the pipeline does

1. **Select.** Match cohort rows to scan folders and keep scans with exactly 25 B-scans named b1 to b25 (531 of 538 in the pilot).
2. **Stack** B-scans by number. No registration within or between scans.
3. **Scale** each scan's 1st to 99th percentile to -1 … 1.
4. **Pad** with -1 to 32 × 512 × 512.
5. **Wavelet.** `w2z1` is 2 Haar levels inside each B-scan and 1 across slices, giving 32 channels on a 16 × 128 × 128 grid. `w1z1` is 1 level on every axis, giving 8 channels on 16 × 256 × 256.
6. **Diffusion U-Net**, the brain recipe: 1000-step linear DDPM, predicts the clean coefficients, MSE, AdamW 1e-5, mixed precision.
7. **Optional SLO conditioner** (`--slo-size 256`). A small 2D CNN turns the SLO into 64 tokens. The U-Net reads them by cross-attention in its two deepest levels. 20% of training steps get learned "no SLO" tokens instead, so the model also works without an SLO. Validation reports the loss with the scan's own SLO, with another scan's SLO, and with no SLO. If own and other are equal, the model ignores the SLO.
   - `--slo-tokens 16` pools the SLO to 256 tokens instead of 64, and `--slo-dim` sets their size.
   - `--slo-xattn 0 1 2` adds a cross-attention to the SLO tokens after those U-Net encoder levels, on top of the built-in one at levels 3 and 4. Each starts at zero, so a new run starts as the plain SLO model.
   - `--slo-adapter` adds SLO feature maps sampled along each scan line (run plan step 8).
8. **Optional joint OCT + SLO volume** (`--slo-slice raster`, no `--slo-size`). The SLO is resampled onto the scan raster: 512 rows along the slow axis, 512 columns along each B-scan, so column j of the SLO lies under A-scan j. It is appended twice after the B-scans, so the across-slice wavelet keeps it in a wavelet slice of its own. The model then learns OCT and SLO together (the joint distribution) instead of OCT given SLO. `--slo-slice full` uses the whole SLO instead of the scan area. It needs an even number of B-scans (`--keep-slices 24`) and two free slices in `--target` (32 × 512 × 512). Validation also reports the loss on the OCT part with the scan's own SLO slice and with another scan's.
   - `--slo-own-step` gives the SLO slice its own noise step (UniDiffuser-style). Per training scan the SLO is left clean (`--slo-clean`, 25%), gets the OCT's step (`--slo-same`, 25%), or gets an independent step (the rest). The noise level of every voxel goes in as one extra input channel, and the SLO's step also enters the time embedding through a second embedding that starts at zero. A clean SLO is left out of the loss. The model then learns the joint distribution and also the OCT given a clean SLO. Validation adds the OCT loss given a clean SLO (own and another scan's) and with the SLO slice all noise. `sample.py` adds the same three in the denoising table and generates OCTs from validation scans' real SLOs (`given_*.png`, `--n-given`, default 8).
9. **Downstream.** `scripts/downstream.py` keeps the input layer, encoder and bottleneck, drops the decoder, and pools each level into a 560-number feature vector. Then either:
   - `probe`: frozen encoder plus a linear head. Weight decay is chosen on validation.
   - `finetune`: encoder plus head trained together. The epoch is chosen on validation.

   Both score the test split once, with a bootstrap 95% CI. `--random-init` runs the same thing without pretrained weights, as the control. Features from both modalities:
   - SLO models: `--slo-features` adds the SLO encoder's pooled features (160 numbers).
   - Joint models: the OCT part and the SLO slice are pooled separately and both are used (1,120 numbers). `--joint-pool oct` or `slo` keeps one.

   Any numeric cohort column works with `--kind regression`, so retinal age needs only the column once the data pull brings it (`--task age --kind regression`; the column name is not yet known).
The split is by patient: 70% train, 10% validation, 20% test (seed 42). It is saved in each run's `split.json`. Pretraining never reads the test part, and a run refuses to continue if the split changes.

## Run plan agreed on 2026-11-09

Run these on the RTX 5090, one at a time unless `nvidia-smi` shows room for two.

1. Check memory for each design:

       bash jobs/lab_run.sh probe scripts/probe_memory.py --designs w2z1 w1z1
       bash jobs/lab_run.sh probe_slo scripts/probe_memory.py --designs w2z1 w1z1 --slo-size 256

2. Pretrain the three variants for 200 epochs. For `w1z1`, set `--batch-size` to the largest batch the probe allowed, and `--accum` so that batch × accum = 4.

       bash jobs/lab_run.sh w2z1 scripts/train.py --out ~/oct-runs/w2z1_slo0 --design w2z1 --batch-size 4 --epochs 200 --auto-resume
       bash jobs/lab_run.sh w1z1 scripts/train.py --out ~/oct-runs/w1z1_slo0 --design w1z1 --batch-size 1 --accum 4 --epochs 200 --auto-resume
       bash jobs/lab_run.sh w2z1_slo scripts/train.py --out ~/oct-runs/w2z1_slo256 --design w2z1 --slo-size 256 --batch-size 4 --epochs 200 --auto-resume

   If a run stops, start the same command again; `--auto-resume` continues from the last epoch.

3. For finished runs, sample and test the heads. One command does it all in order: for each run, sampling, then for `male` and `righteye` a probe and a finetune head, each pretrained and with random init (9 steps per run). The log ends with the number of failed steps.

       bash jobs/lab_run.sh eval jobs/lab_evaluate.sh ~/oct-runs/w2z1_slo0 ~/oct-runs/w2z1_slo256

On the RTX 5090, write runs to `/data/UNIQNAME/oct-runs` rather than `~/oct-runs`, because the home disk is nearly full.

4. Label fractions: what pretraining buys with few labels. Heads are trained on 10%, 25%, 50% and 100% of the training patients, five random subsets per fraction, pretrained and random init. Validation and test stay whole. Use the eye task on runs without the SLO, because the SLO shows the eye directly. The default is probes only, about an hour per run (estimated). Finetuning (`MODES="probe finetune"`) scales the epochs up as the labels shrink, so every fraction gets about the same number of training steps; each finetune takes about as long as one on all labels.

       bash jobs/lab_run.sh frac jobs/lab_label_fraction.sh ~/oct-runs/w2z1_slo0 ~/oct-runs/w1z1_slo0

   The log ends with a table per run. Print it again at any time with `python scripts/summarize_downstream.py --run RUN`.

5. Texture fix (decided 2026-11-11): retrain `w2z1` with `--band-scale`. Before training, the run measures each wavelet band's root mean square on the training scans and divides the band by it, so every band reaches the noise at the same size. The sizes are stored in the checkpoint; sampling multiplies them back and the heads divide by them too, so nothing else changes. Then evaluate it the same way as `w2z1`:

       bash jobs/lab_run.sh w2z1_band scripts/train.py --out /data/UNIQNAME/oct-runs/w2z1_band --design w2z1 --batch-size 4 --epochs 200 --auto-resume --band-scale
       bash jobs/lab_run.sh eval_band jobs/lab_evaluate.sh /data/UNIQNAME/oct-runs/w2z1_band
       bash jobs/lab_run.sh frac_band jobs/lab_label_fraction.sh /data/UNIQNAME/oct-runs/w2z1_band

   Compare with `w2z1` on the band table, the smoothness numbers, the eye probe and the label-fraction table. Do not compare its validation loss or denoising ratio with `w2z1`: they are in band-scaled units. Its random-init control also sees band-scaled input, so compare the gap between pretrained and random init, not the random-init numbers across runs.

   Done 2026-11-11. At full strength the scaling overcorrects: too much fine detail, a weaker coarse image and a worse eye probe (Results, "Band scaling").

6. Milder band scaling (decided 2026-11-11): `--band-scale 0.5` divides each band by the square root of its root mean square. On the training-scan sizes, the fine bands enter at 0.13 to 0.24 instead of 0.018 to 0.06 (`w2z1`) or 1 (`w2z1_band`), and the coarse band keeps about half of the power instead of 95% or 3%. Run and evaluate it one job at a time:

       bash jobs/lab_run.sh w2z1_sqrt scripts/train.py --out /data/UNIQNAME/oct-runs/w2z1_sqrt --design w2z1 --batch-size 4 --epochs 200 --auto-resume --band-scale 0.5
       bash jobs/lab_run.sh eval_sqrt jobs/lab_evaluate.sh /data/UNIQNAME/oct-runs/w2z1_sqrt
       bash jobs/lab_run.sh frac_sqrt jobs/lab_label_fraction.sh /data/UNIQNAME/oct-runs/w2z1_sqrt

   It works if the coarse row of the band table stays near 1, the fine rows land between `w2z1` and `w2z1_band` (ideally near 1), black share and median brightness come back towards real, and the eye probe at 37 to 93 labels is close to `w2z1`'s. Its loss is in its own units, so it cannot be compared with either run.

   Done 2026-11-12. It meets these checks on the numbers (Results, "Milder band scaling"); the images are still to be viewed.

   The same scaling on `w1z1` (`w1z1_sqrt`, started 2026-11-13, batch 2 × accum 2 as for `w1z1`), run one job at a time:

       bash jobs/lab_run.sh w1z1_sqrt scripts/train.py --out /data/oct_data/oct-runs/w1z1_sqrt --design w1z1 --batch-size 2 --accum 2 --epochs 200 --auto-resume --band-scale 0.5
       bash jobs/lab_run.sh eval_w1sqrt jobs/lab_evaluate.sh /data/oct_data/oct-runs/w1z1_sqrt
       bash jobs/lab_run.sh frac_w1sqrt jobs/lab_label_fraction.sh /data/oct_data/oct-runs/w1z1_sqrt

   Done 2026-11-18. Texture comes back, but the eye probe falls below `w1z1` at every label count and below `w2z1_sqrt` with few labels (Results, "Milder band scaling on `w1z1`").

7. Scan report (asked for 2026-11-17, for Lubomir): cohort counts and reconstruction quality per scan, as plots. Each validation and test scan is noised to steps 50, 200 and 500 and restored in one model call. PSNR and SSIM compare the restored scan with the original; SSIM is computed per B-scan and averaged. The noisy input's PSNR shows how hard each step is. The plots group scans by `male` and `righteye` for now; once the data pull adds age or a diagnosis, pass that column with `--group`. Plots need matplotlib (`pip install matplotlib` in `oct-env`); without it, the numbers and the table are still written.

       bash jobs/lab_run.sh report_w2sqrt scripts/scan_report.py --run /data/oct_data/oct-runs/w2z1_sqrt

   Output goes to `RUN/scan_report`: `cohort.png`, `reconstruction.png`, one `reconstruction_COLUMN.png` per group, `scans.csv` (one row per scan, no IDs) and `worst_*.png` (noisy input, restored and original middle B-scan of the lowest-PSNR scans). View the images on the VM. Noise is added in each run's own wavelet units, so compare groups within a run, not runs with each other. PSNR is the error on a log scale, 10·log10(4 / MSE) on the -1 to 1 scale. Generated scans have no real counterpart, so PSNR does not apply to them.

   Done 2026-11-18 for `w2z1_sqrt` (Results, "Scan report").

8. SLO adapter (proposed to Matthias 2026-11-17, built 2026-11-18): cross-attention reaches only the two deepest levels, and own SLO beat another scan's by about 1%. `--slo-adapter` also feeds the SLO encoder's feature maps from its second stage on into the U-Net encoder at every level. Each map is sampled along every B-scan's line on the SLO (from `metadata.csv`), averaged over each U-Net cell, repeated along depth and added through a 1×1 convolution that starts at zero. From the second stage on, each feature covers more than the gap between two scan lines (receptive field 19, 43 and 91 px at 256, against about 7 px between lines). A pooled SLO vector goes into the time embedding, so SLO content outside the scan area also counts. Padded slices get no SLO features, and dropped-SLO steps (`--slo-drop`) get none either. A new run starts out identical to the cross-attention model.

   First check that every scan has one metadata line per B-scan. `check_cohort.py` prints `row count differs from B-scans: N`, which should be 0. A scan with a mismatch stops the run with `volume index N failed: ValueError`. Then check memory, and train on `w2z1` with square-root band scaling:

       bash jobs/lab_run.sh cohort scripts/check_cohort.py
       bash jobs/lab_run.sh probe_adapter scripts/probe_memory.py --designs w2z1 --slo-size 256 --slo-adapter --max-batch 8
       bash jobs/lab_run.sh w2z1_adapter scripts/train.py --out /data/oct_data/oct-runs/w2z1_adapter --design w2z1 --slo-size 256 --slo-adapter --band-scale 0.5 --batch-size 4 --epochs 200 --auto-resume

   The test is in the training log: `val mse X (own SLO)  other SLO Y  no SLO Z`. Own SLO should now be clearly below another scan's (the cross-attention model: about 1%). Then run `lab_evaluate.sh` on it for the denoising ratios with own, other and no SLO. The eye task does not test the OCT features of an SLO model, because the SLO shows the optic disc.

9. 24 slices (Matthias asked 2026-11-17 whether dropping one slice helps; agreed to try it after the SLO change): `--keep-slices 24` keeps B-scans 1 to 24, and `--z-down 0 0 1 1` halves the slice axis only at the last two of the four U-Net downsamplings. With `--target 24 512 512` that gives 12 wavelet slices, then 6 and 3 at the two deepest levels, and padding falls from 24% to 3% of the volume. Halving late keeps the slice axis, already the coarsest (0.25 mm between B-scans), whole while the in-plane cells are still small. By my count it needs about 10% less activation memory than the current 32-slice setup; the probe measures it.

       bash jobs/lab_run.sh probe_24 scripts/probe_memory.py --designs w2z1 --target 24 512 512 --z-down 0 0 1 1 --max-batch 8
       bash jobs/lab_run.sh w2z1_sqrt24 scripts/train.py --out /data/oct_data/oct-runs/w2z1_sqrt24 --design w2z1 --band-scale 0.5 --batch-size 4 --epochs 200 --auto-resume --keep-slices 24 --target 24 512 512 --z-down 0 0 1 1

   Steps 8 and 9 run back to back with `bash jobs/lab_run.sh next_runs jobs/lab_next_runs.sh /data/oct_data/oct-runs` (started 2026-11-18). A failed step does not stop the next; the log ends with `next runs finished, failed steps: N`.

   Compare with `w2z1_sqrt` on the band table, the intensity numbers, the eye probe at label fractions and the scan report, not on loss: padding is easy to predict, so less of it raises the loss per voxel. The two options combine (`--slo-adapter` with the 24-slice flags).

10. Second pretraining seed (planned 2026-11-19): `w2z1_sqrt24` beat `w2z1_sqrt` on the eye probe by 0.06 to 0.08, and there is one pretraining run per design. Until 2026-11-19 `--seed` also set the train, validation and test split, so a new seed meant a new test set. `--split-seed` now sets the split on its own (it defaults to `--seed`, so earlier runs are unchanged); pass `--split-seed 42` to keep the split of every earlier run. Pull the code first. The two runs fit side by side (about 19 GB).

        bash jobs/lab_run.sh sqrt24_s1 scripts/train.py --out /data/oct_data/oct-runs/w2z1_sqrt24_s1 --design w2z1 --band-scale 0.5 --batch-size 4 --epochs 200 --auto-resume --keep-slices 24 --target 24 512 512 --z-down 0 0 1 1 --seed 1 --split-seed 42
        bash jobs/lab_run.sh sqrt_s1 scripts/train.py --out /data/oct_data/oct-runs/w2z1_sqrt_s1 --design w2z1 --band-scale 0.5 --batch-size 4 --epochs 200 --auto-resume --seed 1 --split-seed 42

   Once both have started, check that their split matches the earlier runs; this prints only `same split` or `DIFFERENT split`:

        python -c "import json,sys; s=[json.load(open(p)) for p in sys.argv[1:]]; print('same split' if all(x==s[0] for x in s) else 'DIFFERENT split')" /data/oct_data/oct-runs/w2z1_sqrt/split.json /data/oct_data/oct-runs/w2z1_sqrt24/split.json /data/oct_data/oct-runs/w2z1_sqrt24_s1/split.json /data/oct_data/oct-runs/w2z1_sqrt_s1/split.json

   Then run `lab_label_fraction.sh` on both. If the gap between the two designs holds at both seeds, it is not the seed.

11. SLO through cross-attention at more levels, with more tokens (Matthias, 2026-11-23: "stick with the cross-attention first and make the feature vector larger, use it at different levels"). The cross-attention model reached only the two deepest levels with 64 tokens, and own SLO beat another scan's by about 1%; the adapter did not help at any noise level. This run gives the SLO 256 tokens and an extra cross-attention after encoder levels 0, 1 and 2, on the `w2z1_sqrt24` setup (the best `w2z1` run so far).
12. SLO as its own slice, a joint model of OCT and SLO (Matthias's next step if step 11 still shows no signal). Pipeline step 8 describes the layout. It needs no SLO encoder, and its features cover both modalities by construction.

    Steps 11 and 12 run back to back, with a memory probe before each (pull the code first):

        bash jobs/lab_run.sh slo_runs jobs/lab_slo_runs.sh /data/oct_data/oct-runs

    `SLO_RUNS=xattn` or `SLO_RUNS=joint` runs one of them. The log ends with `SLO runs finished, failed steps: N`. Then evaluate with the SLO features going to the heads, and sample:

        DOWN_ARGS=--slo-features bash jobs/lab_run.sh eval_xattn jobs/lab_evaluate.sh /data/oct_data/oct-runs/w2z1_xattn
        bash jobs/lab_run.sh eval_joint jobs/lab_evaluate.sh /data/oct_data/oct-runs/w2z1_joint

    What to read:
    - `w2z1_xattn`, training log: `val mse X (own SLO)  other SLO Y  no SLO Z`. The SLO helps if X is clearly below Y and Z (the cross-attention model: about 1%; the adapter: 0.25%). In `sample.py`, the denoising ratios at steps 800 and 950 with own vs another scan's SLO.
    - `w2z1_joint`, training log: `OCT with own SLO slice A  with another scan's B`. B clearly above A means the model uses the SLO to denoise the OCT. In `sample.py`, the line `OCT en-face view against the SLO`: real scans score about 0.2 with their own SLO and about 0 with another scan's (alignment check, 2026-11-06). Generated pairs that score near the real value mean the model learned how the two modalities fit together. The images `sample_*_slo.png` are generated SLOs.
    - Both: the band table and intensity numbers as for `w2z1_sqrt24`, the eye probe (the SLO shows the optic disc, so for SLO models it does not test the OCT features), and the generated scans on the VM. Matthias asked to see the generated OCTs once the SLO is in.

    Compare with `w2z1_sqrt24` on the heads, the band table and the images, not on loss: the SLO slice and the different padding change what the loss averages over.

13. Joint model with an own noise step for the SLO (2026-11-24, after steps 11 and 12 showed no SLO signal in training). Tests the inferred reason that the joint model ignores the SLO because both parts always share one noise step. Same setup as `w2z1_joint` plus `--slo-own-step` (pipeline step 8). Pull the code first:

        SLO_RUNS=joint_step bash jobs/lab_run.sh slo_step jobs/lab_slo_runs.sh /data/oct_data/oct-runs

    Then evaluate (no `--slo-features`; the joint model has no SLO encoder):

        bash jobs/lab_run.sh eval_step jobs/lab_evaluate.sh /data/oct_data/oct-runs/w2z1_joint_step

    What to read:
    - Training log, `OCT given a clean SLO: own A  another scan's B  SLO all noise C`. The SLO is used if A is clearly below B and C. The shared-step numbers on the same line should stay close to `w2z1_joint`'s.
    - `sample.py`, denoising table: own SLO clean against another scan's and against all noise, at steps 800 and 950, where the OCT is mostly noise.
    - `sample.py`, `OCT generated from a given real SLO`: the `from SLO` line, own against another scan's. Unlike the generated pairs of `w2z1_joint`, the SLOs here are real, so a pattern shared by all generated scans cannot raise own and other together; the `similarity between different scans` lines show whether generated scans share one. Real scans score about 0.2 against their own SLO and about 0 against another scan's. A value near the real one means the model draws the OCT to match the SLO it was given. View `given_*.png` next to `real_*.png` and `real_*_slo.png` on the VM.

`male` and `righteye` are the only labels the pilot has. They check that the heads run end to end. They are not research results. Matthias (2026-11-23): retinal age would be the next target, but the pilot has no ages, so sex stays the placeholder task until the data pull; the pilot is for setting up the pipeline, and the model is retrained once the 5,000 to 10,000 scans from the pull arrive.

### Reading the logs

| Line | Meaning |
|---|---|
| `epoch N: train mse … val mse …` | Diffusion loss. Lower is better. Compare runs of the same design only. The wavelet scaling differs between `w2z1` and `w1z1`, so their losses are on different scales |
| `val mse X (own SLO)  other SLO Y  no SLO Z` | The SLO test. X clearly below Y means the model uses the SLO |
| `val mse X (OCT A, SLO B)  OCT with own SLO slice C  with another scan's D` | Joint model. A and B are the loss on each part. D clearly above C means the model uses the SLO to denoise the OCT |
| Denoising table in `sample.py` | Ratio of model error to the error of guessing the average scan. Below 1 is better than the average. Compare within a design only: `wavelet.forward` scales coefficient power to 1/32 of the pixels' in `w2z1` and 1/8 in `w1z1`, while the added noise has the same size, so the same noise step hides more of the scan in `w2z1`. Compare designs on the heads and the generated-scan numbers |
| `test auroc …` in `downstream.py` | The head on held-out patients. Compare pretrained against `--random-init` |
| Band table in `sample.py` | Size of the wavelet coefficients, grouped by how many axes are detail (high-pass), for real and generated scans, and the noise added near the end of sampling. A generated / real ratio well below 1 for the detail groups means the model leaves out fine detail |

## Decisions in force

| Date | Decision | From |
|---|---|---|
| 2026-11-04 | OCT runs only on Armis2 and the lab computers; adapt the brain pipeline | Matthias |
| 2026-11-06 | No registration for now; check the stacks by slicing through them | Matthias, Hayagreev |
| 2026-11-06 | Use the 25-slice scans; set the 7 longer scans aside | Hayagreev |
| 2026-11-09 | The two 512-slice scans look smooth across slices, so the scanner corrects motion and registration stays out | Dense check, Matthias's rule |
| 2026-11-09 | Split 70 / 10 / 20 into train, validation and test | Lubomir |
| 2026-11-09 | Try one wavelet level (8 channels) | Matthias |
| 2026-11-09 | Prepare global classification and regression heads; drop the decoder at downstream time | Matthias |
| 2026-11-09 | Add the SLO as a conditioner through an encoder | Matthias |
| 2026-11-09 | Make runs work the same on both lab computers | Lubomir |
| 2026-11-09 | Use the pilot for setup and pretraining only; disease sets come separately from Peter | Matthias, Lubomir |
| 2026-11-23 | The pipeline and its spacing choices stand; no further changes needed there | Matthias |
| 2026-11-23 | The model must be multimodal: features from both OCT and SLO for the downstream tasks | Matthias |
| 2026-11-23 | SLO: first cross-attention with a larger SLO representation at several levels; if that still shows no signal, the SLO as an extra slice after the wavelet step (joint model) | Matthias |
| 2026-11-23 | Downstream target: retinal age when ages exist; until then sex prediction as the placeholder task, only to have the fine-tuning pipeline ready | Matthias |
| 2026-11-23 | Organize the scripts so Hayagreev can take over: pretraining, distillation and fine-tuning on a new data folder | Matthias |
| 2026-11-23 | Next meeting with Peter on 2026-12-07; Hayagreev to have the MRN lists ready by then | Matthias |

## Results so far (pilot, aggregates only)

**Cohort.** 538 scans: 531 with 25 B-scans, 5 with 97, 2 with 512. B-scans are 512 wide × 496 deep, with slices about 0.25 mm apart (0.20 to 0.33 mm). The SLO is 768 × 768, 30°; the scan area is a square 20° raster. Two spacing columns in the cohort CSV are swapped: `bscanPixelspacingWidth` holds the axial 3.87 µm.

**QC.**
- **OCT matches its SLO.** Median correlation 0.197 with its own SLO vs 0.000 with another scan's; its own SLO is the best match in 527 of 531 scans.
- **Slice jumps.** 431 of 531 scans have no sudden jump between neighbouring slices of 5 px (about 19 µm) or more; 25 have one of 20 px or more.
- **Dense scans.** The two 512-slice scans show a smooth outer retina across slices and sharp vessels from above.

**First baseline** (Armis2, `w2z1`, batch 4, 50 epochs, older 80/20 split with no test part):
- Validation loss 0.00708 at epoch 1, 0.00197 at epoch 50; training loss 0.00172.
- Samples show a dark area above the retina, a bright band at its base and layered stripes, but are very blurry.
- The run was continued to 200 epochs on 2026-11-09 under the same older split. Keep it for reference only. It cannot be used for downstream tests, because its training scans include patients that are in the new test split.

**Corrected split on the RTX 5090** (2026-11-10). Split 70/10/20, so 372 training, 53 validation and 106 test scans. 200 epochs, effective batch 4, one seed. Numbers typed back from the logs.

Validation diffusion loss. Compare rows of the same design only.

| Run | Epoch 50 | 100 | 150 | 200 |
|---|---|---|---|---|
| `w2z1` | 0.001562 | 0.001479 | 0.001467 | 0.001399 |
| `w2z1` + SLO, own SLO | 0.001569 | 0.001512 | 0.001431 | 0.001401 |
| `w2z1` + SLO, another scan's SLO | 0.001572 | 0.001555 | 0.001465 | 0.001417 |
| `w2z1` + SLO, no SLO | 0.001573 | 0.001520 | 0.001430 | 0.001412 |
| `w1z1` (batch 2 × accum 2, peak 20.1 GB) | 0.005062 | 0.004604 | 0.004586 | 0.004486 |

`w2z1` with `--band-scale` (2026-11-11, `w2z1_band`, batch 4, 43 s per epoch, peak 9.30 GiB, as for `w2z1`):
- **Band sizes measured on the 372 training scans:** root mean square 0.8245, 0.0592, 0.0291, 0.0228, 0.0191 and 0.0178 for 0 to 5 detail axes. The smallest single band is 0.0118. The four validation scans in the band table are smoother: 0.0103 to 0.0186 for 2 to 5 axes.
- **Validation loss, in band-scaled units:** 0.788 at epoch 1, then 0.573, 0.544, 0.532 and 0.526 at epochs 50, 100, 150 and 200. Predicting zero would give about 1. This cannot be compared with the table above.

`w2z1` with `--band-scale 0.5` (2026-11-12, `w2z1_sqrt`, batch 4, 42 to 44 s per epoch, peak 9.30 GiB):
- **Band sizes:** the same as `w2z1_band` (same split), now divided by their square roots.
- **Validation loss, in its own units:** 0.0318 at epoch 1, then 0.0201, 0.0197, 0.0195 and 0.0193 at epochs 50, 100, 150 and 200. Predicting zero would give at most about 0.055. Not comparable with `w2z1` or `w2z1_band`.

`w1z1` with `--band-scale 0.5` (2026-11-13, `w1z1_sqrt`, batch 2 × accum 2, 125 s per epoch, peak 20.08 GiB, as for `w1z1`):
- **Band sizes measured on the 372 training scans:** root mean square 0.8276, 0.0839, 0.0451 and 0.0297 for 0 to 3 detail axes, divided by their square roots.
- **Validation loss, in its own units:** 0.0788 at epoch 1, then 0.0333, 0.0324, 0.0320 and 0.0319 at epochs 50, 100, 150 and 200 (lowest 0.0318 at epoch 196). Predicting zero would give about 0.16. Not comparable with any other run.

`w2z1` with the SLO adapter and `--band-scale 0.5` (2026-11-18 to 19, `w2z1_adapter`, batch 4, 45 to 48 s per epoch, peak 9.63 GiB, 23,971,680 parameters; memory probe 9.25 GiB at batch 4, 17.60 GiB at batch 8):
- **Band sizes:** identical to `w2z1_band` and `w2z1_sqrt`, so its loss is in the same units as `w2z1_sqrt`.
- **Validation loss, own SLO:** 0.0315 at epoch 1, then 0.0201, 0.0198, 0.0194 and 0.0192 at epochs 50, 100, 150 and 200 (lowest 0.019189 at epoch 197). `w2z1_sqrt` without any SLO: 0.0201, 0.0197, 0.0195, 0.0193.
- **SLO test:** at epoch 200, another scan's SLO 0.019316 (+0.44%) and no SLO 0.019261 (+0.16%) against own 0.019231. Over epochs 151 to 200 another scan's SLO is worse than own in all 50 epochs, by about 0.25% on average; no SLO is worse in 43 of 50, by about 0.15%. The gap was near zero in the first 10 epochs.
- **What the training log says:** the model uses the SLO consistently but it barely lowers the loss, and with no SLO it matches `w2z1_sqrt`. The cross-attention model's 1% was in unscaled units, where the coarse bands that an SLO can inform weigh more, so the two percentages are not like for like (inferred). The loss averages over all noise levels; an SLO can only help where the OCT is mostly noise, so the per-step denoising numbers from `sample.py` (own, other, no SLO) are the better test.
- **Denoising on 20 validation scans** (2026-11-19; ratio to the average scan, lower is better), own / another scan's / no SLO:

  | Noise step | Own SLO | Another scan's SLO | No SLO |
  |---|---|---|---|
  | 50 | 0.413 | 0.413 | 0.413 |
  | 200 | 0.761 | 0.760 | 0.758 |
  | 500 | 0.895 | 0.896 | 0.894 |
  | 800 | 0.973 | 0.977 | 0.972 |
  | 950 | 1.056 | 1.060 | 1.051 |

  No SLO is as good as the scan's own SLO at every step, including 800 and 950, where the OCT is almost all noise and the SLO should help most. Another scan's SLO costs about 0.4% at 800 and 950. The ratios are in the same units as `w2z1_sqrt`'s, but only its step-950 ratio (1.04) is recorded here.
- **Generated scans** (4, each from the SLO of a validation scan): less fine texture than `w2z1_sqrt`. Generated / real is 1.080 for the coarse band, 1.332 for one detail axis, then 0.751, 0.537, 0.482 and 0.440 (band table above; `w2z1_sqrt` 1.25 to 1.98). Intensity on the same 4 real scans:

  | | Real | `w2z1_sqrt` | `w2z1_adapter` |
  |---|---|---|---|
  | Mean | -0.665 | -0.579 | -0.669 |
  | Median | -0.908 | -0.827 | -0.994 |
  | Black share | 0.346 | 0.176 | 0.568 |
  | Within-slice change | 0.084 | 0.110 | 0.027 |
  | Slice-to-slice change | 0.123 | 0.172 | 0.069 |

  Smoother and with more pure background than real scans. The cause is not known, and 4 scans is few; the 8 images are for viewing on the VM.
- **Heads** (heads table above): the eye probe scores 0.948, the highest of any run. The adapter feeds the SLO, which shows the optic disc, into every encoder level, so this does not test the OCT features. Sex is at chance. In sex finetuning the training loss falls to 0.0001 from pretrained weights and from random init while validation stays between 0.43 and 0.64: the SLO is memorised as a per-person fingerprint, as with the cross-attention model, now completely (`w2z1` about 0.68).
- **What this says.** The adapter does not help the denoiser at any noise level. It costs 4.2 million parameters and about 10% more time per epoch, and its generated scans have less texture. On this evidence it should not go forward into pretraining; whether the SLO enters later as a separate input to a diagnosis head is a decision for the lab.

`w2z1` with 24 slices and `--band-scale 0.5` (2026-11-19, `w2z1_sqrt24`, `--keep-slices 24 --target 24 512 512 --z-down 0 0 1 1`, batch 4, 36 to 42 s per epoch, peak 9.38 GiB, 19,731,792 parameters as for `w2z1`; memory probe 9.10 GiB at batch 4, 17.47 GiB at batch 8):
- **Memory did not fall:** peak 9.38 GiB against 9.30 for `w2z1_sqrt`, not the 10% saving I estimated. Keeping 12 slices to the third level makes the two attention levels hold 3 times as many positions (6 × 16 × 16 and 3 × 8 × 8), and attention cost grows with their square (inferred). Epochs are about 13% faster (37 against 42 to 44 s).
- **Band sizes:** 0.7728, 0.0548, 0.0329, 0.0259, 0.0217, 0.0202 for 0 to 5 detail axes. With less -1 padding the coarse band is smaller and the in-plane detail bands larger than with 32 slices.
- **Validation loss, in its own units:** 0.0325 at epoch 1, then 0.0225, 0.0219, 0.0215 and 0.0213 at epochs 50, 100, 150 and 200 (lowest 0.021251 at epoch 196). Not comparable with `w2z1_sqrt`: different band sizes and less padding.
- **Eye probe at label fractions** (2026-11-19, same five subsets as the other runs), test AUROC mean (min to max):

  | Training scans with labels | `w2z1_sqrt24` pretrained | `w2z1_sqrt24` random init | `w2z1_sqrt` pretrained | `w1z1` pretrained |
  |---|---|---|---|---|
  | 37 (10%) | 0.803 (0.703–0.885) | 0.491 (0.392–0.559) | 0.745 (0.665–0.843) | 0.801 (0.740–0.830) |
  | 93 (25%) | 0.869 (0.848–0.922) | 0.501 (0.410–0.589) | 0.805 (0.773–0.824) | 0.849 (0.794–0.878) |
  | 186 (50%) | 0.880 (0.857–0.901) | 0.500 (0.477–0.542) | 0.816 (0.804–0.843) | 0.868 (0.810–0.901) |
  | 372 (100%) | 0.897 (CI 0.826–0.952) | 0.465 | 0.821 (CI 0.728–0.900) | 0.891 |

  - **Above `w2z1_sqrt` at every label count,** by 0.06 to 0.08. With 93 and 186 labels its lowest subset is above `w2z1_sqrt`'s highest; with 37 the ranges overlap, and with all labels the CIs overlap.
  - **Level with `w1z1`,** the best probe so far, at 9.4 GB and 37 s per epoch against 20 GB and 125 s.
  - **Cause not known.** Three things changed at once: B-scan 25 is dropped, padding falls from 24% to 3%, and the slice axis stays whole for two more levels. The probe averages each level's features over the whole volume, padding included, but padding is the same for every scan, and the standardised linear probe should absorb a constant shift (inferred). There is one pretraining run per design, so a gap of this size could partly be the pretraining seed (not measured).
  - The random-init probe is at chance and sees band-scaled input, so it is not compared across runs.
- **Scan report** (2026-11-19, all 159 validation and test scans, 399 s). PSNR is computed on the scan's own slices with padding removed: 24 B-scans here, 25 for `w2z1_sqrt`. The noisy input PSNR is within 0.07 dB of `w2z1_sqrt` at every step, so the steps are about equally hard in both runs and the restored PSNR can be compared roughly. Median (25th to 75th percentile):

  | Noise step | PSNR, dB | Noisy input PSNR, dB | SSIM | Gain over noisy input, `w2z1_sqrt24` / `w2z1_sqrt` | SSIM, `w2z1_sqrt` |
  |---|---|---|---|---|---|
  | 50 | 24.28 (22.13–25.71) | 20.23 (20.07–20.38) | 0.632 (0.609–0.662) | 4.05 / 4.41 dB | 0.653 |
  | 200 | 21.81 (19.27–23.47) | 9.94 (9.93–9.95) | 0.402 (0.339–0.493) | 11.87 / 11.84 dB | 0.402 |
  | 500 | 20.31 (18.21–21.74) | 5.86 (5.74–6.02) | 0.335 (0.237–0.446) | 14.45 / 14.30 dB | 0.331 |

  - **Level with `w2z1_sqrt` at steps 200 and 500, slightly worse at step 50** (0.4 dB, SSIM 0.02 lower). Cause not known.
  - **Same group pattern as `w2z1_sqrt`:** no difference by eye (p 0.32 to 0.86); `male` 0 restores slightly better, SSIM p 0.020 and 0.028 at steps 200 and 500, PSNR p 0.07 to 0.11. None passes the Bonferroni threshold of 0.004. Both runs score the same scans, so the repeat points to the scans rather than to one model (inferred).
  - **Lowest PSNR at step 200:** 14.68, 15.10, 15.34 and 15.63 dB, the same range as `w2z1_sqrt`; whether they are the same scans was not checked.
- **What this says.** On the eye probe, `w2z1_sqrt24` is the best `w2z1` run so far, at the same memory and 13% less time per epoch, and it restores as well as `w2z1_sqrt` except at the lightest noise. Still to come: the band table, intensity numbers and finetune heads (`eval_next`), the images on the VM, and a second pretraining seed to size the seed effect.

`w2z1_xattn`, SLO through cross-attention at levels 0, 1 and 2 with 256 tokens of 160 (2026-11-24, run plan step 11, on the `w2z1_sqrt24` setup, batch 4, 43 to 48 s per epoch with a few slower ones, peak 10.52 GiB, 24,044,840 parameters; memory probe 10.23 GiB at batch 4, 19.69 GiB at batch 8, 0 failed steps):
- **Band sizes:** identical to `w2z1_sqrt24`, so the two losses are in the same units.
- **Validation loss, own SLO:** 0.0324 at epoch 1, then 0.0226, 0.0218, 0.0214 and 0.0213 at epochs 50, 100, 150 and 200 (lowest 0.021246 at epoch 192). `w2z1_sqrt24` without any SLO: 0.0225, 0.0219, 0.0215 and 0.0213 (lowest 0.021251). The SLO does not lower the loss.
- **SLO test:** at epoch 200, another scan's SLO 0.021369 (+0.32%) and no SLO 0.021321 (+0.09%) against own 0.021301. Over epochs 151 to 200, another scan's SLO is worse than own in all 50 epochs, by about 0.18% on average; no SLO is worse in 41 of 50 (2 ties), by about 0.07%. The adapter gave 0.25% and 0.15%, so more tokens and more cross-attention levels did not strengthen the SLO's effect.
- **Denoising on 20 validation scans** (2026-11-25; ratio to the average scan), own / another scan's / no SLO: 0.413 / 0.413 / 0.413 at step 50, 0.751 / 0.751 / 0.751 at 200, 0.886 / 0.887 / 0.887 at 500, 0.972 / 0.975 / 0.973 at 800 and 1.040 / 1.045 / 1.040 at 950. As with the adapter, no SLO is as good as the scan's own SLO at every step, and another scan's costs about 0.3 to 0.5% at 800 and 950.
- **Generated scans** (4, each from the SLO of a validation scan): generated / real by detail axes 1.005, 0.939, 0.760, 0.672, 0.595 and 0.462, so less fine texture than real. Intensity against the same 4 real scans: mean -0.653 (real -0.666), median -0.930 (-0.908), p99 0.846 (1.000), black share 0.201 (0.347), within-slice change 0.039 (0.084), slice-to-slice change 0.086 (0.123). The en-face check against the SLO each scan was generated from was not in this version of `sample.py`; it is added for the next run (rerun `sample.py` alone to get it).
- **Heads** (heads table, with `--slo-features`): the eye probe scores 0.983 pretrained and 0.946 from random init, because the SLO features show the optic disc even from an untrained SLO encoder. Sex is at chance (probe 0.517, finetune 0.593). In sex finetuning the training loss falls to 0.0016 while validation stays near 0.5, so the SLO is again memorised as a per-person fingerprint.
- **What this says.** 256 tokens and cross-attention at five levels give the same result as the adapter: the model reads the SLO a little, but its own SLO does not help it denoise at any noise level. This closes run plan step 11 with no SLO signal.

`w2z1_joint`, OCT and SLO in one volume (2026-11-24, run plan step 12, `--slo-slice raster`, 24 B-scans and the SLO pair in a 32 × 512 × 512 target, `--band-scale 0.5`, batch 4, 47 to 58 s per epoch with a few slower ones, peak 13.00 GiB, 19,731,792 parameters; memory probe 12.63 GiB at batch 4, 24.39 GiB at batch 8, 0 failed steps):
- **Band sizes:** 0.8056, 0.0481, 0.0286, 0.0224, 0.0188 and 0.0175 for 0 to 5 detail axes (smallest band 0.0115). They are measured with the SLO slice and 6 padded slices, so this loss is not comparable with the other runs.
- **Validation loss:** whole volume 0.0306 at epoch 1, then 0.0204, 0.0200, 0.0198 and 0.0196 at epochs 50, 100, 150 and 200 (lowest 0.019541 at epoch 192). OCT part 0.1386 at epoch 1 and 0.0986 at epoch 200; SLO part 0.0674 and 0.0335.
- **SLO test:** the OCT part scores 0.099462 with the scan's own SLO slice and 0.099472 with another scan's at epoch 200 (+0.01%). Over epochs 151 to 200, another scan's slice is worse in 36 of 50 epochs, by about 0.006% on average, about 30 times less than in `w2z1_xattn`. The model learns to generate the SLO slice (its loss halves) but does not use it to denoise the OCT. The `OCT` and `OCT with own SLO slice` numbers differ because the last validation batch holds one scan (53 = 13 × 4 + 1), which has no partner for the swap; the paired numbers leave it out.
- **Possible reasons (inferred, not tested):** both parts share one noise step, so wherever the OCT is noisy enough for the SLO to help, the SLO slice is just as noisy. The SLO's slow axis also lies along depth inside its slice, so B-scan k has to be matched to SLO row k across the volume rather than through nearby voxels. A separate noise step for each part, with the SLO sometimes left clean (as in UniDiffuser), would test the first reason and would also give an OCT-given-SLO model. Not built.
- **Denoising on 20 validation scans** (2026-11-25, whole volume including the SLO slice): 0.390, 0.743, 0.894, 0.977 and 1.048 at steps 50, 200, 500, 800 and 950.
- **Generated OCT** (4 scans, B-scans only): generated / real by detail axes 0.999, 0.992, 1.117, 1.064, 0.958 and 0.723 (same real scans and real sizes as `w2z1_xattn`). Intensity: mean -0.671 (real -0.666), median -0.909 (-0.908), p99 0.672 (1.000), black share 0.269 (0.347), within-slice change 0.067 (0.084), slice-to-slice change 0.112 (0.123). On these numbers it is the closest to real of any run so far, on 4 samples and one seed; the images have not been viewed yet.
- **Generated SLO slice:** too bright, mean 0.368 against -0.020 real (std 0.427 against 0.443).
- **OCT en-face view against the SLO:** real scans 0.147 with their own SLO (0.120 to 0.162 on these 4 scans) and -0.001 with another scan's. Generated pairs 0.353 with their own generated SLO and 0.341 with another sample's. Own and other are almost equal, so the generated OCT does not match its own SLO more than any other generated SLO: the model did not learn which SLO goes with which OCT. The high level of both means the generated samples share a pattern that real scans do not (fewer differences between samples, or an artefact at fixed positions). `sample.py` now prints the similarity between different scans to tell these apart; view the 8 images on the VM side by side.
- **Heads** (heads table): eye probe 0.866 pretrained against 0.679 from random init (random features of the SLO slice already show the optic disc); finetune 0.984 against 0.955. In eye finetuning from pretrained weights, validation AUROC reaches 0.99 at epoch 4; from random init it stays between 0.54 and 0.61 for 8 epochs and reaches 1.0 at epoch 12. Sex is at chance (probe 0.436, finetune 0.541), and unlike the SLO-encoder models the training loss stays near 0.67, so the joint model does not memorise sex in 20 epochs.
- **What this says.** The joint model generates OCTs with the most realistic texture and intensity so far and keeps the OCT features (eye probe 0.866), but it does not tie the OCT to the SLO. The own-step test is run plan step 13 (`w2z1_joint_step`).

Heads on the 106 test scans: AUROC with 95% CI.

| Encoder | Task | Probe, pretrained | Probe, random init | Finetune, pretrained | Finetune, random init |
|---|---|---|---|---|---|
| `w2z1` | right eye | 0.845 (0.758–0.915) | 0.541 (0.422–0.646) | 0.993 (0.978–1.000) | 0.988 (0.963–1.000) |
| `w2z1` | male | 0.486 (0.372–0.595) | 0.524 (0.409–0.651) | 0.591 (0.489–0.704) | 0.528 (0.418–0.651) |
| `w2z1` + SLO | right eye | 0.883 (0.818–0.941) | 0.530 (0.418–0.638) | 0.992 (0.981–0.999) | 0.993 (0.981–1.000) |
| `w2z1` + SLO | male | 0.503 (0.384–0.615) | 0.493 (0.378–0.602) | 0.566 (0.452–0.679) | 0.580 (0.474–0.686) |
| `w1z1` | right eye | 0.891 | 0.763 | 0.993 | 0.963 |
| `w1z1` | male | 0.532 | 0.506 | 0.548 | 0.543 |
| `w2z1_band` | right eye | 0.748 | 0.518 | 0.961 | 0.970 |
| `w2z1_band` | male | 0.546 | 0.433 | 0.474 | 0.481 |
| `w2z1_sqrt` | right eye | 0.821 (0.728–0.900) | 0.404 (0.299–0.516) | 0.988 (0.959–1.000) | 0.980 (0.946–0.999) |
| `w2z1_sqrt` | male | 0.440 (0.327–0.550) | 0.455 (0.339–0.577) | 0.581 (0.476–0.689) | 0.602 (0.490–0.712) |
| `w1z1_sqrt` | right eye | 0.841 (0.754–0.910) | 0.610 (0.506–0.713) | 0.988 (0.968–1.000) | 0.983 (0.962–0.996) |
| `w1z1_sqrt` | male | 0.525 (0.409–0.632) | 0.559 (0.435–0.669) | 0.555 (0.450–0.668) | 0.593 (0.482–0.696) |
| `w2z1_adapter` | right eye | 0.948 (0.907–0.982) | 0.410 (0.308–0.519) | pending | pending |
| `w2z1_adapter` | male | 0.533 (0.422–0.646) | 0.542 (0.430–0.664) | 0.578 (0.462–0.684) | 0.581 (0.474–0.686) |
| `w2z1_xattn`, SLO features | right eye | 0.983 (0.955–0.998) | 0.946 (0.901–0.981) | 0.999 (0.997–1.000) | 0.995 (0.983–1.000) |
| `w2z1_xattn`, SLO features | male | 0.517 (0.405–0.635) | 0.593 (0.469–0.702) | 0.593 (0.484–0.701) | 0.560 (0.456–0.666) |
| `w2z1_joint` | right eye | 0.866 (0.784–0.928) | 0.679 (0.565–0.777) | 0.984 (0.954–1.000) | 0.955 (0.920–0.985) |
| `w2z1_joint` | male | 0.436 (0.332–0.540) | 0.539 (0.419–0.643) | 0.541 (0.434–0.649) | 0.610 (0.504–0.720) |

The `w1z1` and `w2z1_band` rows come from the summary table, which has no CIs; they are in each `results.json`. The random-init controls of `w2z1_band`, `w2z1_sqrt` and `w1z1_sqrt` see band-scaled input, so their random-init numbers are not comparable with the other rows.

`w2z1_sqrt` eye finetune curves (2026-11-12, one seed): from pretrained weights, validation AUROC 0.94 at epoch 3 and 0.996 at epoch 5; from random init, 0.61 at epoch 5, 0.95 at epoch 9 and 0.98 at epoch 11. Pretraining speeds up finetuning here as it does for `w2z1`.

`w1z1_sqrt` eye finetune curves (2026-11-18, one seed): from pretrained weights, validation AUROC stays between 0.45 and 0.61 for 8 epochs, then 0.81 at epoch 9, 0.94 at epoch 10 and 0.99 at epoch 11; from random init, 0.92 at epoch 12, 0.96 at epoch 15 and 0.99 at epoch 16. Pretraining is about 5 epochs ahead, but starts later than for `w2z1_sqrt`. The curve for plain `w1z1` was not recorded.

Sampling (denoising on 20 validation scans; 4 generated scans):
- **Denoising ratio, `w2z1`:** 0.256, 0.291, 0.365, 0.615 and 0.980 at noise steps 50, 200, 500, 800 and 950.
- **SLO model:** within 0.02 of these with any SLO. Own SLO beats another scan's only at step 800 (0.627 vs 0.631) and step 950 (0.982 vs 0.987).
- **Smoothness:** generated scans are much smoother than real ones. Mean change between neighbouring pixels within a B-scan is 0.003 against 0.084 real; between slices it is about 0.03 against 0.12.
- **Background:** real scans are 35% black background; generated scans 26% (`w2z1`) and 16% (SLO model).

What this shows:
- **Pretraining learned features a head can use.** Frozen pretrained features predict the eye (AUROC 0.85 to 0.88), where frozen random features do not (0.53 to 0.54). Finetuning learns the eye just as well from scratch, so that task cannot separate pretrained from random. Sex is at chance everywhere, as expected with 372 training scans.
- **The SLO model makes little use of the SLO.**
  - Own SLO is about 1% better than another scan's at epoch 200, and the model is no better than `w2z1` without an SLO.
  - A wrong SLO is slightly worse than none from epoch 100 on, so the SLO is not ignored entirely.
  - The SLO model's eye probe may draw partly on the SLO itself, which shows the optic disc.
  - In the random-init control the SLO has no effect at all, because MONAI starts the cross-attention output at zero.
- **Finetuning curves** (from the full evaluation log, one seed):
  - **Faster with pretraining.** For the eye, `w2z1` finetuning from pretrained weights reaches validation AUROC 0.94 at epoch 5 and 0.99 at epoch 6. From random init it is 0.75 and 0.81 at those epochs, and 0.97 only at epoch 9. Both end at about 0.99.
  - **The SLO shows the eye directly.** The SLO model learns the eye within 1 to 2 epochs even from random init (0.98 at epoch 2). For SLO models, the eye task does not test the OCT features.
  - **The SLO model memorises sex.** Its training loss for sex falls to 0.09 (pretrained) and 0.46 (random init), against about 0.68 for `w2z1`, while validation stays near chance. The SLO seems to act as an easy per-person fingerprint.
  - **Epoch choice by validation AUROC is noise for tasks near chance.** For sex, validation AUROC jumps between 0.39 and 0.71 from epoch to epoch on 53 scans.
- **Generated scans lack fine texture,** as in the first baseline. The band table (4 generated scans per run, 2026-11-10) shows where:

  | Detail axes | Real size (`w2z1`) | Generated / real: `w2z1` | SLO model | `w1z1` | `w2z1_band` | `w2z1_sqrt` | `w1z1_sqrt` | `w2z1_adapter` |
  |---|---|---|---|---|---|---|---|---|
  | 0 (coarse image) | 0.863 | 1.013 | 1.028 | 1.016 | 0.749 | 0.955 | 1.027 | 1.080 |
  | 1 | 0.048 | 0.799 | 0.840 | 0.770 | 1.191 | 1.366 | 1.138 | 1.332 |
  | 2 | 0.019 | 0.204 | 0.189 | 0.136 | 2.268 | 1.979 | 0.930 | 0.751 |
  | 3 | 0.014 | 0.046 | 0.042 | 0.044 | 2.515 | 1.942 | 0.770 | 0.537 |
  | 4 | 0.011 | 0.018 | 0.014 | none | 2.567 | 1.723 | none | 0.482 |
  | 5 | 0.010 | 0.013 | 0.011 | none | 2.239 | 1.253 | none | 0.440 |

  Coarse content and one-axis detail are reproduced; bands with three or more detail axes, mostly speckle and fine texture, come out at 1 to 5% of their real size. The real fine bands (0.010 to 0.019; 0.017 to 0.027 in `w1z1`) are about the size of the noise added at the last step (0.010) and well below that at step 10 (0.047). This fits the idea that the scaling in `wavelet.forward` hides fine detail under the noise for all but the last few steps and leaves it almost no weight in the loss.
- **Label fractions** (eye probe, 2026-11-10). Heads trained on part of the training scans, five random subsets per fraction; validation and test stay whole. Test AUROC, mean (min to max):

  | Training scans with labels | `w2z1` pretrained | `w2z1` random init | `w1z1` pretrained | `w1z1` random init |
  |---|---|---|---|---|
  | 37 (10%) | 0.760 (0.701–0.807) | 0.547 (0.434–0.662) | 0.801 (0.740–0.830) | 0.567 (0.508–0.652) |
  | 93 (25%) | 0.808 (0.785–0.831) | 0.545 (0.481–0.646) | 0.849 (0.794–0.878) | 0.627 (0.545–0.679) |
  | 186 (50%) | 0.840 (0.798–0.870) | 0.556 (0.513–0.619) | 0.868 (0.810–0.901) | 0.707 (0.676–0.735) |
  | 372 (100%) | 0.845 | 0.541 | 0.891 | 0.763 |

  - **Pretraining is what makes the probe work.** On pretrained features, a linear head learns the eye from 37 scans (0.76 to 0.80). On `w2z1`'s frozen random features it fails even with all 372 (0.54).
  - **`w1z1` scores higher at every fraction,** by 0.03 to 0.05.
  - **Part of that is the design itself.** `w1z1`'s random features improve with more labels (0.57 to 0.76), because its finer grid keeps more spatial detail even untrained. So pretraining adds 0.13 to 0.23 for `w1z1` and 0.21 to 0.30 for `w2z1`.
  - **These are small differences.** There is one easy task and one test set of 106 scans, and the gaps are smaller than a single CI.
  - **`w1z1` costs more:** 20 GB and about 125 s per epoch, against 9.3 GB and 44 s for `w2z1`.

  Training from scratch with all labels still does better than any probe (finetune from random init, 0.988).
- **Finetuning at label fractions** (`w2z1`, eye, subsets 0 to 2, finished 2026-11-11). Training steps are matched across fractions. Test AUROC, mean (min to max); the probe column is the mean over subsets 0 to 4 from the table above:

  | Training scans with labels | Probe, pretrained | Finetune, pretrained | Finetune, random init |
  |---|---|---|---|
  | 37 (10%) | 0.760 | 0.716 (0.627–0.789) | 0.798 (0.616–0.928) |
  | 93 (25%) | 0.808 | 0.844 (0.814–0.902) | 0.922 (0.918–0.926) |
  | 186 (50%) | 0.840 | 0.971 (0.957–0.985) | 0.969 (0.947–0.984) |
  | 372 (100%) | 0.845 | 0.993 | 0.988 |

  - **From 186 labelled scans up, the start does not matter.** Both reach 0.97 to 0.99.
  - **Below that, random init scores higher on test.**
    - At 25% the test ranges still do not overlap (0.918–0.926 against 0.814–0.902), but each run's 95% CI is wide (subset 2 pretrained: 0.902, CI 0.836 to 0.956).
    - Validation agrees on average (0.942 against 0.879) but not subset by subset. On subset 0, the pretrained run scored higher on validation (best 0.912 against 0.894) but lower on test (about 0.815 against about 0.92).
    - At 10% the test means are 0.80 against 0.72, but the ranges overlap widely.
  - **Finetuning the pretrained encoder on few labels gains little over its frozen probe** (0.72 against 0.76 at 10%, 0.84 against 0.81 at 25%).
  - **Training curves, 25%.**
    - Subset 0: the pretrained run learns the eye sooner, validation 0.88 at epoch 12 against 0.53 from random init. Subset 2, pretrained: slower, 0.63 at epoch 12 and 0.84 at epoch 16.
    - Both subset 0 runs fit the 93 training scans completely by epoch 32 (training loss 0.001), and validation then stays flat (0.91 and 0.89). Subset 2 pretrained does the same by epoch 28.
    - So the idea that pretrained features let the head memorise before the encoder learns does not fit. Random init memorises just as completely, only later.
  - **What this says.** On the eye task, pretraining clearly helps the frozen probe and speeds up finetuning. With 53 validation and 106 test scans, which start gives the better final finetune at small fractions cannot be told: the two sets rank them differently. The eye is a coarse cue that a network learns from scratch. A diagnosis task is the real test.
  - **One run failed the first time:** 25% pretrained, subset 2, stopped with a `RuntimeError` while `eval_w1z1` shared the GPU. The script prints only the error type. Rerun alone on 2026-11-11, it finished: test 0.902, validation 0.927, best epoch 64 of 80. A GPU out-of-memory error would have printed `OutOfMemoryError`. `RuntimeError` is what the loader raises when a scan fails to read, or when a loader worker dies, so a read hiccup on the mount is the more likely cause (not confirmed). Running finetunes one at a time on the 5090 is still safer.
- **Band scaling** (`w2z1_band`, evaluated 2026-11-11; generated images not yet viewed). The run is the same as `w2z1` apart from `--band-scale`, and the band table uses the same 4 validation scans (the real column matches the earlier one).
  - **Texture is back, but overshoots.** Bands with 2 to 5 detail axes come out at 2.2 to 2.6 times their real size (`w2z1`: 0.01 to 0.20). Within-slice change is 0.142 against 0.084 real (`w2z1`: 0.003); slice-to-slice 0.192 against 0.123. Part of the excess comes from the comparison scans: those 4 are smoother than the training average. Against the band sizes measured on the 372 training scans, the generated fine bands are 1.0 to 1.5 times as large.
  - **The coarse image got worse.** Generated / real for the zero-axes band is 0.749 (`w2z1`: 1.013). Generated scans are too bright: mean -0.267 against -0.665 real, median -0.355 against -0.908, black share 0.081 against 0.346. Added fine detail alone cannot lift the median that far, so the coarse image itself is off (inferred). This is the risk noted under Open items: the coarse band went from about 95% of the loss to 1 of 32 equal bands.
  - **The eye probe got worse at every label count.** Test AUROC, mean over subsets:

    | Training scans with labels | `w2z1` pretrained | `w2z1` random init | `w2z1_band` pretrained | `w2z1_band` random init |
    |---|---|---|---|---|
    | 37 (10%) | 0.760 | 0.547 | 0.542 (0.454–0.616) | 0.504 (0.390–0.606) |
    | 93 (25%) | 0.808 | 0.545 | 0.556 (0.514–0.612) | 0.483 (0.379–0.567) |
    | 186 (50%) | 0.840 | 0.556 | 0.630 (0.598–0.657) | 0.482 (0.439–0.545) |
    | 372 (100%) | 0.845 | 0.541 | 0.748 | 0.518 |

    What pretraining adds over random init falls from 0.21–0.30 to 0.04–0.23, and the loss is largest with few labels. The eye signal is still in the features (0.75 with all labels), but a linear head needs many more scans to find it.
  - **Finetuning and sex are not informative here.** Eye finetune: 0.961 pretrained and 0.970 random init, one seed each. Sex is at chance everywhere (0.43 to 0.55).
  - **The denoising ratio of 1.05 at step 950 is expected,** not a fault: the "average scan" is the mean of the same 20 scans, which gives it a head start of 1/(1 − 1/20) = 1.053 over a model that has learned the average of all scans.
  - **What this says.** Full-strength band scaling trades the coarse anatomy for speckle. The eye is a coarse cue, so this task favours `w2z1`; whether texture helps a diagnosis task is still open. `w2z1` stays the main model for now.
- **Milder band scaling** (`w2z1_sqrt`, `--band-scale 0.5`, evaluated 2026-11-12; generated images not yet viewed). Same 4 validation scans in the band table as the other runs.
  - **Coarse image kept, texture back at about the right size.** Generated / real is 0.955 for the coarse band and 1.25 to 1.98 for 1 to 5 detail axes. Against the band sizes measured on the 372 training scans, which are larger than those of the 4 comparison scans, the fine bands come out at 0.73 to 1.27 times.
  - **Intensity:**

    | | Real | `w2z1` | `w2z1_band` | `w2z1_sqrt` |
    |---|---|---|---|---|
    | Mean | -0.665 | | -0.267 | -0.579 |
    | Median | -0.908 | | -0.355 | -0.827 |
    | 99th percentile | 1.000 | | 1.000 | 0.886 |
    | Black share | 0.346 | 0.258 | 0.081 | 0.176 |
    | Within-slice change | 0.084 | 0.003 | 0.142 | 0.110 |
    | Slice-to-slice change | 0.123 | about 0.03 | 0.192 | 0.172 |

    Brightness is close to real. The lower black share is expected once there is texture: real background is clipped to exactly -1, and any fine detail on top of it lifts about half of those pixels above the -0.99 cut-off (inferred). The brightest parts are a little dimmer than real. Generated scans change somewhat more between neighbouring pixels and slices than these 4 scans, which are smoother than the average training scan.
  - **The denoising ratio of 1.04 at step 950** is the same head start for the in-sample average as for `w2z1_band`; its other ratios are in this run's own units and are not compared.
  - Eye probe at label fractions, pretrained, test AUROC mean (min to max) over five subsets:

  | Training scans with labels | `w2z1` | `w2z1_band` | `w2z1_sqrt` |
  |---|---|---|---|
  | 37 (10%) | 0.760 (0.701–0.807) | 0.542 (0.454–0.616) | 0.745 (0.665–0.843) |
  | 93 (25%) | 0.808 (0.785–0.831) | 0.556 (0.514–0.612) | 0.805 (0.773–0.824) |
  | 186 (50%) | 0.840 (0.798–0.870) | 0.630 (0.598–0.657) | 0.816 (0.804–0.843) |
  | 372 (100%) | 0.845 | 0.748 | 0.821 |

  - The milder scaling keeps the frozen features about as useful as `w2z1`'s: level with it at 37 and 93 labels, about 0.02 lower from 186 up, inside the spread over subsets.
  - Its random-init probe is at chance (0.40 to 0.50) and sees band-scaled input, so it is not compared across runs.
  - **What this says.** Square-root scaling keeps the coarse image and the probe of `w2z1` and brings fine detail to about its real size. It is the main candidate for the `w2z1` design, pending a look at the images on the VM (layer order and shape, background). The default of `--band-scale` stays off until the lab agrees.
- The evaluation ran beside `w1z1` without trouble; the GPU showed 24.6 GB in use before it started. With the SLO, the encoder has 10,287,288 parameters.
- **Milder band scaling on `w1z1`** (`w1z1_sqrt`, `--band-scale 0.5`, typed back 2026-11-18; generated images not yet viewed). Same 4 validation scans in the band table as the other runs.
  - **Texture is back, the finest detail still short.** Generated / real is 1.027 for the coarse band and 1.138, 0.930 and 0.770 for 1, 2 and 3 detail axes (`w1z1`: 0.770, 0.136 and 0.044). Against the band sizes measured on the 372 training scans, the 1 to 3 axis bands come out at 0.86, 0.56 and 0.43 times (`w2z1_sqrt`: 0.73 to 1.27).
  - **Intensity:**

    | | Real | `w1z1` | `w1z1_sqrt` |
    |---|---|---|---|
    | Mean | -0.665 | -0.678 | -0.681 |
    | Median | -0.908 | -0.953 | -0.983 |
    | 99th percentile | 1.000 | 0.840 | 0.906 |
    | Black share | 0.346 | 0.164 | 0.461 |
    | Within-slice change | 0.084 | 0.006 | 0.052 |
    | Slice-to-slice change | 0.123 | 0.046 | 0.097 |

    Mean brightness is close to real. Black share moves the other way from `w2z1_sqrt` (0.176): these 4 generated scans have more pure background than real ones, and a darker median. Whether that is a thinner or shifted retina, or chance in 4 scans, needs the images (not known).
  - **Denoising ratio:** 0.201, 0.554, 0.764, 0.858 and 0.947 at steps 50, 200, 500, 800 and 950, in this run's own units; not compared.
  - Eye probe at label fractions, test AUROC mean (min to max) over five subsets. A subset is the same patients in every run (same split and seeds in `label_subset`):

  | Training scans with labels | `w1z1` pretrained | `w1z1_sqrt` pretrained | `w1z1_sqrt` random init | `w2z1_sqrt` pretrained |
  |---|---|---|---|---|
  | 37 (10%) | 0.801 (0.740–0.830) | 0.660 (0.631–0.717) | 0.502 (0.423–0.571) | 0.745 (0.665–0.843) |
  | 93 (25%) | 0.849 (0.794–0.878) | 0.754 (0.745–0.770) | 0.587 (0.523–0.655) | 0.805 (0.773–0.824) |
  | 186 (50%) | 0.868 (0.810–0.901) | 0.805 (0.781–0.828) | 0.591 (0.567–0.611) | 0.816 (0.804–0.843) |
  | 372 (100%) | 0.891 | 0.841 | 0.610 | 0.821 |

  - **The frozen features got worse for the eye.** `w1z1_sqrt` is below `w1z1` at every label count, by 0.05 (all labels) to 0.14 (37 labels). At 37 labels it is lower on each of the 5 subsets, by 0.10 to 0.19, and the ranges do not overlap at 37 or 93 labels. On `w2z1` the same scaling cost 0.00 to 0.02.
  - **Against `w2z1_sqrt`,** it is lower with 37 and 93 labels (0.66 and 0.75 against 0.75 and 0.81) and level from 186 up.
  - Its random-init probe (0.50 to 0.61) sees band-scaled input, so it is not compared across runs. Pretraining adds 0.16 to 0.23 over it.
  - **Finetuning and sex:** eye finetune 0.988 pretrained and 0.983 random init, one seed each. Sex is at chance everywhere (0.52 to 0.59).
  - **One pretraining run per design.** The fixed subsets rule out which patients carry labels as the cause, but how far a second pretraining seed would move these numbers is not measured.
  - **What this says.** On `w1z1`, square-root scaling brings texture back (within-slice change 0.006 to 0.052) but costs the eye probe most where labels are few, and the finest bands stay short of the training-scan sizes. `w2z1_sqrt` keeps its probe, has fine detail at least as close to real, and costs 9.3 GB and 44 s per epoch against 20 GB and 125 s. It stays the main candidate; `w1z1_sqrt` does not earn its cost on this evidence. The eye is a coarse cue, so a diagnosis task could still rank them differently.
- **Scan report** (`w2z1_sqrt`, 2026-11-18; plots viewed on the VM 2026-11-19: PSNR histograms have a left tail at every step; SSIM is symmetric at step 50, right-tailed at 200, wider and lower at 500; in `worst_003.png` the original middle B-scan, and so the reconstruction, is dark in its top third, while the other three worst views look normal). All 159 validation and test scans read, 321 s. The counts per `male` and `righteye` value match the heads' logs. Median (25th to 75th percentile):

  | Noise step | PSNR, dB | Noisy input PSNR, dB | SSIM |
  |---|---|---|---|
  | 50 | 24.71 (22.51–26.11) | 20.30 (20.13–20.44) | 0.653 (0.633–0.674) |
  | 200 | 21.83 (19.54–23.42) | 9.99 (9.98–10.00) | 0.402 (0.348–0.491) |
  | 500 | 20.18 (18.08–21.50) | 5.88 (5.75–6.04) | 0.331 (0.238–0.443) |

  - **The model removes most of the added noise:** 4.4, 11.8 and 14.3 dB above the noisy input. 21.8 dB is a root mean square pixel error of about 0.16 on the -1 to 1 scale.
  - **SSIM falls faster than PSNR.** A one-step restoration is the model's average guess, so it is smooth: speckle hidden under the noise cannot be guessed back, and SSIM penalises missing texture more than PSNR does (inferred).
  - **No difference by eye** (p 0.33 to 0.87).
  - **`male` 0 scans restore slightly better than `male` 1:** PSNR 0.4 to 0.9 dB higher (p 0.07 to 0.16), SSIM p 0.026 and 0.032 at steps 200 and 500. None of the 12 comparisons passes a Bonferroni threshold (0.05 / 12 = 0.004), and the pilot has no age or quality score to check for confounding.
  - **Lowest PSNR at step 200:** 14.7 to 15.7 dB, about 6 dB below the median, in `worst_*.png`.
  - These PSNR values belong to this test (one step, this run's noise units). They cannot be compared with PSNR values in papers.

## Full pull from Peter (plan from 2026-11-09)

1. **Disease lists (Hayagreev).** For each condition in the grant, the DataDirect MRNs with that diagnosis and an OCT billing code, capped near 1,000 per condition.
2. **All-OCT list (Hayagreev).** Every MRN with an OCT billing code.
3. **Pretraining list.** The all-OCT list minus every disease-list MRN, by set arithmetic. Disease patients never enter pretraining.
4. **Controls.** Try DataDirect exclusion (NOT) queries, or the same set arithmetic. If the remaining list is long, also exclude ICD codes that affect the eye, such as vascular and ocular conditions.
5. **Peter** receives only MRN lists. He doesn't need to know the diseases.
6. **Pseudo-IDs.** How MRNs become IDs that stay the same across visits will be agreed with Peter in a short call. Matthias is away the week of 2026-11-09, so the call is likely the following Wednesday.

What the pipeline then needs:
- In `oct_wd/cohort.py`, the patient field must switch from `anon_id` to the agreed pseudo-ID, so the split is truly by patient.
- The returned table needs the scan date and the scan pattern or slice count.
- The MRN key never enters the image tree.

## Open items

- Matthias and Hayagreev count 498 pilot scans; the folders give 531 with 25 B-scans.
- Which downstream tasks come first, and how controls are defined.
- `w1z1` against `w2z1`: compare on the heads and the generated-scan numbers once `w1z1` is evaluated, not on loss or denoising ratio.
- Generated scans lack texture: measured on 2026-11-10 (band table under Results). If the lost detail is mostly speckle, the features may still serve classification, but small lesions also live at fine scales. Fix tried 2026-11-11: `--band-scale` (run plan step 5). As feared, the coarse image, which held about 95% of the coefficient power on the training scans, got worse once it became 1 of 32 equal bands; fine detail overshoots and the eye probe dropped (Results, "Band scaling"). `--band-scale 0.5` (run plan step 6, 2026-11-12) keeps the coarse image and the eye probe and brings fine detail to about its real size (Results, "Milder band scaling"). The same scaling on `w1z1` (`w1z1_sqrt`, 2026-11-13) brings texture back but lowers the eye probe by 0.05 to 0.14 (Results, "Milder band scaling on `w1z1`"). Still to do: view the `w2z1_sqrt`, `w2z1_band` and `w1z1_sqrt` images on the VM, and decide with the lab whether `w2z1_sqrt` replaces `w2z1` as the main model.
- A stronger SLO test than the loss: correlate each generated scan's en-face image with its own SLO and with another scan's, using the same measure as the QC (real scans: 0.197 own vs 0.000 other). Built for the joint model (`sample.py`, 2026-11-23); for the cross-attention models the generated SLO is the input, so the same measure applies to their samples against the SLO they were made from (not yet added).
- **No distillation step for OCT yet.** Matthias listed distillation among the steps Hayagreev should be able to run. The heads read the pretrained diffusion encoder at noise step 0 directly. The brain pipeline's CleanDIFT distillation (`src/train_cleandift_multilayer.py` in the brain repo) could be ported. In the brain journal work, fine-tuning the undistilled 19.8M model scored like the distilled student on the image-level tasks (2026-11-22), so its value for OCT is open.
- **Does `anon_id` identify a person?** In the pilot each `anon_id` has exactly one scan: the label-fraction runs report 372 training "patients" for 372 training scans. If one person's two eyes, or two visits, carry different `anon_id`s, they can fall on both sides of the split. That cannot leak the eye label, but it would leak person-level labels such as sex or a diagnosis. Ask Hayagreev whether a person-level ID exists for the pilot. For the full pull, the agreed pseudo-ID must be per person. Bootstrap CIs should then resample persons.
- Label fractions on the eye task are done for `w2z1` (probe and finetune) and `w1z1` (probe). Pretrained finetuning scored lower than random init on test at 10% and 25%, but validation disagrees on at least one subset. The training loss rules out early memorisation by the head. The eye task is too easy and the evaluation sets too small to settle this. Repeat the fractions on a diagnosis task once one is chosen.
- With 53 validation scans, choose finetune epochs in a steadier way: a fixed number of epochs set in advance, the validation loss, or cross-validation on train plus validation.
- Next modelling idea: a latent-feature loss alongside the diffusion loss (REPA-style), possibly aligned to SLO features at the matching retinal location.
