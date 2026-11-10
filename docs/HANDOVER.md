# OCT wavelet diffusion: handover

Status as of 2026-11-09. Written so that someone else (Hayagreev, or whoever picks this up) can run every step without asking.

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
8. **Downstream.** `scripts/downstream.py` keeps the input layer, encoder and bottleneck, drops the decoder, and pools each level into a 560-number feature vector. Then either:
   - `probe`: frozen encoder plus a linear head. Weight decay is chosen on validation.
   - `finetune`: encoder plus head trained together. The epoch is chosen on validation.

   Both score the test split once, with a bootstrap 95% CI. `--random-init` runs the same thing without pretrained weights, as the control.

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

`male` and `righteye` are the only labels the pilot has. They check that the heads run end to end. They are not research results.

### Reading the logs

| Line | Meaning |
|---|---|
| `epoch N: train mse … val mse …` | Diffusion loss. Lower is better. Compare runs of the same design only. The wavelet scaling differs between `w2z1` and `w1z1`, so their losses are on different scales |
| `val mse X (own SLO)  other SLO Y  no SLO Z` | The SLO test. X clearly below Y means the model uses the SLO |
| Denoising table in `sample.py` | Ratio of model error to the error of guessing the average scan. Below 1 is better than the average; this ratio can be compared across designs |
| `test auroc …` in `downstream.py` | The head on held-out patients. Compare pretrained against `--random-init` |

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
- After the runs above:
  - Does `w1z1` beat `w2z1` on the denoising ratio and the heads?
  - Does the SLO model use the SLO (own vs other)?
- Next modelling idea: a latent-feature loss alongside the diffusion loss (REPA-style), possibly aligned to SLO features at the matching retinal location.
