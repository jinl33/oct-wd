import argparse
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generative.networks.schedulers import DDPMScheduler

from oct_wd import cohort as co
from oct_wd import wavelet as wv
from oct_wd.model import CONTEXT_DIM, Z_DOWN, align_loss, build_model, options, predict, to_device
from oct_wd.volumes import OctVolumes


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="w2z1", choices=sorted(wv.DESIGNS))
    ap.add_argument("--n-slices", type=int, default=25)
    ap.add_argument("--target", type=int, nargs=3, default=[32, 512, 512])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="batches per optimizer step")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--slo-size", type=int, default=0, help="0 trains without SLO; otherwise the SLO is resized to this size and used as a conditioner")
    ap.add_argument("--slo-drop", type=float, default=0.2, help="share of training steps that see no SLO")
    ap.add_argument("--slo-adapter", action="store_true", help="also add SLO features, sampled along each B-scan's line on the SLO, to the U-Net encoder at every level, and a pooled SLO vector to the time embedding; needs --slo-size")
    ap.add_argument("--slo-tokens", type=int, default=8, help="the SLO encoder's last map is pooled to this many tokens per side for cross-attention (8 gives 64 tokens, 16 gives 256)")
    ap.add_argument("--slo-dim", type=int, default=CONTEXT_DIM, help="size of each SLO token")
    ap.add_argument("--slo-xattn", type=int, nargs="*", default=[], help="U-Net encoder levels (0 to 4) that get an extra cross-attention to the SLO tokens, on top of the built-in one at the two deepest levels; needs --slo-size")
    ap.add_argument("--slo-slice", default=None, choices=("raster", "full"), help="joint model: append the SLO as its own slice after the B-scans, resampled onto the scan raster (raster) or whole (full); needs an even number of B-scans and two free slices in --target")
    ap.add_argument("--slo-align", type=float, default=0.0, help="weight of a contrastive loss that aligns the OCT encoder with a separate SLO encoder, at each place along the scan lines and over the whole scan; the SLO is not fed to the denoiser; needs --slo-size")
    ap.add_argument("--slo-align-loss", default="contrastive", choices=("contrastive", "jepa"), help="with --slo-align: contrastive (InfoNCE, other scans and places as negatives) or jepa (each modality predicts the other's features from slowly updated copies of the encoders, no negatives)")
    ap.add_argument("--slo-sigreg", type=float, default=0.0, help="with --slo-align-loss jepa: weight lambda of SIGReg (LeJEPA's guard against collapse) on the OCT and SLO features at each place; the alignment term becomes (1 - lambda) x prediction + lambda x SIGReg; LeJEPA uses 0.05")
    ap.add_argument("--slo-own-step", action="store_true", help="joint model: give the SLO slice its own noise step, drawn apart from the OCT's, so the model also learns the OCT given a clean SLO; needs --slo-slice")
    ap.add_argument("--slo-clean", type=float, default=0.25, help="with --slo-own-step: share of training scans whose SLO slice is left clean")
    ap.add_argument("--slo-same", type=float, default=0.25, help="with --slo-own-step: share of training scans whose SLO slice gets the OCT's noise step; the rest get an independent step")
    ap.add_argument("--keep-slices", type=int, default=0, help="keep this many middle B-scans of each scan, 0 for all; 24 of 25 drops the last")
    ap.add_argument("--z-down", type=int, nargs=4, default=list(Z_DOWN), choices=(0, 1), help="for each of the U-Net's four downsamplings, 1 if it also halves the slice axis")
    ap.add_argument("--band-scale", type=float, nargs="?", const=1.0, default=0.0, help="divide each wavelet band by its root mean square on the training scans raised to this power (1 if given alone, 0.5 for the square root) before noise is added, so fine detail is not buried under the noise; sampling and heads read the divisors from the checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=None, help="seed of the train, validation and test split; defaults to --seed, so a run with a new --seed and --split-seed 42 keeps the split of earlier runs")
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch, for smoke tests")
    ap.add_argument("--keep", type=int, default=3, help="number of recent checkpoints to keep")
    ap.add_argument("--resume", action="store_true", help="continue from latest.pt in --out")
    ap.add_argument("--auto-resume", action="store_true", help="continue if latest.pt exists, otherwise start")
    return ap.parse_args()


def git_hash():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def unpack(batch, device, conditioned):
    if not conditioned:
        return batch.to(device, non_blocking=True), None, None
    other = to_device(batch[2], device) if len(batch) > 2 else None
    return batch[0].to(device, non_blocking=True), to_device(batch[1], device), other


@torch.no_grad()
def measure_bands(scans, args, axes, device):
    loader = torch.utils.data.DataLoader(OctVolumes(scans, args.target, keep=args.keep_slices, slo_slice=args.slo_slice), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    total, values, n = torch.zeros(wv.channels(axes), dtype=torch.float64, device=device), 0, 0
    for step, vol in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            break
        x = wv.forward(vol.to(device), axes)
        total += x.pow(2).sum(dim=(0, 2, 3, 4)).double()
        values += x[:, 0].numel()
        n += x.shape[0]
    return (total / values).sqrt().float().clamp_min(1e-4).cpu(), n


def region_mse(pred, x0, lo, hi):
    return torch.nn.functional.mse_loss(pred[:, :, lo:hi].float(), x0[:, :, lo:hi], reduction="sum").item() / x0[0, :, lo:hi].numel()


def slo_steps(t, clean, same):
    u = torch.rand(t.shape, device=t.device)
    own = torch.randint(0, 1000, t.shape, device=t.device)
    return torch.where(u < clean, torch.full_like(t, -1), torch.where(u < clean + same, t, own))


def with_slo(noisy, x0_slo, noise_slo, t_slo, sched, s0, s1):
    x = noisy.clone()
    x[:, :, s0:s1] = torch.where((t_slo < 0)[:, None, None, None, None], x0_slo, sched.add_noise(original_samples=x0_slo, noise=noise_slo, timesteps=t_slo.clamp_min(0)))
    return x


@torch.no_grad()
def validate(model, loader, axes, sched, device, use_amp, max_steps, conditioned, band, joint=None, own_step=False, aligned=False):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    names = ("own", "other", "none") if conditioned else ("none",)
    total, count = dict.fromkeys(names, 0.0), 0
    parts, pairs = dict.fromkeys(("oct", "slo", "oct_other_slo", "oct_paired", "oct_clean_own", "oct_clean_other", "oct_slo_noise"), 0.0), 0
    near = dict.fromkeys(("own", "other", "n_own", "n_other"), 0.0)
    whole_oct, whole_slo = [], []
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        vol, slo, other = unpack(batch, device, conditioned or aligned)
        x0 = wv.scaled(wv.forward(vol, axes), band)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        t = torch.randint(0, 1000, (x0.shape[0],), generator=gen).to(device)
        noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
        for name in names:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = predict(model, noisy, t, {"own": slo, "other": other, "none": None}[name])
            total[name] += torch.nn.functional.mse_loss(pred.float(), x0, reduction="sum").item() / x0[0].numel()
        if joint is not None:
            (lo, hi), (s0, s1) = joint
            parts["oct"] += region_mse(pred, x0, lo, hi) * x0.shape[0]
            parts["slo"] += region_mse(pred, x0, s0, s1) * x0.shape[0]
            if x0.shape[0] > 1:
                swapped = noisy.clone()
                swapped[:, :, s0:s1] = sched.add_noise(original_samples=x0.roll(1, 0)[:, :, s0:s1], noise=noise[:, :, s0:s1], timesteps=t)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_other = predict(model, swapped, t, None)
                parts["oct_other_slo"] += region_mse(pred_other, x0, lo, hi) * x0.shape[0]
                parts["oct_paired"] += region_mse(pred, x0, lo, hi) * x0.shape[0]
                pairs += x0.shape[0]
                if own_step:
                    clean = torch.full_like(t, -1)
                    for key, x_slo, t_slo in (("oct_clean_own", x0, clean), ("oct_clean_other", x0.roll(1, 0), clean), ("oct_slo_noise", x0, torch.full_like(t, 999))):
                        given = with_slo(noisy, x_slo[:, :, s0:s1], noise[:, :, s0:s1], t_slo, sched, s0, s1)
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            p = predict(model, given, t, None, t_slo)
                        parts[key] += region_mse(p, x0, lo, hi) * x0.shape[0]
        if aligned:
            with torch.amp.autocast("cuda", enabled=use_amp):
                o, v, valid, go, gs = model.match(x0, slo)
            near["own"] += (o * v).sum(-1)[valid].sum().item()
            near["n_own"] += valid.sum().item()
            if x0.shape[0] > 1:
                both = valid & valid.roll(1, 0)
                near["other"] += (o * v.roll(1, 0)).sum(-1)[both].sum().item()
                near["n_other"] += both.sum().item()
            whole_oct.append(go)
            whole_slo.append(gs)
        count += x0.shape[0]
    model.train()
    out = {k: v / max(count, 1) for k, v in total.items()}
    if aligned:
        sim = torch.cat(whole_oct) @ torch.cat(whole_slo).t()
        out.update(near_own=near["own"] / max(near["n_own"], 1), near_other=near["other"] / max(near["n_other"], 1),
                   whole_own=sim.diagonal().mean().item(), whole_other=(sim.sum() - sim.diagonal().sum()).item() / max(sim.numel() - len(sim), 1),
                   top1=int((sim.argmax(dim=1) == torch.arange(len(sim), device=sim.device)).sum().item()), n_whole=len(sim))
    if joint is not None:
        out.update(oct=parts["oct"] / max(count, 1), slo=parts["slo"] / max(count, 1), oct_own_slo=parts["oct_paired"] / max(pairs, 1), oct_other_slo=parts["oct_other_slo"] / max(pairs, 1))
        if own_step:
            out.update({k: parts[k] / max(pairs, 1) for k in ("oct_clean_own", "oct_clean_other", "oct_slo_noise")})
    return out


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    aligned = args.slo_align > 0
    conditioned = args.slo_size > 0 and not aligned

    if args.slo_align < 0:
        raise SystemExit("--slo-align is a loss weight, 0 or above")
    if aligned and not args.slo_size:
        raise SystemExit("--slo-align needs --slo-size")
    if args.slo_align_loss != "contrastive" and not aligned:
        raise SystemExit("--slo-align-loss needs --slo-align")
    if args.slo_sigreg and args.slo_align_loss != "jepa":
        raise SystemExit("--slo-sigreg needs --slo-align-loss jepa")
    if not 0 <= args.slo_sigreg < 1:
        raise SystemExit("--slo-sigreg takes a weight from 0 to below 1")
    if aligned and (args.slo_adapter or args.slo_xattn or args.slo_slice):
        raise SystemExit("--slo-align keeps the SLO out of the denoiser; leave out --slo-adapter, --slo-xattn and --slo-slice")

    if not 0 <= args.band_scale <= 1:
        raise SystemExit("--band-scale takes a power from 0 to 1")
    if args.slo_adapter and not conditioned:
        raise SystemExit("--slo-adapter needs --slo-size")
    if args.slo_xattn and not conditioned:
        raise SystemExit("--slo-xattn needs --slo-size")
    if args.slo_slice and conditioned:
        raise SystemExit("--slo-slice is a joint model without the SLO encoder; leave out --slo-size")
    if args.slo_own_step and not args.slo_slice:
        raise SystemExit("--slo-own-step needs --slo-slice")
    if not (0 <= args.slo_clean and 0 <= args.slo_same and args.slo_clean + args.slo_same <= 1):
        raise SystemExit("--slo-clean and --slo-same are shares from 0 to 1 that add up to at most 1")
    if args.slo_slice and (args.keep_slices or args.n_slices) % 2:
        raise SystemExit("--slo-slice needs an even number of B-scans; use --keep-slices 24")
    if args.slo_slice and (args.keep_slices or args.n_slices) + 2 > args.target[0]:
        raise SystemExit("--slo-slice needs two free slices in --target after the B-scans")
    if not 0 <= args.keep_slices <= args.n_slices:
        raise SystemExit("--keep-slices must be from 0 to --n-slices")
    if (args.keep_slices or args.n_slices) > args.target[0]:
        raise SystemExit("--target has %d slices, fewer than the %d B-scans kept" % (args.target[0], args.keep_slices or args.n_slices))
    latest = os.path.join(args.out, "latest.pt")
    resume = args.resume or (args.auto_resume and os.path.exists(latest))
    if os.path.exists(latest) and not resume:
        raise SystemExit("%s already has a checkpoint. Use --resume or a new --out." % args.out)
    os.makedirs(args.out, exist_ok=True)
    ckpt = None
    if resume:
        if not os.path.exists(latest):
            raise SystemExit("--resume was given but %s has no latest.pt" % args.out)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        power = ckpt.get("band_power", 0.0 if ckpt.get("band_scale") is None else 1.0)
        if ckpt["design"] != args.design or ckpt.get("slo_size", 0) != args.slo_size or power != args.band_scale:
            raise SystemExit("latest.pt was trained with design %s, SLO size %d, band scaling %s" % (ckpt["design"], ckpt.get("slo_size", 0), "on, power %g" % power if power else "off"))
        saved = options(ckpt)
        if (saved["adapter"], saved["keep"], saved["z_down"]) != (args.slo_adapter, args.keep_slices, tuple(args.z_down)):
            raise SystemExit("latest.pt was trained with SLO adapter %s, slices kept %s, slice-axis halving %s" % ("on" if saved["adapter"] else "off", saved["keep"] or "all", " ".join(map(str, saved["z_down"]))))
        if (saved["tokens"], saved["dim"], tuple(sorted(saved["xattn"])), saved["slo_slice"]) != (args.slo_tokens, args.slo_dim, tuple(sorted(set(args.slo_xattn))), args.slo_slice):
            raise SystemExit("latest.pt was trained with SLO tokens %d, token size %d, cross-attention levels %s, SLO slice %s" % (saved["tokens"], saved["dim"], " ".join(map(str, saved["xattn"])) or "none", saved["slo_slice"] or "none"))
        if saved["align"] != args.slo_align or (aligned and (saved["align_loss"], saved["sigreg"]) != (args.slo_align_loss, args.slo_sigreg)):
            raise SystemExit("latest.pt was trained with SLO alignment weight %g%s" % (saved["align"], ", loss %s, SIGReg weight %g" % (saved["align_loss"], saved["sigreg"]) if saved["align"] else ""))
        if saved["own_step"] != args.slo_own_step:
            raise SystemExit("latest.pt was trained %s an own noise step for the SLO slice" % ("with" if saved["own_step"] else "without"))

    axes = wv.DESIGNS[args.design]
    n_ch = wv.channels(axes)
    try:
        grid = wv.check_unet_grid(args.target, axes, args.z_down)
    except ValueError as e:
        raise SystemExit(str(e))
    slo_text = "not used"
    if aligned:
        slo_text = "%d px, own encoder aligned with the OCT encoder by a %s loss (weight %g), not fed to the denoiser" % (args.slo_size, "contrastive" if args.slo_align_loss == "contrastive" else "JEPA-style latent prediction" + (" with SIGReg %g" % args.slo_sigreg if args.slo_sigreg else ""), args.slo_align)
    elif conditioned:
        slo_text = "%d px" % args.slo_size + (", %d tokens of %d" % (args.slo_tokens ** 2, args.slo_dim) if (args.slo_tokens, args.slo_dim) != (8, CONTEXT_DIM) else "") + (", adapter" if args.slo_adapter else "") + (", cross-attention at levels %s" % " ".join(map(str, sorted(set(args.slo_xattn)))) if args.slo_xattn else "")
    elif args.slo_slice:
        slo_text = "joint, own slice after the B-scans (%s)" % args.slo_slice
        if args.slo_own_step:
            slo_text += ", own noise step for the SLO (clean %g, same step %g, independent %g)" % (args.slo_clean, args.slo_same, 1 - args.slo_clean - args.slo_same)
    print("design %s: %d channels, target %s, wavelet grid %s, SLO %s, device %s" % (args.design, n_ch, tuple(args.target), grid, slo_text, device))
    if args.keep_slices or tuple(args.z_down) != Z_DOWN:
        print("B-scans kept: %s of %d; slice axis halved at %d of 4 downsamplings (%s)" % (args.keep_slices or "all", args.n_slices, sum(args.z_down), " ".join(map(str, args.z_down))))
    if use_amp:
        print("GPU: %s" % torch.cuda.get_device_name(0))

    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("cohort:", counts)
    train_scans, val_scans, test_scans = co.split_by_patient(scans, args.val_fraction, args.test_fraction, args.seed if args.split_seed is None else args.split_seed)
    groups = [set(s["patient"] for s in part) for part in (train_scans, val_scans, test_scans)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise SystemExit("patient overlap between train, validation and test")
    print("train scans: %d   validation scans: %d   test scans (never used here): %d" % (len(train_scans), len(val_scans), len(test_scans)))

    split_path = os.path.join(args.out, "split.json")
    split = {"train": sorted(groups[0]), "val": sorted(groups[1])}
    if test_scans:
        split["test"] = sorted(groups[2])
    if os.path.exists(split_path):
        with open(split_path) as f:
            if json.load(f) != split:
                raise SystemExit("split differs from split.json in --out")
    else:
        with open(split_path, "w") as f:
            json.dump(split, f)
    band, rms = None, None
    if args.band_scale:
        if ckpt is not None:
            rms, source = torch.tensor(ckpt.get("band_rms") or ckpt["band_scale"], dtype=torch.float32), "from latest.pt"
        else:
            rms, n = measure_bands(train_scans, args, axes, device)
            source = "measured on %d training scans" % n
        band = rms.pow(args.band_scale)
        print("band scaling %s, power %g: root mean square by number of detail axes %s; smallest band %.4f, largest %.4f" % (
            source, args.band_scale, ", ".join("%d: %.4f" % kv for kv in enumerate(wv.by_detail_axes(rms, len(axes)))), rms.min().item(), rms.max().item()))
    record = dict(vars(args), counts=counts, git=git_hash(), band_rms=None if rms is None else rms.tolist())
    if ckpt is None:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(record, f, indent=1)
    else:
        with open(os.path.join(args.out, "resumes.jsonl"), "a") as f:
            f.write(json.dumps(dict(record, after_epoch=ckpt["epoch"])) + "\n")

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
    data = dict(adapter=args.slo_adapter or aligned, keep=args.keep_slices, slo_slice=args.slo_slice)
    joint = None
    if args.slo_slice:
        n_oct = (args.keep_slices or args.n_slices) // 2 ** sum(1 for a in axes if a == wv.Z)
        joint = ((0, n_oct), (n_oct, n_oct + 1))
    train_loader = torch.utils.data.DataLoader(OctVolumes(train_scans, args.target, args.slo_size, **data), shuffle=True, drop_last=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(OctVolumes(val_scans, args.target, args.slo_size, other=conditioned, **data), shuffle=False, **loader_args)

    model = build_model(n_ch, args.slo_size, args.slo_adapter, tuple(args.z_down), args.slo_tokens, args.slo_dim, sorted(set(args.slo_xattn)), joint[1] if args.slo_own_step else None, aligned, args.slo_align_loss, args.slo_sigreg).to(device)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")

    start_epoch = 0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        print("resumed after epoch %d" % start_epoch)
        torch.manual_seed(args.seed + start_epoch)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.time()
        total, total_align, total_reg, steps = 0.0, 0.0, 0.0, 0
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            vol, slo, _ = unpack(batch, device, conditioned or aligned)
            x0 = wv.scaled(wv.forward(vol, axes), band)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],), device=device)
            noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
            weight = None
            if args.slo_own_step:
                s0, s1 = joint[1]
                t_slo = slo_steps(t, args.slo_clean, args.slo_same)
                noisy = with_slo(noisy, x0[:, :, s0:s1], noise[:, :, s0:s1], t_slo, sched, s0, s1)
                weight = torch.ones((x0.shape[0], 1) + x0.shape[2:], device=device)
                weight[:, :, s0:s1] = (t_slo >= 0).float()[:, None, None, None, None]
            with torch.amp.autocast("cuda", enabled=use_amp):
                if conditioned:
                    keep = torch.rand(x0.shape[0], device=device) >= args.slo_drop
                    pred = model(noisy, t, slo, keep)
                elif args.slo_own_step:
                    pred = model(noisy, t, t_slo)
                else:
                    pred = model(x=noisy, timesteps=t)
                if weight is None:
                    loss = torch.nn.functional.mse_loss(pred.float(), x0)
                else:
                    loss = ((pred.float() - x0).pow(2) * weight).sum() / (weight.sum() * x0.shape[1])
                mse = loss
                if aligned:
                    if args.slo_align_loss == "jepa":
                        dense, whole, reg = model.terms(x0, slo)
                        contrast = (1 - args.slo_sigreg) * (dense + whole) / 2 + args.slo_sigreg * reg
                    else:
                        dense, whole = align_loss(*model.embed(x0, slo))
                        contrast = (dense + whole) / 2
                    loss = mse + args.slo_align * contrast
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss at epoch %d step %d" % (epoch, step))
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if aligned and args.slo_align_loss == "jepa":
                    model.update_target()
            total += mse.item()
            if aligned:
                total_align += contrast.item()
                if args.slo_sigreg:
                    total_reg += reg.item()
            steps += 1
        if steps < args.accum:
            raise SystemExit("no optimizer step ran: fewer batches than --accum")
        val = validate(model, val_loader, axes, sched, device, use_amp, args.max_steps, conditioned, band, joint, args.slo_own_step, aligned)
        peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else float("nan")
        if conditioned:
            shown = "val mse %.6f (own SLO)  other SLO %.6f  no SLO %.6f" % (val["own"], val["other"], val["none"])
        elif joint is not None:
            shown = "val mse %.6f (OCT %.6f, SLO %.6f)  OCT with own SLO slice %.6f  with another scan's %.6f" % (val["none"], val["oct"], val["slo"], val["oct_own_slo"], val["oct_other_slo"])
            if args.slo_own_step:
                shown += "  OCT given a clean SLO: own %.6f  another scan's %.6f  SLO all noise %.6f" % (val["oct_clean_own"], val["oct_clean_other"], val["oct_slo_noise"])
        else:
            shown = "val mse %.6f" % val["none"]
        if aligned:
            shown = "align loss %.4f%s  %s  same place: own SLO %.3f  another scan's %.3f  whole scan: own %.3f  others %.3f  own SLO ranked first %d of %d" % (
                total_align / steps, " (SIGReg %.3f)" % (total_reg / steps) if args.slo_sigreg else "", shown, val["near_own"], val["near_other"], val["whole_own"], val["whole_other"], val["top1"], val["n_whole"])
        print("epoch %d: train mse %.6f  %s  %.0f s  peak %.2f GiB  steps %d" % (epoch, total / steps, shown, time.time() - started, peak, steps), flush=True)

        path = os.path.join(args.out, "epoch_%04d.pt" % epoch)
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "design": args.design, "target": args.target, "slo_size": args.slo_size, "band_scale": None if band is None else band.tolist(), "band_rms": None if rms is None else rms.tolist(), "band_power": args.band_scale, "slo_adapter": args.slo_adapter, "keep_slices": args.keep_slices, "z_down": list(args.z_down), "slo_tokens": args.slo_tokens, "slo_dim": args.slo_dim, "slo_xattn": sorted(set(args.slo_xattn)), "slo_slice": args.slo_slice, "slo_own_step": args.slo_own_step, "slo_clean": args.slo_clean, "slo_same": args.slo_same, "slo_align": args.slo_align, "slo_align_loss": args.slo_align_loss, "slo_sigreg": args.slo_sigreg}
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
        torch.save(state, latest + ".tmp")
        os.replace(latest + ".tmp", latest)
        old = os.path.join(args.out, "epoch_%04d.pt" % (epoch - args.keep))
        if os.path.exists(old):
            os.remove(old)
    print("done")


if __name__ == "__main__":
    main()
