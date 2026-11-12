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
from oct_wd.model import build_model, predict
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
    ap.add_argument("--band-scale", type=float, nargs="?", const=1.0, default=0.0, help="divide each wavelet band by its root mean square on the training scans raised to this power (1 if given alone, 0.5 for the square root) before noise is added, so fine detail is not buried under the noise; sampling and heads read the divisors from the checkpoint")
    ap.add_argument("--seed", type=int, default=42)
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
    vol, slo = batch[0], batch[1]
    other = batch[2].to(device) if len(batch) > 2 else None
    return vol.to(device, non_blocking=True), slo.to(device), other


@torch.no_grad()
def measure_bands(scans, args, axes, device):
    loader = torch.utils.data.DataLoader(OctVolumes(scans, args.target), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    total, values, n = torch.zeros(wv.channels(axes), dtype=torch.float64, device=device), 0, 0
    for step, vol in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            break
        x = wv.forward(vol.to(device), axes)
        total += x.pow(2).sum(dim=(0, 2, 3, 4)).double()
        values += x[:, 0].numel()
        n += x.shape[0]
    return (total / values).sqrt().float().clamp_min(1e-4).cpu(), n


@torch.no_grad()
def validate(model, loader, axes, sched, device, use_amp, max_steps, conditioned, band):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    names = ("own", "other", "none") if conditioned else ("none",)
    total, count = dict.fromkeys(names, 0.0), 0
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        vol, slo, other = unpack(batch, device, conditioned)
        x0 = wv.scaled(wv.forward(vol, axes), band)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        t = torch.randint(0, 1000, (x0.shape[0],), generator=gen).to(device)
        noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
        for name in names:
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = predict(model, noisy, t, {"own": slo, "other": other, "none": None}[name])
            total[name] += torch.nn.functional.mse_loss(pred.float(), x0, reduction="sum").item() / x0[0].numel()
        count += x0.shape[0]
    model.train()
    return {k: v / max(count, 1) for k, v in total.items()}


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    conditioned = args.slo_size > 0

    if not 0 <= args.band_scale <= 1:
        raise SystemExit("--band-scale takes a power from 0 to 1")
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

    axes = wv.DESIGNS[args.design]
    n_ch = wv.channels(axes)
    grid = wv.check_unet_grid(args.target, axes)
    print("design %s: %d channels, target %s, wavelet grid %s, SLO %s, device %s" % (args.design, n_ch, tuple(args.target), grid, "%d px" % args.slo_size if conditioned else "not used", device))
    if use_amp:
        print("GPU: %s" % torch.cuda.get_device_name(0))

    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("cohort:", counts)
    train_scans, val_scans, test_scans = co.split_by_patient(scans, args.val_fraction, args.test_fraction, args.seed)
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
    train_loader = torch.utils.data.DataLoader(OctVolumes(train_scans, args.target, args.slo_size), shuffle=True, drop_last=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(OctVolumes(val_scans, args.target, args.slo_size, other=conditioned), shuffle=False, **loader_args)

    model = build_model(n_ch, args.slo_size).to(device)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        total, steps = 0.0, 0
        if use_amp:
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            vol, slo, _ = unpack(batch, device, conditioned)
            x0 = wv.scaled(wv.forward(vol, axes), band)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],), device=device)
            noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if conditioned:
                    keep = torch.rand(x0.shape[0], device=device) >= args.slo_drop
                    pred = model(noisy, t, slo, keep)
                else:
                    pred = model(x=noisy, timesteps=t)
                loss = torch.nn.functional.mse_loss(pred.float(), x0)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss at epoch %d step %d" % (epoch, step))
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += loss.item()
            steps += 1
        if steps < args.accum:
            raise SystemExit("no optimizer step ran: fewer batches than --accum")
        val = validate(model, val_loader, axes, sched, device, use_amp, args.max_steps, conditioned, band)
        peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else float("nan")
        if conditioned:
            shown = "val mse %.6f (own SLO)  other SLO %.6f  no SLO %.6f" % (val["own"], val["other"], val["none"])
        else:
            shown = "val mse %.6f" % val["none"]
        print("epoch %d: train mse %.6f  %s  %.0f s  peak %.2f GiB  steps %d" % (epoch, total / steps, shown, time.time() - started, peak, steps), flush=True)

        path = os.path.join(args.out, "epoch_%04d.pt" % epoch)
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "design": args.design, "target": args.target, "slo_size": args.slo_size, "band_scale": None if band is None else band.tolist(), "band_rms": None if rms is None else rms.tolist(), "band_power": args.band_scale}
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
