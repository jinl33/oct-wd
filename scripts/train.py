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
from oct_wd.model import build_unet
from oct_wd.volumes import OctVolumes


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="w2z1", choices=sorted(wv.DESIGNS))
    ap.add_argument("--n-slices", type=int, default=25)
    ap.add_argument("--target", type=int, nargs=3, default=[32, 512, 512])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None, help="cap steps per epoch, for smoke tests")
    ap.add_argument("--keep", type=int, default=3, help="number of recent checkpoints to keep")
    ap.add_argument("--resume", action="store_true", help="continue from latest.pt in --out")
    return ap.parse_args()


def git_hash():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@torch.no_grad()
def validate(model, loader, axes, sched, device, use_amp, max_steps):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    total, count = 0.0, 0
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x0 = wv.forward(batch.to(device), axes)
        noise = torch.randn(x0.shape, generator=gen).to(device)
        t = torch.randint(0, 1000, (x0.shape[0],), generator=gen).to(device)
        noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x=noisy, timesteps=t)
        total += torch.nn.functional.mse_loss(pred.float(), x0, reduction="sum").item() / x0[0].numel()
        count += x0.shape[0]
    model.train()
    return total / max(count, 1)


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    latest = os.path.join(args.out, "latest.pt")
    if os.path.exists(latest) and not args.resume:
        raise SystemExit("%s already has a checkpoint. Use --resume or a new --out." % args.out)
    os.makedirs(args.out, exist_ok=True)

    axes = wv.DESIGNS[args.design]
    n_ch = wv.channels(axes)
    grid = wv.check_unet_grid(args.target, axes)
    print("design %s: %d channels, target %s, wavelet grid %s, device %s" % (args.design, n_ch, tuple(args.target), grid, device))
    if use_amp:
        print("GPU: %s" % torch.cuda.get_device_name(0))

    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("cohort:", counts)
    train_scans, val_scans = co.split_by_patient(scans, args.val_fraction, args.seed)
    if set(s["patient"] for s in train_scans) & set(s["patient"] for s in val_scans):
        raise SystemExit("patient overlap between train and validation")
    print("train scans: %d   validation scans: %d" % (len(train_scans), len(val_scans)))

    split_path = os.path.join(args.out, "split.json")
    split = {"train": sorted(s["patient"] for s in train_scans), "val": sorted(s["patient"] for s in val_scans)}
    if os.path.exists(split_path):
        with open(split_path) as f:
            if json.load(f) != split:
                raise SystemExit("split differs from split.json in --out")
    else:
        with open(split_path, "w") as f:
            json.dump(split, f)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), counts=counts, git=git_hash()), f, indent=1)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(OctVolumes(train_scans, args.target), shuffle=True, drop_last=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(OctVolumes(val_scans, args.target), shuffle=False, **loader_args)

    model = build_unet(n_ch).to(device)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")

    start_epoch = 0
    if args.resume:
        if not os.path.exists(latest):
            raise SystemExit("--resume was given but %s has no latest.pt" % args.out)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
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
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            x0 = wv.forward(batch.to(device, non_blocking=True), axes)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],), device=device)
            noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x=noisy, timesteps=t)
                loss = torch.nn.functional.mse_loss(pred.float(), x0)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss at epoch %d step %d" % (epoch, step))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            steps += 1
        if steps == 0:
            raise SystemExit("no training steps ran: fewer scans than one batch")
        val_mse = validate(model, val_loader, axes, sched, device, use_amp, args.max_steps)
        peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else float("nan")
        print("epoch %d: train mse %.6f  val mse %.6f  %.0f s  peak %.2f GiB  steps %d" % (epoch, total / steps, val_mse, time.time() - started, peak, steps), flush=True)

        path = os.path.join(args.out, "epoch_%04d.pt" % epoch)
        state = {"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "design": args.design, "target": args.target}
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
