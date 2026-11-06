#!/usr/bin/env python3
import argparse
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import wavelet as wv
from oct_wd.model import build_unet

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, nargs=3, default=[32, 512, 512])
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--max-batch", type=int, default=32)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
if use_amp:
    print("GPU: %s, %.1f GiB" % (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30))
else:
    print("no GPU: memory numbers are not measured")

for design, batches in (("w2z1", (1, 2, 4, 8, 16, 32)), ("w1z1", (1, 2, 4, 8))):
    axes = wv.DESIGNS[design]
    ch = wv.channels(axes)
    g = wv.check_unet_grid(args.target, axes)
    print("%s: %d channels, wavelet grid %s" % (design, ch, g))
    for bs in (b for b in batches if b <= args.max_batch):
        model = opt = scaler = x = pred = loss = None
        try:
            if use_amp:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = build_unet(ch).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            x = wv.forward(torch.rand(bs, 1, *args.target, device=device) * 2 - 1, axes)
            t = torch.randint(0, 1000, (bs,), device=device)
            start = time.time()
            for _ in range(args.steps):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x=x, timesteps=t)
                    loss = torch.nn.functional.mse_loss(pred.float(), x)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            peak = torch.cuda.max_memory_allocated() / 2**30 if use_amp else float("nan")
            print("  batch %2d: OK, peak %.2f GiB, %.2f s/step" % (bs, peak, (time.time() - start) / args.steps))
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print("  batch %2d: out of memory" % bs)
            break
        finally:
            del model, opt, scaler, x, pred, loss
