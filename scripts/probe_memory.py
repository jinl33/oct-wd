import argparse
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import wavelet as wv
from oct_wd.model import build_model, predict

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, nargs=3, default=[32, 512, 512])
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--max-batch", type=int, default=32)
ap.add_argument("--designs", nargs="+", default=["w2z1", "w1z1"], choices=sorted(wv.DESIGNS))
ap.add_argument("--slo-size", type=int, default=0)
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
if use_amp:
    print("GPU: %s, %.1f GiB" % (torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30))
else:
    print("no GPU: memory numbers are not measured")

for design in args.designs:
    batches = (1, 2, 4, 8, 16, 32)
    axes = wv.DESIGNS[design]
    ch = wv.channels(axes)
    g = wv.check_unet_grid(args.target, axes)
    print("%s: %d channels, wavelet grid %s, SLO %s" % (design, ch, g, "%d px" % args.slo_size if args.slo_size else "not used"))
    for bs in (b for b in batches if b <= args.max_batch):
        model = opt = scaler = x = slo = pred = loss = None
        try:
            if use_amp:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = build_model(ch, args.slo_size).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            x = wv.forward(torch.rand(bs, 1, *args.target, device=device) * 2 - 1, axes)
            t = torch.randint(0, 1000, (bs,), device=device)
            slo = torch.rand(bs, args.slo_size, args.slo_size, device=device) * 2 - 1 if args.slo_size else None
            start = time.time()
            for _ in range(args.steps):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = predict(model, x, t, slo)
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
            del model, opt, scaler, x, slo, pred, loss
