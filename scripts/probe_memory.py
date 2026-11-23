import argparse
import os
import sys
import time

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import wavelet as wv
from oct_wd.model import CONTEXT_DIM, Z_DOWN, build_model, predict
from oct_wd.volumes import slo_grid

ap = argparse.ArgumentParser()
ap.add_argument("--target", type=int, nargs=3, default=[32, 512, 512])
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--max-batch", type=int, default=32)
ap.add_argument("--designs", nargs="+", default=["w2z1", "w1z1"], choices=sorted(wv.DESIGNS))
ap.add_argument("--slo-size", type=int, default=0)
ap.add_argument("--slo-adapter", action="store_true")
ap.add_argument("--slo-tokens", type=int, default=8)
ap.add_argument("--slo-dim", type=int, default=CONTEXT_DIM)
ap.add_argument("--slo-xattn", type=int, nargs="*", default=[])
ap.add_argument("--z-down", type=int, nargs=4, default=list(Z_DOWN), choices=(0, 1))
args = ap.parse_args()
if (args.slo_adapter or args.slo_xattn) and not args.slo_size:
    raise SystemExit("--slo-adapter and --slo-xattn need --slo-size")

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
    g = wv.check_unet_grid(args.target, axes, args.z_down)
    print("%s: %d channels, wavelet grid %s, SLO %s, slice axis halved at %d of 4 downsamplings" % (design, ch, g, ("%d px" % args.slo_size + (", %d tokens of %d" % (args.slo_tokens ** 2, args.slo_dim) if (args.slo_tokens, args.slo_dim) != (8, CONTEXT_DIM) else "") + (", adapter" if args.slo_adapter else "") + (", cross-attention at levels %s" % " ".join(map(str, args.slo_xattn)) if args.slo_xattn else "")) if args.slo_size else "not used", sum(args.z_down)))
    for bs in (b for b in batches if b <= args.max_batch):
        model = opt = scaler = x = slo = pred = loss = None
        try:
            if use_amp:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model = build_model(ch, args.slo_size, args.slo_adapter, tuple(args.z_down), args.slo_tokens, args.slo_dim, sorted(set(args.slo_xattn))).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            x = wv.forward(torch.rand(bs, 1, *args.target, device=device) * 2 - 1, axes)
            t = torch.randint(0, 1000, (bs,), device=device)
            slo = torch.rand(bs, args.slo_size, args.slo_size, device=device) * 2 - 1 if args.slo_size else None
            if args.slo_adapter:
                n = args.target[0]
                lines = np.array([[r, 128.0, r, 640.0] for r in np.linspace(128, 640, n)])
                slo = (slo, torch.from_numpy(slo_grid(lines, (768, 768), tuple(args.target), tuple(args.target)))[None].expand(bs, -1, -1, -1).to(device))
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
