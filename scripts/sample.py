import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generative.networks.schedulers import DDIMScheduler

from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd import wavelet as wv
from oct_wd.model import build_unet
from oct_wd.volumes import load_volume, pad, scale

EVAL_STEPS = (50, 200, 500, 800, 950)


def to_uint8(x):
    return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)


def unpad(x, native):
    start = [(t - s) // 2 for s, t in zip(native, x.shape)]
    return x[tuple(slice(b, b + s) for b, s in zip(start, native))]


def stretch(a):
    low, high = np.percentile(a, [2, 98])
    return (np.clip((a - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def panel(a, size):
    return Image.fromarray(a).resize(size, Image.NEAREST).convert("RGB")


def save_view(vol, path):
    n, depth, width = vol.shape
    panels = [
        panel(vol[n // 2], (width, depth)),
        panel(vol[:, :, width // 2].T, (n * 8, depth)),
        panel(stretch(uniform_filter1d(al.en_face(vol), 9, axis=1)), (width, depth)),
    ]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), depth))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 10
    canvas.save(path)


def describe(label, volumes):
    a = np.concatenate([v.ravel()[::7] for v in volumes])
    bands = [float(np.abs(np.diff(v, axis=0)).mean()) for v in volumes]
    print("  %-10s mean %.3f  std %.3f  p50 %.3f  p99 %.3f  black share %.3f  slice-to-slice change %.3f" % (label, a.mean(), a.std(), np.percentile(a, 50), np.percentile(a, 99), (a <= -0.99).mean(), np.mean(bands)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--ddim-steps", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ckpt = torch.load(os.path.join(args.run, "latest.pt"), map_location=device, weights_only=False)
    with open(os.path.join(args.run, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.run, "split.json")) as f:
        val_patients = set(json.load(f)["val"])
    axes = wv.DESIGNS[ckpt["design"]]
    target = tuple(ckpt["target"])
    grid = wv.check_unet_grid(target, axes)
    model = build_unet(wv.channels(axes)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("checkpoint epoch %d, design %s, device %s" % (ckpt["epoch"], ckpt["design"], device))

    scans, _ = co.list_scans(args.root or co.default_root(), config["n_slices"])
    val = [s for s in scans if s["patient"] in val_patients]
    if not val:
        raise SystemExit("no validation scans found")
    out = os.path.join(args.run, "samples")
    os.makedirs(out, exist_ok=True)

    real, clean = [], []
    for s in val[: max(args.n_eval, args.n_samples)]:
        vol = load_volume(s["folder"], s["n"])
        real.append(vol)
        clean.append(wv.forward(torch.from_numpy(pad(scale(vol), target))[None, None], axes))
    native = real[0].shape
    average = torch.cat(clean).mean(dim=0, keepdim=True)
    ddim = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample", clip_sample=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    print("denoising on %d validation scans (lower is better; ratio 1 means no better than the average scan):" % len(clean))
    print("  %-8s %-12s %-14s %s" % ("noise t", "model MSE", "average MSE", "ratio"))
    with torch.no_grad():
        for t_value in EVAL_STEPS:
            model_err, avg_err = 0.0, 0.0
            for x0 in clean:
                noise = torch.randn(x0.shape, generator=gen).to(device)
                t = torch.full((1,), t_value, device=device, dtype=torch.long)
                noisy = ddim.add_noise(original_samples=x0.to(device), noise=noise, timesteps=t)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x=noisy, timesteps=t)
                model_err += torch.nn.functional.mse_loss(pred.float().cpu(), x0).item()
                avg_err += torch.nn.functional.mse_loss(average, x0).item()
            print("  %-8d %-12.6f %-14.6f %.3f" % (t_value, model_err / len(clean), avg_err / len(clean), model_err / max(avg_err, 1e-12)), flush=True)

        ddim.set_timesteps(args.ddim_steps, device=device)
        made = []
        for i in range(args.n_samples):
            x = torch.randn((1, wv.channels(axes)) + grid, generator=gen).to(device)
            for t_value in ddim.timesteps:
                t = torch.full((1,), int(t_value), device=device, dtype=torch.long)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x=x, timesteps=t)
                x, _ = ddim.step(pred.float(), int(t_value), x)
            pixels = wv.inverse(x.cpu(), axes)[0, 0].clamp(-1, 1).numpy()
            made.append(unpad(pixels, native))
            save_view(to_uint8(made[-1]), os.path.join(out, "sample_%03d.png" % i))
            print("generated %d of %d" % (i + 1, args.n_samples), flush=True)

    for i, vol in enumerate(real[: args.n_samples]):
        save_view(vol, os.path.join(out, "real_%03d.png" % i))
    print("intensity on the -1 to 1 scale:")
    describe("real", [scale(v) for v in real[: args.n_samples]])
    describe("generated", made)
    print("images written:", 2 * len(made))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
