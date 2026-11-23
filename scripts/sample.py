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
from oct_wd.model import band_scale, joint_parts, load_checkpoint, options, predict, to_device
from oct_wd.volumes import as_batch, assemble, load_condition, load_volume, oct_region, other_index, raster_row, scale, slo_region, with_other_slo

EVAL_STEPS = (50, 200, 500, 800, 950)


def to_uint8(x):
    return ((np.clip(x, -1, 1) + 1) / 2 * 255).astype(np.uint8)


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
    across = np.mean([float(np.abs(np.diff(v, axis=0)).mean()) for v in volumes])
    within = np.mean([float(np.abs(np.diff(v, axis=2)).mean()) for v in volumes])
    print("  %-10s mean %.3f  std %.3f  p50 %.3f  p99 %.3f  black share %.3f  slice-to-slice change %.3f  within-slice change %.3f" % (label, a.mean(), a.std(), np.percentile(a, 50), np.percentile(a, 99), (a <= -0.99).mean(), across, within))


def band_table(real, made, n_axes, alphas):
    r = torch.cat(real).float().pow(2).mean(dim=(0, 2, 3, 4))
    g = torch.cat(made).float().pow(2).mean(dim=(0, 2, 3, 4))
    print("wavelet bands by number of detail axes (root mean square of coefficients):")
    print("  %-14s %-6s %-10s %-10s %s" % ("detail axes", "bands", "real", "generated", "generated / real"))
    for k in range(n_axes + 1):
        idx = [c for c in range(len(r)) if bin(c).count("1") == k]
        rr, gg = r[idx].mean().sqrt().item(), g[idx].mean().sqrt().item()
        print("  %-14d %-6d %-10.4f %-10.4f %.3f" % (k, len(idx), rr, gg, gg / max(rr, 1e-12)))
    print("  noise added at step 0, 10, 50: %s" % ", ".join("%.3f" % (1 - alphas[t]).sqrt().item() for t in (0, 10, 50)))


def pair_score(vol, slo):
    v = to_uint8(vol).astype(np.float32)
    rows = [raster_row(k, v.shape[0], slo.shape[0]) for k in range(v.shape[0])]
    return al.corr(al.highpass(al.en_face(v)), al.highpass(slo[rows].astype(np.float32)))


def correspondence(label, vols, slos):
    n = len(vols)
    own = [pair_score(vols[i], slos[i]) for i in range(n)]
    other = [pair_score(vols[i], slos[other_index(i, n)]) for i in range(n)] if n > 1 else [float("nan")]
    print("  %-10s own SLO %.3f (min %.3f, max %.3f)   another scan's SLO %.3f" % (label, np.mean(own), np.min(own), np.max(own), np.mean(other)))


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
    model, ckpt, axes = load_checkpoint(os.path.join(args.run, "latest.pt"), device)
    with open(os.path.join(args.run, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.run, "split.json")) as f:
        val_patients = set(json.load(f)["val"])
    target = tuple(ckpt["target"])
    saved = options(ckpt)
    grid = wv.check_unet_grid(target, axes, saved["z_down"])
    slo_size = ckpt.get("slo_size", 0)
    band = band_scale(ckpt)
    model.eval()
    joint = joint_parts(ckpt, axes)
    slo_text = "not used"
    if slo_size:
        slo_text = "%d px" % slo_size + (", adapter" if saved["adapter"] else "") + (", cross-attention at levels %s" % " ".join(map(str, saved["xattn"])) if saved["xattn"] else "")
    elif joint:
        slo_text = "joint slice (%s)" % saved["slo_slice"]
    print("checkpoint epoch %d, design %s, SLO %s, band scaling %s, device %s" % (ckpt["epoch"], ckpt["design"], slo_text, "on" if band is not None else "off", device))

    scans, _ = co.list_scans(args.root or co.default_root(), config["n_slices"])
    val = [s for s in scans if s["patient"] in val_patients]
    if not val:
        raise SystemExit("no validation scans found")
    out = os.path.join(args.run, "samples")
    os.makedirs(out, exist_ok=True)

    real, clean, slos, real_slo = [], [], [], []
    chosen = val[: max(args.n_eval, args.n_samples)]
    for s in chosen:
        vol = load_volume(s["folder"], s["n"], saved["keep"])
        real.append(vol)
        _, full = assemble(s["folder"], s["n"], target, saved["keep"], saved["slo_slice"])
        if joint:
            real_slo.append(slo_region(full, vol.shape))
        clean.append(wv.scaled(wv.forward(torch.from_numpy(full)[None, None], axes), band))
        if slo_size:
            slos.append(as_batch(load_condition(s["folder"], s["n"], slo_size, saved["adapter"], vol.shape, target, saved["keep"])))
    n_samples = min(args.n_samples, len(chosen))
    conditions = [("own SLO", lambda i: slos[i]), ("other SLO", lambda i: with_other_slo(slos[i], slos[other_index(i, len(slos))])), ("no SLO", lambda i: None)] if slo_size else [("model", lambda i: None)]
    native = real[0].shape
    average = torch.cat(clean).mean(dim=0, keepdim=True)
    ddim = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample", clip_sample=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    print("denoising on %d validation scans (lower is better; ratio 1 means no better than the average scan):" % len(clean))
    print("  %-8s %-14s %s" % ("noise t", "average MSE", "   ".join("ratio, %s" % name for name, _ in conditions)))
    with torch.no_grad():
        for t_value in EVAL_STEPS:
            errs, avg_err = [0.0] * len(conditions), 0.0
            for i, x0 in enumerate(clean):
                noise = torch.randn(x0.shape, generator=gen).to(device)
                t = torch.full((1,), t_value, device=device, dtype=torch.long)
                noisy = ddim.add_noise(original_samples=x0.to(device), noise=noise, timesteps=t)
                for k, (_, pick) in enumerate(conditions):
                    slo = pick(i)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        pred = predict(model, noisy, t, to_device(slo, device))
                    errs[k] += torch.nn.functional.mse_loss(pred.float().cpu(), x0).item()
                avg_err += torch.nn.functional.mse_loss(average, x0).item()
            print("  %-8d %-14.6f %s" % (t_value, avg_err / len(clean), "   ".join("%-14.3f" % (e / max(avg_err, 1e-12)) for e in errs)), flush=True)

        ddim.set_timesteps(args.ddim_steps, device=device)
        made, made_coefs, made_slo = [], [], []
        for i in range(n_samples):
            x = torch.randn((1, wv.channels(axes)) + grid, generator=gen).to(device)
            for t_value in ddim.timesteps:
                t = torch.full((1,), int(t_value), device=device, dtype=torch.long)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = predict(model, x, t, to_device(slos[i], device) if slo_size else None)
                x, _ = ddim.step(wv.unscaled(pred.float(), band), int(t_value), wv.unscaled(x, band))
                x = wv.scaled(x, band)
            made_coefs.append(wv.unscaled(x, band).cpu())
            pixels = wv.inverse(made_coefs[-1], axes)[0, 0].clamp(-1, 1).numpy()
            made.append(oct_region(pixels, native, joint is not None))
            save_view(to_uint8(made[-1]), os.path.join(out, "sample_%03d.png" % i))
            if joint:
                made_slo.append(slo_region(pixels, native))
                Image.fromarray(to_uint8(made_slo[-1])).save(os.path.join(out, "sample_%03d_slo.png" % i))
            print("generated %d of %d%s" % (i + 1, n_samples, " from the SLO of real_%03d" % i if slo_size else ""), flush=True)

    for i, vol in enumerate(real[:n_samples]):
        save_view(vol, os.path.join(out, "real_%03d.png" % i))
        if joint:
            Image.fromarray(to_uint8(real_slo[i])).save(os.path.join(out, "real_%03d_slo.png" % i))
    print("intensity on the -1 to 1 scale:")
    describe("real", [scale(v) for v in real[:n_samples]])
    describe("generated", made)
    lo, hi = joint[0] if joint else (0, None)
    band_table([wv.unscaled(c, band)[:, :, lo:hi] for c in clean[:n_samples]], [c[:, :, lo:hi] for c in made_coefs], len(axes), ddim.alphas_cumprod.cpu())
    if joint:
        print("SLO slice, intensity: real mean %.3f std %.3f   generated mean %.3f std %.3f" % (
            np.mean([a.mean() for a in real_slo[:n_samples]]), np.mean([a.std() for a in real_slo[:n_samples]]), np.mean([a.mean() for a in made_slo]), np.mean([a.std() for a in made_slo])))
        print("OCT en-face view against the SLO along the scan lines (correlation after high-pass, as in the alignment check):")
        correspondence("real", [scale(v) for v in real[:n_samples]], real_slo[:n_samples])
        correspondence("generated", made, made_slo)
    print("images written:", (4 if joint else 2) * len(made))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
