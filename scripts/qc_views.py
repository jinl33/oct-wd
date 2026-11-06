import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd.volumes import depth_shift, layer_depths, load_volume


def panel(a, size):
    return Image.fromarray(a).resize(size, Image.NEAREST).convert("RGB")


def stretch(a):
    low, high = np.percentile(a, [2, 98])
    return (np.clip((a - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def make_view(vol, folder, out_path):
    n, depth, width = vol.shape
    gray = al.read_slo(folder)
    lines = al.read_lines(folder)
    slo = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(slo)
    for i, (start_row, start_col, end_row, end_col) in enumerate(lines):
        draw.line([start_col, start_row, end_col, end_row], fill=(255, 0, 0) if i == n // 2 else (0, 255, 0), width=1)
    order = np.argsort(lines[:, 0])
    panels = [
        slo.resize((depth, depth), Image.BILINEAR),
        panel(vol[n // 2], (width, depth)),
        panel(vol[:, :, width // 2].T, (n * 8, depth)),
        panel(stretch(uniform_filter1d(al.en_face(vol), 9, axis=1)[order]), (width, depth)),
        panel(stretch(uniform_filter1d(al.sample_slo(gray, lines, width), 9, axis=1)[order]), (width, depth)),
    ]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), depth))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 10
    canvas.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/oct-qc"))
    ap.add_argument("--n-views", type=int, default=12)
    ap.add_argument("--n-worst", type=int, default=12)
    ap.add_argument("--n-slices", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("counts:", counts)
    os.makedirs(args.out, exist_ok=True)
    chosen = set(random.Random(args.seed).sample(range(len(scans)), min(args.n_views, len(scans))))
    shifts, layer_shifts, worst, layer_worst, ranked, zero, top_zero, low, high, dark = [], [], [], [], [], [], [], [], [], []
    failed = written = near_edge = pairs = agree = 0
    for i, s in enumerate(scans):
        try:
            vol = load_volume(s["folder"], s["n"])
            profile = vol.mean(axis=2)
            jumps = np.array([abs(depth_shift(profile[k], profile[k + 1])) for k in range(len(vol) - 1)])
            layer = layer_depths(vol)
            layer_jumps = np.abs(np.diff(layer))
            shifts += jumps.tolist()
            layer_shifts += layer_jumps.tolist()
            worst.append(jumps.max())
            layer_worst.append(layer_jumps.max())
            ranked.append((int(jumps.max()), i))
            big = jumps >= 20
            pairs += int(big.sum())
            agree += int((np.abs(jumps[big] - layer_jumps[big]) <= 5).sum())
            zero.append(float((vol == 0).mean()))
            top_zero.append(float((vol[:, : vol.shape[1] // 10] == 0).mean()))
            p1, p99 = np.percentile(vol, [1, 99])
            low.append(p1)
            high.append(p99)
            brightness = vol.mean(axis=(1, 2))
            dark.append(float(brightness.min() / max(np.median(brightness), 1e-6)))
            near_edge += bool(layer.min() < 30 or layer.max() > vol.shape[1] - 30)
            if i in chosen:
                make_view(vol, s["folder"], os.path.join(args.out, "qc_%03d.png" % written))
                written += 1
        except Exception as e:
            failed += 1
            print("volume index %d failed: %s" % (i, type(e).__name__))
    print("scans checked: %d   failed: %d   QC images written: %d" % (len(worst), failed, written))
    ranked = sorted(ranked, reverse=True)[: args.n_worst]
    for rank, (_, i) in enumerate(ranked):
        s = scans[i]
        make_view(load_volume(s["folder"], s["n"]), s["folder"], os.path.join(args.out, "qc_worst_%03d.png" % rank))
    print("largest jump in each qc_worst image (pixels):", [size for size, _ in ranked])
    if not worst:
        return
    for label, values in (("profile match", shifts), ("bright layer", layer_shifts)):
        a = np.array(values)
        print("neighboring-slice jump, %s (pixels): median %.1f  p90 %.1f  p99 %.1f  max %d" % (label, np.median(a), np.percentile(a, 90), np.percentile(a, 99), a.max()))
    for label, values in (("profile match", worst), ("bright layer", layer_worst)):
        w = np.array(values)
        print("scans by largest jump, %s: under 5 px %d, 5-9 %d, 10-19 %d, 20-39 %d, 40-79 %d, 80+ %d" % (label, (w < 5).sum(), ((w >= 5) & (w < 10)).sum(), ((w >= 10) & (w < 20)).sum(), ((w >= 20) & (w < 40)).sum(), ((w >= 40) & (w < 80)).sum(), (w >= 80).sum()))
    print("slice pairs with a 20+ px profile jump: %d   bright layer agrees within 5 px: %d" % (pairs, agree))
    print("share of pixels exactly 0, per scan: median %.4f  p90 %.4f  max %.4f" % (np.median(zero), np.percentile(zero, 90), np.max(zero)))
    print("same, top tenth of the frame: median %.4f  p90 %.4f  max %.4f" % (np.median(top_zero), np.percentile(top_zero, 90), np.max(top_zero)))
    print("scans with over 5%% exact-0 pixels in the top tenth: %d" % (np.array(top_zero) > 0.05).sum())
    print("scaling range per scan: p1 median %.0f  max %.0f   p99 median %.0f  min %.0f" % (np.median(low), np.max(low), np.median(high), np.min(high)))
    print("scans with a slice under half their typical brightness: %d" % (np.array(dark) < 0.5).sum())
    print("scans with the bright layer within 30 px of the top or bottom: %d" % near_edge)


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
