import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd.volumes import depth_shift, load_volume


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
        panel(stretch(al.en_face(vol)[order]), (width, depth)),
        panel(stretch(al.sample_slo(gray, lines, width)[order]), (width, depth)),
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
    shifts, worst, ranked, failed, written = [], [], [], 0, 0
    for i, s in enumerate(scans):
        try:
            vol = load_volume(s["folder"], s["n"])
            profile = vol.mean(axis=2)
            jumps = [abs(depth_shift(profile[k], profile[k + 1])) for k in range(len(vol) - 1)]
            shifts += jumps
            worst.append(max(jumps))
            ranked.append((max(jumps), i))
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
    print("largest jump in each qc_worst image (pixels):", [int(size) for size, _ in ranked])
    if shifts:
        a, w = np.array(shifts), np.array(worst)
        print("depth jump between neighboring slices (pixels): median %.1f  p90 %.1f  p99 %.1f  max %d" % (np.median(a), np.percentile(a, 90), np.percentile(a, 99), a.max()))
        print("scans by largest jump: under 5 px %d, 5-9 px %d, 10-19 px %d, 20+ px %d" % ((w < 5).sum(), ((w >= 5) & (w < 10)).sum(), ((w >= 10) & (w < 20)).sum(), (w >= 20).sum()))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
