import argparse
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import cohort as co
from oct_wd.volumes import load_volume, neighbor_steps, split_steps

STRIP = 400
GAP = 10
HALF_BAND = 4


def make_view(vol, out_path):
    n, depth, width = vol.shape
    strip = n * max(1, STRIP // n)
    canvas = Image.new("L", (3 * strip + 2 * GAP, depth))
    for i, column in enumerate((width // 4, width // 2, 3 * width // 4)):
        band = vol[:, :, max(column - HALF_BAND, 0): column + HALF_BAND + 1].mean(axis=2)
        section = Image.fromarray(band.T.astype(np.uint8)).resize((strip, depth), Image.NEAREST)
        canvas.paste(section, (i * (strip + GAP), 0))
    canvas.save(out_path)


def bins(values):
    w = np.array(values)
    return "under 5 px %d, 5-9 %d, 10-19 %d, 20-39 %d, 40-79 %d, 80+ %d" % ((w < 5).sum(), ((w >= 5) & (w < 10)).sum(), ((w >= 10) & (w < 20)).sum(), ((w >= 20) & (w < 40)).sum(), ((w >= 40) & (w < 80)).sum(), (w >= 80).sum())


def measures(profile):
    steps = neighbor_steps(profile)
    slope, sudden = split_steps(steps)
    return np.abs(steps).max(), np.abs(slope).max(), sudden.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/oct-qc"))
    ap.add_argument("--n-views", type=int, default=12)
    ap.add_argument("--n-slices", type=int, default=25)
    ap.add_argument("--as-slices", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("counts:", counts)
    os.makedirs(args.out, exist_ok=True)
    chosen = random.Random(args.seed).sample(range(len(scans)), min(args.n_views, len(scans)))
    views = {i: rank for rank, i in enumerate(chosen)}
    results, thinned, lines = [], [], {}
    failed = 0
    for i, s in enumerate(scans):
        try:
            vol = load_volume(s["folder"], s["n"])
            profile = vol.mean(axis=2)
            results.append(measures(profile))
            line = "view %03d: largest step %d, largest smooth slope %d, largest sudden jump %d" % ((views.get(i, -1),) + results[-1])
            if s["n"] > args.as_slices:
                keep = np.round(np.linspace(0, s["n"] - 1, args.as_slices)).astype(int)
                thinned.append(measures(profile[keep]))
                line += "   as %d slices: %d, %d, %d" % ((args.as_slices,) + thinned[-1])
            if i in views:
                make_view(vol, os.path.join(args.out, "stack_%03d.png" % views[i]))
                lines[views[i]] = line
        except Exception as e:
            failed += 1
            print("volume index %d failed: %s" % (i, type(e).__name__))
    print("scans checked: %d   failed: %d   images written: %d" % (len(results), failed, len(lines)))
    for label, rows in (("", results), ("as %d slices, " % args.as_slices, thinned)):
        if rows:
            a = np.array(rows)
            print("%sscans by largest step: %s" % (label, bins(a[:, 0])))
            print("%sscans by largest smooth slope: %s" % (label, bins(a[:, 1])))
            print("%sscans by largest sudden jump: %s" % (label, bins(a[:, 2])))
    for rank in sorted(lines):
        print(lines[rank])


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
