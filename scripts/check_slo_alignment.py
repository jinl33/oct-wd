import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd.volumes import load_volume

LABELS = (("as_given", "as given"), ("slices_reversed", "slices reversed"), ("mirrored", "mirrored"), ("both", "reversed and mirrored"))


def spread(vals):
    return "median %.3f  p10 %.3f  p90 %.3f" % (np.median(vals), np.percentile(vals, 10), np.percentile(vals, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--n-slices", type=int, default=25)
    args = ap.parse_args()
    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("counts:", counts)
    scores = collections.defaultdict(list)
    chance, best_corr, shifts = [], [], []
    wins = collections.defaultdict(collections.Counter)
    failed = flat = 0
    previous = None
    for i, s in enumerate(scans):
        try:
            vol = load_volume(s["folder"], s["n"])
            slo = al.read_slo(s["folder"])
            lines = al.read_lines(s["folder"])
            result, best = al.alignment_scores(vol, slo, lines)
        except Exception as e:
            failed += 1
            print("volume index %d failed: %s" % (i, type(e).__name__))
            continue
        for key, value in result.items():
            scores[key].append(value)
        top = max(result, key=result.get)
        if result[top] == 0:
            flat += 1
        else:
            wins[s["eye"]][top] += 1
        best_corr.append(best[0])
        shifts.append(best[1:])
        if previous is not None:
            view = al.highpass(al.en_face(vol))
            chance.append(al.corr(view, al.highpass(al.sample_slo(previous[0], previous[1], vol.shape[2]))))
        previous = (slo, lines)
    print("scans checked: %d   failed: %d" % (len(best_corr), failed))
    if not best_corr:
        return
    print("correlation of the OCT top-down view with the SLO along the scan lines:")
    for key, label in LABELS:
        print("  %-24s %s" % (label, spread(scores[key])))
    if chance:
        print("  %-24s %s" % ("another scan's SLO", spread(chance)))
    print("scans with no usable pattern: %d" % flat)
    print("orientation that scored highest (scans):")
    for eye in sorted(wins):
        print("  eye %-8s %s" % (eye, ", ".join("%s %d" % (label, wins[eye][key]) for key, label in LABELS)))
    a = np.array(shifts)
    print("best shift of the scan lines, as given (SLO pixels, searched up to 30 in steps of 3):")
    print("  rows %s" % spread(a[:, 0]))
    print("  cols %s" % spread(a[:, 1]))
    print("  correlation at best shift %s" % spread(best_corr))
    largest = np.abs(a).max(axis=1)
    print("  scans within 6 px: %d   within 15 px: %d   at the search limit: %d   of %d" % ((largest <= 6).sum(), (largest <= 15).sum(), (largest >= 30).sum(), len(a)))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
