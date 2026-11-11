import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

MAIN = {"binary": "auroc", "regression": "mae"}


def spread(values):
    a = np.array(values, dtype=float)
    a = a[np.isfinite(a)]
    return (a.mean(), a.min(), a.max()) if len(a) else (float("nan"),) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    args = ap.parse_args()
    for run in args.run:
        groups = defaultdict(list)
        for path in sorted(glob.glob(os.path.join(run, "downstream", "*", "results.json"))):
            with open(path) as f:
                r = json.load(f)
            key = (r["task"], r["mode"], "random init" if r["random_init"] else "pretrained", r["t"], r.get("label_fraction", 1.0))
            metric = MAIN[r["kind"]]
            groups[key].append((r["test"][metric], r["val"], metric, r["splits"]["train"]["n"]))
        print("== %s: %d results" % (os.path.basename(os.path.normpath(run)), sum(len(v) for v in groups.values())))
        if not groups:
            continue
        print("  %-10s %-9s %-12s %-5s %-9s %-7s %-6s %-22s %s" % ("task", "mode", "encoder", "t", "fraction", "scans", "runs", "test mean (min to max)", "val mean"))
        for key in sorted(groups, key=lambda k: (k[0], k[1], k[3], -k[4], k[2])):
            rows = groups[key]
            test = spread([v[0] for v in rows])
            val = spread([v[1] for v in rows])
            print("  %-10s %-9s %-12s %-5d %-9g %-7d %-6d %.3f (%.3f to %.3f)    %.3f   %s" % (
                key[0], key[1], key[2], key[3], key[4], int(np.mean([v[3] for v in rows])), len(rows),
                test[0], test[1], test[2], val[0], rows[0][2]))


if __name__ == "__main__":
    main()
