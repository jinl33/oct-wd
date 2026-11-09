import argparse
import gzip
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import cohort as co
from oct_wd.volumes import load_volume

POOL = 12


def nifti_header(shape, spacing):
    n, depth, width = shape
    lateral, axial, between = spacing
    h = bytearray(348)
    struct.pack_into("<i", h, 0, 348)
    struct.pack_into("<8h", h, 40, 3, width, depth, n, 1, 1, 1, 1)
    struct.pack_into("<2h", h, 70, 2, 8)
    struct.pack_into("<8f", h, 76, 1.0, lateral, axial, between, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<3f", h, 108, 352.0, 1.0, 0.0)
    struct.pack_into("<B", h, 123, 2)
    struct.pack_into("<2h", h, 252, 1, 1)
    struct.pack_into("<6f", h, 256, -np.sqrt(0.5), 0.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", h, 280, lateral, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", h, 296, 0.0, 0.0, between, 0.0)
    struct.pack_into("<4f", h, 312, 0.0, -axial, 0.0, 0.0)
    h[344:348] = b"n+1\x00"
    return bytes(h) + bytes(4)


def write_nifti(path, vol, spacing):
    with gzip.open(path, "wb", compresslevel=1) as f:
        f.write(nifti_header(vol.shape, spacing))
        f.write(np.ascontiguousarray(vol, dtype=np.uint8).tobytes())


def spacing(row, labels_as_given):
    names = ("bscanPixelspacingWidth", "bscanPixelspacingDepth") if labels_as_given else ("bscanPixelspacingDepth", "bscanPixelspacingWidth")
    values, missing = [], 0
    for name in names + ("sliceThickness",):
        try:
            v = float(row.get(name) or "nan")
        except ValueError:
            v = float("nan")
        if not (np.isfinite(v) and v > 0):
            v, missing = 1.0, missing + 1
        values.append(v)
    return tuple(values), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=os.path.expanduser("~/oct-qc/nifti"))
    ap.add_argument("--n-views", type=int, default=4)
    ap.add_argument("--n-slices", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--labels-as-given", action="store_true")
    args = ap.parse_args()
    scans, counts = co.list_scans(args.root or co.default_root(), args.n_slices)
    print("counts:", counts)
    os.makedirs(args.out, exist_ok=True)
    chosen = random.Random(args.seed).sample(range(len(scans)), min(POOL, len(scans)))[: args.n_views]
    for rank, i in enumerate(chosen):
        s = scans[i]
        name = "n%d_%03d.nii.gz" % (s["n"], rank)
        try:
            vol = load_volume(s["folder"], s["n"])
            (lateral, axial, between), missing = spacing(s["row"], args.labels_as_given)
            write_nifti(os.path.join(args.out, name), vol, (lateral, axial, between))
            n, depth, width = vol.shape
            print("%s: %d x %d x %d voxels; mm per voxel across %.5f, depth %.5f, between slices %.5f; scan width / scan length %.2f; spacing values missing %d" % (name, width, depth, n, lateral, axial, between, (width - 1) * lateral / ((n - 1) * between), missing))
        except Exception as e:
            print("%s failed: %s" % (name, type(e).__name__))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
