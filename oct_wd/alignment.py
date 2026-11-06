import csv
import os

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d


def read_lines(folder):
    with open(os.path.join(folder, "metadata.csv"), newline="", encoding="utf-8-sig", errors="replace") as f:
        meta = list(csv.DictReader(f))
    return np.array([[float(m["pixel_start_row"]), float(m["pixel_start_col"]), float(m["pixel_end_row"]), float(m["pixel_end_col"])] for m in meta])


def read_slo(folder):
    with Image.open(os.path.join(folder, "slo.png")) as im:
        return np.asarray(im.convert("L"))


def en_face(vol, top_fraction=0.1):
    k = max(1, int(vol.shape[1] * top_fraction))
    view = np.partition(vol, -k, axis=1)[:, -k:, :].mean(axis=1).astype(np.float32)
    return view / np.maximum(np.median(view, axis=1, keepdims=True), 1e-6)


def sample_slo(slo, lines, width, dr=0, dc=0):
    t = np.linspace(0, 1, width)
    rows = lines[:, 0, None] + (lines[:, 2, None] - lines[:, 0, None]) * t + dr
    cols = lines[:, 1, None] + (lines[:, 3, None] - lines[:, 1, None]) * t + dc
    r = np.clip(np.rint(rows).astype(int), 0, slo.shape[0] - 1)
    c = np.clip(np.rint(cols).astype(int), 0, slo.shape[1] - 1)
    return slo[r, c].astype(np.float32)


def highpass(a, window=51):
    return a - uniform_filter1d(a, window, axis=1, mode="nearest")


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def alignment_scores(vol, slo, lines, max_offset=30, step=3):
    width = vol.shape[2]
    view = highpass(en_face(vol))
    base = highpass(sample_slo(slo, lines, width))
    scores = {
        "as_given": corr(view, base),
        "slices_reversed": corr(view[::-1], base),
        "mirrored": corr(view[:, ::-1], base),
        "both": corr(view[::-1, ::-1], base),
    }
    best = (-np.inf, 0, 0)
    offsets = sorted(range(-max_offset, max_offset + 1, step), key=abs)
    for dr in offsets:
        for dc in offsets:
            c = corr(view, highpass(sample_slo(slo, lines, width, dr, dc)))
            if c > best[0]:
                best = (c, dr, dc)
    return scores, best
