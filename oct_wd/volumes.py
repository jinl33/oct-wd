import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import uniform_filter1d


def load_volume(folder, n):
    slices = []
    for i in range(1, n + 1):
        with Image.open(os.path.join(folder, "b%d.png" % i)) as im:
            a = np.asarray(im)
        if a.ndim != 2 or a.dtype != np.uint8:
            raise ValueError("expected 8-bit grayscale")
        slices.append(a)
    return np.stack(slices)


def scale(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    if p99 - p1 < 1e-6:
        raise ValueError("constant volume")
    return np.clip((vol.astype(np.float32) - p1) / (p99 - p1), 0, 1) * 2 - 1


def pad(x, target, value=-1.0):
    if any(s > t for s, t in zip(x.shape, target)):
        raise ValueError("volume %s is larger than target %s" % (x.shape, tuple(target)))
    widths = [((t - s) // 2, (t - s) - (t - s) // 2) for s, t in zip(x.shape, target)]
    return np.pad(x, widths, constant_values=value)


def depth_shift(p, q, max_shift=120):
    n = len(p)
    max_shift = min(max_shift, n // 4)
    best, best_s = -np.inf, 0
    for s in sorted(range(-max_shift, max_shift + 1), key=abs):
        a = p[s:] if s >= 0 else p[: n + s]
        b = q[: n - s] if s >= 0 else q[-s:]
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        score = float((a * b).sum() / denom) if denom > 0 else 0.0
        if score > best:
            best, best_s = score, s
    return best_s


def neighbor_steps(profile):
    return np.array([depth_shift(profile[k], profile[k + 1]) for k in range(len(profile) - 1)], dtype=np.float64)


def split_steps(steps, window=2):
    slope = np.array([np.median(steps[max(k - window, 0): k + window + 1]) for k in range(len(steps))])
    return slope, np.abs(steps - slope)


def layer_depths(vol, smooth=9):
    smoothed = uniform_filter1d(vol.astype(np.float32), smooth, axis=1)
    return np.median(smoothed.argmax(axis=1), axis=1)


def load_slo(folder, size):
    with Image.open(os.path.join(folder, "slo.png")) as im:
        a = np.asarray(im.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    low, high = np.percentile(a, [1, 99])
    if high - low < 1e-6:
        raise ValueError("constant SLO")
    return np.clip((a - low) / (high - low), 0, 1) * 2 - 1


def other_index(i, n):
    return (i + n // 2) % n


class OctVolumes(torch.utils.data.Dataset):
    def __init__(self, scans, target, slo_size=0, other=False):
        self.scans = scans
        self.target = tuple(target)
        self.slo_size = slo_size
        self.other = other

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        s = self.scans[i]
        try:
            x = torch.from_numpy(pad(scale(load_volume(s["folder"], s["n"])), self.target))[None]
            if not self.slo_size:
                return x
            item = (x, torch.from_numpy(load_slo(s["folder"], self.slo_size)))
            if self.other:
                j = other_index(i, len(self.scans))
                item += (torch.from_numpy(load_slo(self.scans[j]["folder"], self.slo_size)),)
        except Exception as e:
            raise RuntimeError("volume index %d failed: %s" % (i, type(e).__name__)) from None
        return item
