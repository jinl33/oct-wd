import os

import numpy as np
import torch
from PIL import Image


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


def depth_shift(p, q, max_shift=40):
    max_shift = min(max_shift, len(p) // 4)
    p = p - p.mean()
    q = q - q.mean()
    best, best_s = -np.inf, 0
    for s in sorted(range(-max_shift, max_shift + 1), key=abs):
        score = float(np.dot(p[max_shift:-max_shift], np.roll(q, s)[max_shift:-max_shift]))
        if score > best:
            best, best_s = score, s
    return best_s


class OctVolumes(torch.utils.data.Dataset):
    def __init__(self, scans, target):
        self.scans = scans
        self.target = tuple(target)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        s = self.scans[i]
        try:
            x = pad(scale(load_volume(s["folder"], s["n"])), self.target)
        except Exception as e:
            raise RuntimeError("volume index %d failed: %s" % (i, type(e).__name__)) from None
        return torch.from_numpy(x)[None]
