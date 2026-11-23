import os

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates, uniform_filter1d

from oct_wd import alignment as al


def first_slice(n, keep=0):
    if keep < 0 or keep > n:
        raise ValueError("cannot keep %d of %d B-scans" % (keep, n))
    return (n - keep) // 2 if keep else 0


def load_volume(folder, n, keep=0):
    start = first_slice(n, keep)
    slices = []
    for i in range(start + 1, start + (keep or n) + 1):
        with Image.open(os.path.join(folder, "b%d.png" % i)) as im:
            a = np.asarray(im)
        if a.ndim != 2 or a.dtype != np.uint8:
            raise ValueError("expected 8-bit grayscale")
        slices.append(a)
    return np.stack(slices)


def scale(vol):
    p1, p99 = (float(v) for v in np.percentile(vol, [1, 99]))
    if p99 - p1 < 1e-6:
        raise ValueError("constant volume")
    return (np.clip((vol.astype(np.float32) - p1) / (p99 - p1), 0, 1) * 2 - 1).astype(np.float32, copy=False)


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
    low, high = (float(v) for v in np.percentile(a, [1, 99]))
    if high - low < 1e-6:
        raise ValueError("constant SLO")
    return (np.clip((a - low) / (high - low), 0, 1) * 2 - 1).astype(np.float32, copy=False)


def slo_grid(lines, slo_shape, native, target):
    n, width = native[0], native[-1]
    zt, wt = target[0], target[-1]
    if n < 2 or len(lines) != n:
        raise ValueError("need one scan line per B-scan, at least 2")
    k = np.arange(zt) - (zt - n) // 2
    j = np.arange(wt) - (wt - width) // 2
    i = np.clip(k, 0, n - 2)
    p = lines[i] + (k - i)[:, None] * (lines[i + 1] - lines[i])
    u = (j / max(width - 1, 1))[None, :]
    rows = p[:, 0:1] + u * (p[:, 2:3] - p[:, 0:1])
    cols = p[:, 1:2] + u * (p[:, 3:4] - p[:, 1:2])
    valid = ((k >= 0) & (k < n))[:, None] & ((j >= 0) & (j < width))[None, :]
    return np.stack([(cols + 0.5) / slo_shape[1] * 2 - 1, (rows + 0.5) / slo_shape[0] * 2 - 1, valid]).astype(np.float32)


def load_slo_grid(folder, n, native, target, keep=0):
    lines = al.read_lines(folder)
    if len(lines) != n:
        raise ValueError("metadata has %d scan lines for %d B-scans" % (len(lines), n))
    start = first_slice(n, keep)
    with Image.open(os.path.join(folder, "slo.png")) as im:
        width, height = im.size
    return slo_grid(lines[start: start + native[0]], (height, width), native, target)


def raster_positions(lines, rows, cols):
    k = np.linspace(0, len(lines) - 1, rows)
    i = np.clip(np.floor(k).astype(int), 0, len(lines) - 2)
    p = lines[i] + (k - i)[:, None] * (lines[i + 1] - lines[i])
    u = np.linspace(0, 1, cols)[None, :]
    return p[:, 0:1] + u * (p[:, 2:3] - p[:, 0:1]), p[:, 1:2] + u * (p[:, 3:4] - p[:, 1:2])


def raster_row(k, n, rows):
    return int(round(k * (rows - 1) / max(n - 1, 1)))


def load_slo_slice(folder, n, shape, keep=0, region="raster"):
    with Image.open(os.path.join(folder, "slo.png")) as im:
        im = im.convert("L")
        if region == "full":
            return scale(np.asarray(im.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32))
        a = np.asarray(im, dtype=np.float32)
    if region != "raster":
        raise ValueError("unknown SLO slice region %s" % region)
    lines = al.read_lines(folder)
    if len(lines) != n:
        raise ValueError("metadata has %d scan lines for %d B-scans" % (len(lines), n))
    start = first_slice(n, keep)
    rows, cols = raster_positions(lines[start: start + (keep or n)], shape[0], shape[1])
    return scale(map_coordinates(a, [rows, cols], order=1, mode="nearest"))


def joint_volume(vol, slo, target):
    z = vol.shape[0]
    if z % 2 or z + 2 > target[0]:
        raise ValueError("the joint volume needs an even number of B-scans and two free slices for the SLO")
    out = np.full(tuple(target), -1.0, dtype=np.float32)
    out[:z] = pad(vol, (z,) + tuple(target[1:]))
    out[z: z + 2] = slo
    return out


def oct_region(x, native, joint):
    if not joint:
        start = [(t - s) // 2 for s, t in zip(native, x.shape)]
    else:
        start = [0] + [(t - s) // 2 for s, t in zip(native[1:], x.shape[1:])]
    return x[tuple(slice(b, b + s) for b, s in zip(start, native))]


def slo_region(x, native):
    return x[native[0]]


def assemble(folder, n, target, keep=0, slo_slice=None):
    vol = scale(load_volume(folder, n, keep))
    if not slo_slice:
        return vol, pad(vol, target)
    return vol, joint_volume(vol, load_slo_slice(folder, n, target[1:], keep, slo_slice), target)


def load_condition(folder, n, slo_size, adapter, native, target, keep=0):
    if not slo_size:
        return None
    slo = torch.from_numpy(load_slo(folder, slo_size))
    if not adapter:
        return slo
    return slo, torch.from_numpy(load_slo_grid(folder, n, native, target, keep))


def as_batch(cond):
    if cond is None:
        return None
    if isinstance(cond, (tuple, list)):
        return tuple(c[None] for c in cond)
    return cond[None]


def with_other_slo(cond, other):
    image = other[0] if isinstance(other, (tuple, list)) else other
    return (image, cond[1]) if isinstance(cond, (tuple, list)) else image


def other_index(i, n):
    return (i + n // 2) % n


class OctVolumes(torch.utils.data.Dataset):
    def __init__(self, scans, target, slo_size=0, other=False, adapter=False, keep=0, slo_slice=None):
        self.scans = scans
        self.target = tuple(target)
        self.slo_size = slo_size
        self.other = other
        self.adapter = adapter
        self.keep = keep
        self.slo_slice = slo_slice

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        s = self.scans[i]
        try:
            vol, x = assemble(s["folder"], s["n"], self.target, self.keep, self.slo_slice)
            x = torch.from_numpy(x)[None]
            if not self.slo_size:
                return x
            cond = load_condition(s["folder"], s["n"], self.slo_size, self.adapter, vol.shape, self.target, self.keep)
            item = (x, cond)
            if self.other:
                j = other_index(i, len(self.scans))
                item += (with_other_slo(cond, torch.from_numpy(load_slo(self.scans[j]["folder"], self.slo_size))),)
        except Exception as e:
            raise RuntimeError("volume index %d failed: %s" % (i, type(e).__name__)) from None
        return item
