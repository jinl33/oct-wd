import math

import torch

Z, H, W = -3, -2, -1
SQRT2 = math.sqrt(2.0)

DESIGNS = {
    "w2z1": (Z, H, W, H, W),  # 32 channels: 2 levels within the B-scan, 1 across slices
    "w1z1": (Z, H, W),  # 8 channels: 1 level on every axis
    "w2z2": (Z, H, W, Z, H, W),  # 64 channels: the brain pipeline's layout
}


def channels(axes):
    return 2 ** len(axes)


def grid(shape, axes):
    """Wavelet-domain (Z, H, W) for a pixel-domain (Z, H, W). Raises if an axis cannot be halved."""
    s = list(shape)
    for ax in axes:
        if s[ax] % 2:
            raise ValueError("axis %d has odd length %d in %s" % (ax, s[ax], tuple(shape)))
        s[ax] //= 2
    return tuple(s)


def check_unet_grid(shape, axes, factor=16):
    g = grid(shape, axes)
    if any(d % factor for d in g):
        raise ValueError("wavelet grid %s must be divisible by %d for this U-Net" % (g, factor))
    return g


def forward(x, axes):
    """x: (..., C, Z, H, W)."""
    for ax in axes:
        if x.shape[ax] % 2:
            raise ValueError("axis %d has odd length %d" % (ax, x.shape[ax]))
        a = x.index_select(ax, torch.arange(0, x.shape[ax], 2, device=x.device))
        b = x.index_select(ax, torch.arange(1, x.shape[ax], 2, device=x.device))
        x = torch.cat([a + b, a - b], dim=-4) / SQRT2
    return x / SQRT2 ** len(axes)


def inverse(y, axes):
    y = y * SQRT2 ** len(axes)
    for ax in reversed(axes):
        low, high = y.chunk(2, dim=-4)
        a, b = (low + high) / SQRT2, (low - high) / SQRT2
        y = torch.stack([a, b], dim=ax).flatten(ax - 1, ax)
    return y
