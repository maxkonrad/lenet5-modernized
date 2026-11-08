"""Training-time augmentation in dependency-free NumPy.

Random integer shifts plus Simard-style elastic distortions, fused into a
single bilinear remap per image. Operates on ``(N, C, H, W)`` float32 batches
in ``[0, 1]`` and preserves shape, dtype, and range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float = NDArray[np.float32]


@dataclass
class AugmentConfig:
    max_shift: int = 2
    elastic_prob: float = 0.5
    alpha: float = 20.0
    sigma: float = 4.0


def _box_blur_once(field: NDArray[np.float64], radius: int) -> NDArray[np.float64]:
    h, w = field.shape
    padded = np.pad(field, radius, mode="constant")
    integ = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    integ = np.pad(integ, ((1, 0), (1, 0)), mode="constant")
    k = 2 * radius + 1
    out = (integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]) / k**2
    assert out.shape == (h, w)
    return out


def _smooth_field(field: NDArray[np.float64], sigma: float, passes: int = 2) -> NDArray[np.float64]:
    if sigma <= 0:
        return field
    radius = max(1, round((-1 + np.sqrt(1 + 12 * sigma**2 / passes)) / 2))
    for _ in range(passes):
        field = _box_blur_once(field, radius)
    return field


def _displacement(
    h: int, w: int, rng: np.random.Generator, alpha: float, sigma: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    raw_x = rng.uniform(-1.0, 1.0, size=(h, w))
    raw_y = rng.uniform(-1.0, 1.0, size=(h, w))
    return _smooth_field(raw_x, sigma) * alpha, _smooth_field(raw_y, sigma) * alpha


def _remap(img: Float, map_x: NDArray[np.float64], map_y: NDArray[np.float64]) -> Float:
    h, w = img.shape
    padded = np.pad(img, 1, mode="constant")
    mx = np.clip(map_x + 1, 0, w + 1)
    my = np.clip(map_y + 1, 0, h + 1)
    x0 = np.floor(mx).astype(np.int64)
    y0 = np.floor(my).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w + 1)
    y1 = np.clip(y0 + 1, 0, h + 1)
    wx = (mx - x0).astype(np.float32)
    wy = (my - y0).astype(np.float32)
    return (
        (1 - wx) * (1 - wy) * padded[y0, x0]
        + wx * (1 - wy) * padded[y0, x1]
        + (1 - wx) * wy * padded[y1, x0]
        + wx * wy * padded[y1, x1]
    ).astype(np.float32)


def augment_batch(x: Float, rng: np.random.Generator, cfg: AugmentConfig | None = None) -> Float:
    """Randomly shift (and sometimes elastically distort) each image in batch."""
    cfg = cfg if cfg is not None else AugmentConfig()
    n, c, h, w = x.shape
    base_y, base_x = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty_like(x)
    for i in range(n):
        sx = int(rng.integers(-cfg.max_shift, cfg.max_shift + 1))
        sy = int(rng.integers(-cfg.max_shift, cfg.max_shift + 1))
        if rng.random() < cfg.elastic_prob:
            dx, dy = _displacement(h, w, rng, cfg.alpha, cfg.sigma)
        else:
            dx = np.zeros((h, w))
            dy = np.zeros((h, w))
        map_x = base_x - sx + dx
        map_y = base_y - sy + dy
        for ch in range(c):
            out[i, ch] = _remap(x[i, ch], map_x, map_y)
    return out
