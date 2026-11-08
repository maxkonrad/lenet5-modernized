"""Small pure layer helpers shared by the model.

All math is float32. Each ``*_forward`` returns ``(output, cache)`` and each
``*_backward`` maps upstream gradients back to ``(dx, grads)`` so
``model.backward`` stays readable instead of duplicating conv math per layer.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

Float = NDArray[np.float32]


def relu_forward(x: Float) -> tuple[Float, Float]:
    mask = (x > 0).astype(np.float32)
    return (x * mask).astype(np.float32), mask


def avgpool_forward(x: Float) -> tuple[Float, tuple[int, ...]]:
    n, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"avgpool needs even H/W, got {(h, w)}")
    out = x.reshape(n, c, h // 2, 2, w // 2, 2).mean(axis=(3, 5)).astype(np.float32)
    return out, (n, c, h, w)


def avgpool_backward(dout: Float, input_shape: tuple[int, ...]) -> Float:
    n, c, h, w = input_shape
    up = np.repeat(np.repeat(dout, 2, axis=2), 2, axis=3) / 4.0
    return up.reshape(n, c, h, w).astype(np.float32)


def conv_forward(x: Float, weight: Float, bias: Float) -> tuple[Float, tuple[Float, Float]]:
    """Valid cross-correlation + bias + ReLU. Returns ``(activated, (pre, x))``."""
    kh, kw = weight.shape[2], weight.shape[3]
    windows = sliding_window_view(x, window_shape=(kh, kw), axis=(2, 3))
    fm = np.tensordot(windows, weight, axes=([1, 4, 5], [1, 2, 3]))
    fm = np.moveaxis(fm, -1, 1).astype(np.float32)
    pre = (fm + bias).astype(np.float32)
    activated, _ = relu_forward(pre)
    return activated, (pre, x)


def conv_filter_grad(dz: Float, x: Float, kernel_shape: tuple[int, ...]) -> Float:
    kh, kw = kernel_shape[2], kernel_shape[3]
    windows = sliding_window_view(x, window_shape=(kh, kw), axis=(2, 3))
    return np.tensordot(dz, windows, axes=([0, 2, 3], [0, 2, 3])).astype(np.float32)


def conv_data_grad(dz: Float, weight: Float) -> Float:
    """Gradient w.r.t. conv input via full convolution with the flipped kernel."""
    kh, kw = weight.shape[2], weight.shape[3]
    pad_h, pad_w = kh - 1, kw - 1
    dz_pad = np.pad(dz, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    windows = sliding_window_view(dz_pad, window_shape=(kh, kw), axis=(2, 3))
    w_rot = weight[:, :, ::-1, ::-1]
    out = np.tensordot(windows, w_rot, axes=([1, 4, 5], [0, 2, 3]))
    return np.moveaxis(out, -1, 1).astype(np.float32)


def linear_forward(x: Float, weight: Float, bias: Float) -> Float:
    return (x @ weight + bias).astype(np.float32)


def softmax(x: Float) -> Float:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted).astype(np.float32)
    return (exp / exp.sum(axis=-1, keepdims=True)).astype(np.float32)


def cross_entropy(probs: Float, labels: NDArray[np.uint8]) -> float:
    n = probs.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    return float(-np.sum(one_hot * np.log(clipped)) / n)


def softmax_cross_entropy_grad(probs: Float, labels: NDArray[np.uint8]) -> Float:
    n = probs.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    return ((probs - one_hot) / n).astype(np.float32)
