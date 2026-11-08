"""Tiny SGD/AdamW optimizers operating on :class:`lenet.model.LeNet5`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PARAM_NAMES = (
    "kernel1",
    "bias1",
    "kernel2",
    "bias2",
    "f120_weights",
    "f120_bias",
    "f84_weights",
    "f84_bias",
    "out_weights",
    "out_bias",
)

# Standard practice: decay weights, not biases.
DECAY_NAMES = tuple(n for n in PARAM_NAMES if "bias" not in n)


class SGD:
    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0) -> None:
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, model, grads: dict[str, NDArray]) -> None:  # type: ignore[no-untyped-def]
        if self.weight_decay:
            grads = {
                n: (g + self.weight_decay * getattr(model, n) if n in DECAY_NAMES else g)
                for n, g in grads.items()
            }
        model.update_weights(grads, self.lr)


class Adam:
    """Adam with decoupled weight decay (AdamW style)."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: dict[str, NDArray] = {}
        self.v: dict[str, NDArray] = {}

    def step(self, model, grads: dict[str, NDArray]) -> None:  # type: ignore[no-untyped-def]
        self.t += 1
        for name in PARAM_NAMES:
            g = grads[name].astype(np.float64)
            if name not in self.m:
                self.m[name] = np.zeros_like(g)
                self.v[name] = np.zeros_like(g)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * g * g
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            current = getattr(model, name).astype(np.float64)
            if self.weight_decay and name in DECAY_NAMES:
                current = current * (1 - self.lr * self.weight_decay)
            setattr(model, name, (current - update).astype(np.float32))
