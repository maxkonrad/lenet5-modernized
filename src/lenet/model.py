"""LeNet-5 model assembled from :mod:`lenet.layers`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lenet import layers

Float = NDArray[np.float32]
Labels = NDArray[np.uint8]


def _he(rng: np.random.Generator, fan_in: int, shape: tuple[int, ...]) -> Float:
    return (rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)).astype(np.float32)


class LeNet5:
    """Classic LeNet-5 (dense C3 variant) with manual backprop.

    Args:
        seed: reproducibility seed for init (and dropout masks).
        fc_dropout: inverted-dropout probability on the two FC activations
            (0.0 disables it; applied only when ``training=True``).
    """

    def __init__(self, seed: int | None = None, fc_dropout: float = 0.0) -> None:
        if not 0.0 <= fc_dropout < 1.0:
            raise ValueError(f"fc_dropout must be in [0, 1), got {fc_dropout}")
        rng = np.random.default_rng(seed)
        self.fc_dropout = fc_dropout
        self._rng = rng
        self.kernel1: Float = _he(rng, 1 * 5 * 5, (6, 1, 5, 5))
        self.kernel2: Float = _he(rng, 6 * 5 * 5, (16, 6, 5, 5))
        self.bias1: Float = np.zeros((1, 6, 1, 1), dtype=np.float32)
        self.bias2: Float = np.zeros((1, 16, 1, 1), dtype=np.float32)

        self.f120_weights: Float = _he(rng, 256, (256, 120))
        self.f120_bias: Float = np.zeros((1, 120), dtype=np.float32)
        self.f84_weights: Float = _he(rng, 120, (120, 84))
        self.f84_bias: Float = np.zeros((1, 84), dtype=np.float32)
        self.out_weights: Float = _he(rng, 84, (84, 10))
        self.out_bias: Float = np.zeros((1, 10), dtype=np.float32)

    # -- forward --

    def _dropout(self, x: Float, key: str, cache: dict[str, Any]) -> Float:
        if self.fc_dropout <= 0.0:
            return x
        keep = 1.0 - self.fc_dropout
        mask = (self._rng.random(x.shape) < keep).astype(np.float32) / keep
        cache[key] = {"mask": mask}
        return (x * mask).astype(np.float32)

    def forward(self, data: Float, training: bool = True) -> tuple[Float, dict[str, Any]]:
        cache: dict[str, Any] = {}
        a1, (pre1, x1) = layers.conv_forward(data, self.kernel1, self.bias1)
        cache["conv1"] = {"input": x1, "pre_activation": pre1}
        p1, s1 = layers.avgpool_forward(a1)
        cache["pool1"] = {"input": a1, "input_shape": s1}

        a2, (pre2, x2) = layers.conv_forward(p1, self.kernel2, self.bias2)
        cache["conv2"] = {"input": x2, "pre_activation": pre2}
        p2, s2 = layers.avgpool_forward(a2)
        cache["pool2"] = {"input": a2, "input_shape": s2}

        flat = p2.reshape(p2.shape[0], -1).astype(np.float32)
        c5_pre = layers.linear_forward(flat, self.f120_weights, self.f120_bias)
        c5_out, _ = layers.relu_forward(c5_pre)
        if training:
            c5_out = self._dropout(c5_out, "drop_c5", cache)
        cache["c5"] = {"input": flat, "pre_activation": c5_pre}

        f6_pre = layers.linear_forward(c5_out, self.f84_weights, self.f84_bias)
        f6_out, _ = layers.relu_forward(f6_pre)
        if training:
            f6_out = self._dropout(f6_out, "drop_f6", cache)
        cache["f6"] = {"input": c5_out, "pre_activation": f6_pre}

        logits = layers.linear_forward(f6_out, self.out_weights, self.out_bias)
        cache["output"] = {"input": f6_out, "logits": logits}
        return layers.softmax(logits), cache

    # -- backward --

    def backward(
        self, fw_out: Float, cache: dict[str, Any], labels: Labels
    ) -> tuple[dict[str, Float], float]:
        grads: dict[str, Float] = {}
        loss = layers.cross_entropy(fw_out, labels)

        dz = layers.softmax_cross_entropy_grad(fw_out, labels)
        grads["out_weights"] = cache["output"]["input"].T @ dz
        grads["out_bias"] = dz.sum(axis=0, keepdims=True)
        dx = dz @ self.out_weights.T
        if "drop_f6" in cache:
            dx = dx * cache["drop_f6"]["mask"]

        dz_f6 = dx * (cache["f6"]["pre_activation"] > 0).astype(np.float32)
        grads["f84_weights"] = cache["f6"]["input"].T @ dz_f6
        grads["f84_bias"] = dz_f6.sum(axis=0, keepdims=True)
        dx_f6 = dz_f6 @ self.f84_weights.T
        if "drop_c5" in cache:
            dx_f6 = dx_f6 * cache["drop_c5"]["mask"]

        dz_c5 = dx_f6 * (cache["c5"]["pre_activation"] > 0).astype(np.float32)
        grads["f120_weights"] = cache["c5"]["input"].T @ dz_c5
        grads["f120_bias"] = dz_c5.sum(axis=0, keepdims=True)
        dx_c5 = dz_c5 @ self.f120_weights.T

        n = fw_out.shape[0]
        d_pool2 = layers.avgpool_backward(
            dx_c5.reshape(n, 16, 4, 4).astype(np.float32), cache["pool2"]["input_shape"]
        )
        dz_conv2 = d_pool2 * (cache["conv2"]["pre_activation"] > 0).astype(np.float32)
        grads["bias2"] = dz_conv2.sum(axis=(0, 2, 3), keepdims=True)
        grads["kernel2"] = layers.conv_filter_grad(dz_conv2, cache["conv2"]["input"], (16, 6, 5, 5))
        d_conv2 = layers.conv_data_grad(dz_conv2, self.kernel2)

        d_pool1 = layers.avgpool_backward(
            d_conv2.reshape(n, 6, 12, 12).astype(np.float32), cache["pool1"]["input_shape"]
        )
        dz_conv1 = d_pool1 * (cache["conv1"]["pre_activation"] > 0).astype(np.float32)
        grads["bias1"] = dz_conv1.sum(axis=(0, 2, 3), keepdims=True)
        grads["kernel1"] = layers.conv_filter_grad(dz_conv1, cache["conv1"]["input"], (6, 1, 5, 5))

        return grads, loss

    def update_weights(self, grads: dict[str, Float], learning_rate: float) -> None:
        for name in (
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
        ):
            setattr(self, name, getattr(self, name) - learning_rate * grads[name])

    # -- helpers --

    def cross_entropy_loss(self, probabilities: Float, labels: Labels) -> float:
        return layers.cross_entropy(probabilities, labels)

    def predict(self, data: Float) -> Labels:
        probs, _ = self.forward(np.asarray(data, dtype=np.float32), training=False)
        return np.argmax(probs, axis=1).astype(np.uint8)

    def evaluate(self, data: Float, labels: Labels, batch_size: int = 256) -> tuple[float, float]:
        """Return ``(avg_loss, accuracy_pct)`` over the dataset."""
        total_loss, correct, n = 0.0, 0, data.shape[0]
        for i in range(0, n, batch_size):
            probs, _ = self.forward(data[i : i + batch_size], training=False)
            total_loss += layers.cross_entropy(probs, labels[i : i + batch_size]) * probs.shape[0]
            correct += int(np.sum(np.argmax(probs, axis=1) == labels[i : i + batch_size]))
        return total_loss / n, 100.0 * correct / n

    def state_dict(self) -> dict[str, Float]:
        return {
            name: getattr(self, name)
            for name in (
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
        }

    def load_state_dict(self, state: dict[str, NDArray]) -> None:
        for name, value in state.items():
            setattr(self, name, np.asarray(value, dtype=np.float32))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **self.state_dict())
        return target

    @classmethod
    def load(cls, path: str | Path) -> LeNet5:
        net = cls()
        with np.load(path) as blob:
            net.load_state_dict({k: blob[k] for k in blob.files})
        return net
