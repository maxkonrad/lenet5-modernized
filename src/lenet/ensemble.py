"""Ensemble eval: average test-set probabilities over checkpoints.

Usage:
    uv run lenet-train --seed 0 --checkpoint-dir checkpoints/seed0 ...
    uv run lenet-train --seed 1 --checkpoint-dir checkpoints/seed1 ...
    uv run lenet-ensemble --checkpoints checkpoints/seed0/best.npz \\
        checkpoints/seed1/best.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from lenet.data import default_data_dir, load_mnist
from lenet.model import Float, LeNet5


def model_probs(path: str | Path, data: Float, batch_size: int = 512) -> NDArray[np.float32]:
    net = LeNet5.load(path)
    outs = []
    for i in range(0, data.shape[0], batch_size):
        probs, _ = net.forward(data[i : i + batch_size], training=False)
        outs.append(probs)
    return np.concatenate(outs, axis=0)


def ensemble_probs(
    paths: list[str | Path], data: Float, batch_size: int = 512
) -> tuple[NDArray[np.float32], list[NDArray[np.float32]]]:
    per_model = [model_probs(p, data, batch_size) for p in paths]
    return np.mean(np.stack(per_model, axis=0), axis=0).astype(np.float32), per_model


def accuracy(probs: NDArray[np.float32], labels: NDArray[np.uint8]) -> float:
    return 100.0 * float(np.sum(np.argmax(probs, axis=1) == labels)) / labels.shape[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Average checkpoint probabilities on MNIST test")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args(argv)

    _, _, x_test, y_test = load_mnist(args.data_dir)
    avg, per_model = ensemble_probs(args.checkpoints, x_test, args.batch_size)
    for path, probs in zip(args.checkpoints, per_model, strict=True):
        print(f"{path}: {accuracy(probs, y_test):.2f}%")
    print(f"ensemble ({len(per_model)} models): {accuracy(avg, y_test):.2f}%")


if __name__ == "__main__":
    main()
