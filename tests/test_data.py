"""Data loader: magic validation + real MNIST layout."""

from pathlib import Path

import numpy as np
import pytest

from lenet import data

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_load_mnist_shapes_and_range():
    x_train, y_train, x_test, y_test = data.load_mnist(DATA_DIR)
    assert x_train.shape[1:] == (1, 28, 28)
    assert x_test.shape[1:] == (1, 28, 28)
    assert x_train.dtype == np.float32
    assert float(x_train.max()) <= 1.0 and float(x_train.min()) >= 0.0
    assert y_train.shape == (x_train.shape[0],)
    assert y_test.shape == (x_test.shape[0],)


def test_bad_magic_raises(tmp_path):
    bad = tmp_path / "bad.idx3-ubyte"
    bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 16)
    with pytest.raises(ValueError, match="bad magic"):
        data.load_images(bad)
