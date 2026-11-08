"""Focused regression tests for the from-scratch LeNet-5.

Forward-shape/probability sanity plus proof that one SGD step on a tiny
seeded batch reduces the loss.
"""

import numpy as np

from lenet.model import LeNet5


def test_forward_output_is_probability_distribution():
    net = LeNet5(seed=0)
    rng = np.random.default_rng(0)
    x = rng.random((4, 1, 28, 28), dtype=np.float32)

    out, _ = net.forward(x)

    assert out.shape == (4, 10)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out.sum(axis=1), np.ones(4), rtol=1e-5)


def test_one_sgd_step_reduces_loss():
    net = LeNet5(seed=1)
    rng = np.random.default_rng(2)
    x = rng.random((8, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=(8,)).astype(np.uint8)

    out_before, cache = net.forward(x)
    grads, loss_before = net.backward(out_before, cache, y)
    net.update_weights(grads, learning_rate=0.01)
    out_after, _ = net.forward(x)
    loss_after = net.cross_entropy_loss(out_after, y)

    assert np.isfinite(loss_before)
    assert float(loss_after) < float(loss_before)


def test_save_load_roundtrip(tmp_path):
    net = LeNet5(seed=3)
    rng = np.random.default_rng(4)
    x = rng.random((2, 1, 28, 28), dtype=np.float32)
    before, _ = net.forward(x)

    path = net.save(tmp_path / "model.npz")
    restored = LeNet5.load(path)
    after, _ = restored.forward(x)

    np.testing.assert_allclose(after, before, rtol=1e-6)
