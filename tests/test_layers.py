"""Numeric-gradient check on the output layer + pool roundtrip."""

import numpy as np

from lenet import layers
from lenet.model import LeNet5


def test_out_weights_numeric_gradient():
    net = LeNet5(seed=0)
    rng = np.random.default_rng(7)
    x = rng.random((2, 1, 28, 28), dtype=np.float32)
    y = np.array([3, 7], dtype=np.uint8)

    probs, cache = net.forward(x)
    grads, _ = net.backward(probs, cache, y)
    analytic = grads["out_weights"][0, 0]

    eps = 1e-3
    orig = float(net.out_weights[0, 0])
    net.out_weights[0, 0] = np.float32(orig + eps)
    loss_plus = layers.cross_entropy(net.forward(x)[0], y)
    net.out_weights[0, 0] = np.float32(orig - eps)
    loss_minus = layers.cross_entropy(net.forward(x)[0], y)
    net.out_weights[0, 0] = np.float32(orig)

    numeric = (loss_plus - loss_minus) / (2 * eps)
    assert abs(float(analytic) - numeric) < 5e-2, (analytic, numeric)


def test_avgpool_backward_distributes_evenly():
    rng = np.random.default_rng(11)
    x = rng.random((2, 3, 4, 6), dtype=np.float32)
    out, shape = layers.avgpool_forward(x)
    assert out.shape == (2, 3, 2, 3)
    dx = layers.avgpool_backward(np.ones_like(out), shape)
    np.testing.assert_allclose(dx, np.full_like(x, 0.25), rtol=1e-6)


def test_conv_grad_shapes():
    net = LeNet5(seed=0)
    rng = np.random.default_rng(13)
    x = rng.random((2, 1, 28, 28), dtype=np.float32)
    y = np.array([1, 2], dtype=np.uint8)
    probs, cache = net.forward(x)
    grads, _ = net.backward(probs, cache, y)
    assert grads["kernel1"].shape == net.kernel1.shape
    assert grads["kernel2"].shape == net.kernel2.shape
    assert grads["bias1"].shape == net.bias1.shape
    assert grads["bias2"].shape == net.bias2.shape
