"""Weight decay, dropout, and ensemble averaging."""

import numpy as np
import pytest

from lenet.ensemble import accuracy, ensemble_probs
from lenet.model import LeNet5
from lenet.optim import Adam


def _batch():
    rng = np.random.default_rng(31)
    x = rng.random((4, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=(4,)).astype(np.uint8)
    return x, y


def test_dropout_zero_matches_eval_mode():
    net = LeNet5(seed=0)
    x, _ = _batch()
    train_out, _ = net.forward(x, training=True)
    eval_out, _ = net.forward(x, training=False)
    np.testing.assert_array_equal(train_out, eval_out)


def test_dropout_deterministic_given_seed():
    x, _ = _batch()
    net1 = LeNet5(seed=2, fc_dropout=0.5)
    net2 = LeNet5(seed=2, fc_dropout=0.5)
    out1, _ = net1.forward(x, training=True)
    out2, _ = net2.forward(x, training=True)
    np.testing.assert_array_equal(out1, out2)


def test_dropout_differs_from_eval_and_trains():
    net = LeNet5(seed=3, fc_dropout=0.5)
    x, y = _batch()
    train_out, cache = net.forward(x, training=True)
    eval_out, _ = net.forward(x, training=False)
    assert "drop_c5" in cache and "drop_f6" in cache
    assert not np.array_equal(train_out, eval_out)
    grads, loss_before = net.backward(train_out, cache, y)
    net.update_weights(grads, learning_rate=0.01)
    assert np.isfinite(loss_before)


def test_invalid_dropout_raises():
    with pytest.raises(ValueError, match="fc_dropout"):
        LeNet5(seed=0, fc_dropout=1.0)


def test_weight_decay_shrinks_weights_not_biases():
    x, y = _batch()
    plain = LeNet5(seed=4)
    decayed = LeNet5(seed=4)
    probs, cache = plain.forward(x)
    grads, _ = plain.backward(probs, cache, y)

    Adam(lr=1e-3).step(plain, {k: v.copy() for k, v in grads.items()})
    Adam(lr=1e-3, weight_decay=1.0).step(decayed, {k: v.copy() for k, v in grads.items()})

    assert float(np.abs(decayed.out_weights).sum()) < float(np.abs(plain.out_weights).sum())
    np.testing.assert_allclose(decayed.out_bias, plain.out_bias, rtol=1e-6)


def test_ensemble_averages_members(tmp_path):
    rng = np.random.default_rng(41)
    x = rng.random((6, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=(6,)).astype(np.uint8)
    paths = []
    for seed in (10, 11):
        net = LeNet5(seed=seed)
        paths.append(net.save(tmp_path / f"m{seed}.npz"))

    avg, members = ensemble_probs(paths, x, batch_size=4)
    assert avg.shape == (6, 10)
    np.testing.assert_allclose(avg, np.mean(np.stack(members), axis=0), rtol=1e-6)
    assert 0.0 <= accuracy(avg, y) <= 100.0
