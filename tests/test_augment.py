"""Augmentation preserves shape/dtype/range and is seed-deterministic."""

import numpy as np

from lenet.augment import AugmentConfig, augment_batch


def _batch():
    rng = np.random.default_rng(21)
    return rng.random((4, 1, 28, 28), dtype=np.float32)


def test_identity_config_returns_input_unchanged():
    x = _batch()
    out = augment_batch(x, np.random.default_rng(0), AugmentConfig(max_shift=0, elastic_prob=0.0))
    np.testing.assert_array_equal(out, x)


def test_deterministic_given_seed():
    x = _batch()
    cfg = AugmentConfig()
    out1 = augment_batch(x, np.random.default_rng(5), cfg)
    out2 = augment_batch(x, np.random.default_rng(5), cfg)
    np.testing.assert_array_equal(out1, out2)


def test_shape_dtype_range_preserved():
    x = _batch()
    out = augment_batch(x, np.random.default_rng(6), AugmentConfig())
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_augmentation_actually_changes_images():
    x = _batch()
    out = augment_batch(x, np.random.default_rng(7), AugmentConfig())
    assert not np.array_equal(out, x)
