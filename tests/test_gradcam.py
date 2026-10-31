"""Tests for utils/gradcam.py with the trained model."""
import numpy as np
import pytest

from tests.conftest import needs_model
from utils.gradcam import compute_gradcam, heatmap_centre_share, overlay_heatmap
from utils.preprocessing import IMG_SIZE


@needs_model
def test_gradcam_heatmap_shape_range_and_probability(model):
    x = np.random.default_rng(1).random((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    heat, prob = compute_gradcam(model, x)
    assert heat.shape == (7, 7) and heat.dtype == np.float32
    assert 0.0 <= heat.min() and heat.max() <= 1.0
    assert 0.0 <= prob <= 1.0


@needs_model
def test_gradcam_probability_equals_model_prediction(model):
    x = np.random.default_rng(2).random((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    _, prob = compute_gradcam(model, x)
    assert prob == pytest.approx(float(model.predict(x, verbose=0)[0, 0]), abs=1e-5)


@needs_model
def test_gradcam_for_both_classes(model):
    x = np.random.default_rng(3).random((1, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    for target in (0, 1):
        heat, _ = compute_gradcam(model, x, target_class=target)
        assert heat.shape == (7, 7)


def test_overlay_shape_and_blending():
    rgb = np.full((IMG_SIZE, IMG_SIZE, 3), 100, np.uint8)
    out = overlay_heatmap(rgb, np.zeros((7, 7), np.float32), alpha=0.0)
    assert out.shape == rgb.shape and np.array_equal(out, rgb)   # alpha 0 -> original image
    hot = overlay_heatmap(rgb, np.ones((7, 7), np.float32), alpha=0.45)
    assert hot.shape == rgb.shape and not np.array_equal(hot, rgb)


def test_centre_share():
    peak = np.zeros((7, 7), np.float32)
    peak[3, 3] = 1.0
    border = np.zeros((7, 7), np.float32)
    border[0, 0] = 1.0
    assert heatmap_centre_share(peak) == pytest.approx(1.0)
    assert heatmap_centre_share(border) == pytest.approx(0.0)
    assert heatmap_centre_share(np.zeros((7, 7))) == 0.0
