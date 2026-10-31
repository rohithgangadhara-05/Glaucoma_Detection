"""Unit tests for utils/preprocessing.py (CLAHE, resizing, normalisation, file reading)."""
import cv2
import numpy as np
import pytest

from utils.preprocessing import IMG_SIZE, apply_clahe, preprocess, preprocess_bgr, read_bgr


def _fundus_like(h=300, w=400):
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 60, 150)                                     # reddish background (BGR)
    cv2.circle(img, (w // 2, h // 2), 60, (150, 190, 230), -1)  # bright disc
    cv2.circle(img, (w // 2, h // 2), 25, (190, 220, 245), -1)  # paler cup
    return np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)


def test_clahe_keeps_shape_and_dtype():
    img = _fundus_like()
    out = apply_clahe(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_clahe_stretches_a_low_contrast_luminance_range():
    rng = np.random.default_rng(0)
    grey = rng.integers(100, 121, (256, 256)).astype(np.uint8)      # brightness only spans 100-120
    img = cv2.merge([grey, grey, grey])
    l_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    l_after = cv2.cvtColor(apply_clahe(img), cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    assert l_after.std() > 1.5 * l_before.std()
    assert np.ptp(l_after) > np.ptp(l_before)


def test_clahe_changes_only_the_luminance_channel():
    img = _fundus_like()
    lab_in, lab_out = (cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(int) for x in (img, apply_clahe(img)))
    assert np.abs(lab_in[..., 1:] - lab_out[..., 1:]).mean() < 2.0   # a*, b* colour channels preserved


def test_preprocess_bgr_output_shape_range_and_display():
    tensor, display = preprocess_bgr(_fundus_like())
    assert tensor.shape == (IMG_SIZE, IMG_SIZE, 3) and tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert display.shape == (IMG_SIZE, IMG_SIZE, 3) and display.dtype == np.uint8
    assert np.allclose(tensor, display / 255.0)


def test_preprocess_reads_files_including_non_ascii_paths(tmp_path):
    path = tmp_path / "fundus_ü.png"
    cv2.imencode(".png", _fundus_like())[1].tofile(str(path))
    assert preprocess(str(path)).shape == (IMG_SIZE, IMG_SIZE, 3)


def test_read_bgr_rejects_missing_and_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_bgr(str(tmp_path / "missing.png"))
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"this is not a PNG")
    with pytest.raises(FileNotFoundError):
        read_bgr(str(bad))
