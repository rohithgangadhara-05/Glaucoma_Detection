"""
utils/preprocessing.py
=======================
Shared image preprocessing used by BOTH train.py and app.py, so the model always sees images prepared the same way.

Pipeline: read -> CLAHE on the L channel of CIE L*a*b* -> resize to 224 x 224 -> RGB -> float32 in [0, 1].
(The model itself rescales [0, 1] to the [-1, 1] range that MobileNetV2's ImageNet weights expect.)
"""

import cv2
import numpy as np

IMG_SIZE = 224
CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)


def apply_clahe(bgr_img: np.ndarray, clip_limit: float = CLIP_LIMIT, tile_grid: tuple = TILE_GRID) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the luminance (L) channel of the LAB colour
    space. Only brightness is equalised, so the natural colours of the retina are kept.

    Args:
        bgr_img:    Input image in BGR uint8 format.
        clip_limit: Threshold for contrast limiting.
        tile_grid:  Size of the grid for histogram equalization.

    Returns:
        CLAHE-enhanced BGR uint8 image.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    merged = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_bgr(bgr_img: np.ndarray, img_size: int = IMG_SIZE):
    """
    CLAHE -> resize -> RGB -> [0, 1].

    Returns:
        tensor:  float32 array (img_size, img_size, 3) in [0, 1], the model input (without batch dimension).
        display: uint8 RGB array (img_size, img_size, 3) of the CLAHE-enhanced image, for visualisation.
    """
    enhanced = apply_clahe(bgr_img)
    resized = cv2.resize(enhanced, (img_size, img_size), interpolation=cv2.INTER_AREA)
    display = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return display.astype(np.float32) / 255.0, display


def read_bgr(image_path: str) -> np.ndarray:
    """Read an image file as BGR uint8 (works with non-ASCII Windows paths)."""
    data = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return bgr


def preprocess(image_path: str, img_size: int = IMG_SIZE) -> np.ndarray:
    """Read an image file and return the float32 model input (img_size, img_size, 3) in [0, 1]."""
    return preprocess_bgr(read_bgr(image_path), img_size)[0]
