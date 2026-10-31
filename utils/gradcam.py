"""
utils/gradcam.py
================
Grad-CAM (Selvaraju et al., ICCV 2017) written from scratch with tf.GradientTape, shared by app.py and train.py.

For the class score y (glaucoma probability) and the last convolutional feature maps A^k of MobileNetV2 (7 x 7 x 1280):
    alpha_k = mean_ij( dy / dA^k_ij )          (global average of the gradients)
    heatmap = ReLU( sum_k alpha_k * A^k )      (only regions that increase the glaucoma score)
The heatmap is normalised to [0, 1], upsampled and blended with the image using the JET colormap.
"""

import cv2
import numpy as np
import tensorflow as tf


def _split_model(model: tf.keras.Model):
    """Return (layers before the backbone, backbone sub-model, layers after it) for the nested MobileNetV2 model."""
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            pre = [l for l in model.layers[:idx] if not isinstance(l, tf.keras.layers.InputLayer)]
            return pre, layer, model.layers[idx + 1:]
    raise ValueError("Expected a model with a nested convolutional backbone (MobileNetV2).")


def compute_gradcam(model: tf.keras.Model, input_tensor: np.ndarray, target_class: int = 1):
    """
    Args:
        model:        trained GlaucomaDetector (input [0, 1], sigmoid output = P(glaucoma)).
        input_tensor: (1, 224, 224, 3) float32 in [0, 1].
        target_class: 1 explains the glaucoma score, 0 explains the healthy score (1 - p).

    Returns:
        heatmap (7 x 7, float32 in [0, 1]) and the predicted probability of glaucoma.
    """
    pre, backbone, head = _split_model(model)
    x = tf.cast(input_tensor, tf.float32)
    for layer in pre:                      # e.g. the Rescaling layer [0, 1] -> [-1, 1]
        x = layer(x)
    with tf.GradientTape() as tape:
        conv_maps = backbone(x, training=False)   # (1, 7, 7, 1280)
        tape.watch(conv_maps)
        y = conv_maps
        for layer in head:                         # GAP -> Dropout -> BatchNorm -> Dense(sigmoid)
            y = layer(y, training=False)
        prob = y[:, 0]
        score = prob if target_class == 1 else 1.0 - prob
    grads = tape.gradient(score, conv_maps)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))                 # alpha_k, shape (1280,)
    cam = tf.nn.relu(tf.reduce_sum(conv_maps[0] * weights, axis=-1))  # (7, 7)
    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8) if cam.max() > 0 else np.zeros_like(cam)
    return cam.astype(np.float32), float(prob.numpy()[0])


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Upsample the heatmap to the image size (bicubic), colour it with JET and blend it with the RGB image."""
    h, w = rgb_image.shape[:2]
    up = np.clip(cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * up), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(colour, alpha, rgb_image.astype(np.uint8), 1.0 - alpha, 0)


def heatmap_centre_share(heatmap: np.ndarray, radius: float = 0.35) -> float:
    """
    Share of the heatmap 'mass' that falls inside a central disc of the image. RIM-ONE DL images are cropped around the
    optic nerve head, so a high value means the explanation concentrates on the optic disc region rather than the border.
    """
    h, w = heatmap.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    inside = ((yy - cy) / h) ** 2 + ((xx - cx) / w) ** 2 <= radius ** 2
    total = heatmap.sum()
    return float(heatmap[inside].sum() / total) if total > 0 else 0.0
