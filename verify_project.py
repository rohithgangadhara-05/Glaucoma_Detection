"""
verify_project.py
=================
Quick self-check for evaluators: confirms that every required file exists, the dataset is the real RIM-ONE DL data,
the trained model loads, a held-out test image gives a prediction and a Grad-CAM heatmap, and that the saved test
metrics can be reproduced from the saved model.

Usage:  python verify_project.py
"""

import csv
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["app.py", "train.py", "fetch_data.py", "verify_project.py", "requirements.txt", "run_project.bat", "run_project.sh",
            "README.md", "utils/__init__.py", "utils/preprocessing.py", "utils/gradcam.py",
            "tests/__init__.py", "tests/conftest.py", "tests/test_preprocessing.py", "tests/test_gradcam.py",
            "tests/test_fetch_data.py", "tests/test_dataset.py", "tests/test_model_and_metrics.py", "tests/test_app.py",
            "models/glaucoma_model.h5", "models/metrics.json", "models/test_predictions.csv", "models/training_history.png",
            "models/roc_curve.png", "models/confusion_matrix.png", "models/gradcam_samples.png"]
ok = True


def check(cond, msg):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    ok &= bool(cond)


print("1. Project files")
for f in REQUIRED:
    check(os.path.isfile(os.path.join(BASE, f)), f)

print("2. Dataset (RIM-ONE DL)")
splits = os.path.join(BASE, "data", "splits.csv")
if os.path.isfile(splits):
    rows = list(csv.DictReader(open(splits, encoding="utf-8")))
    n1 = sum(int(r["label"]) for r in rows)
    check(len(rows) == 485 and n1 == 172, f"{len(rows)} images listed ({len(rows) - n1} normal, {n1} glaucoma); expected 485 (313 / 172)")
    missing = [r["file"] for r in rows if not os.path.isfile(os.path.join(BASE, "data", "raw", *r["file"].split("/")))]
    check(not missing, f"all image files present ({len(missing)} missing)")
    check(not any(r["file"].split("/")[-1].startswith("synth") for r in rows), "no synthetic images")
else:
    check(False, "data/splits.csv missing - run: python fetch_data.py")

if not os.path.isfile(os.path.join(BASE, "models", "glaucoma_model.h5")) or not os.path.isfile(splits):
    sys.exit("\nCannot continue without the model and the dataset.")

print("3. Model, prediction and Grad-CAM")
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, BASE)
from utils.gradcam import compute_gradcam  # noqa: E402
from utils.preprocessing import preprocess  # noqa: E402

model = tf.keras.models.load_model(os.path.join(BASE, "models", "glaucoma_model.h5"), compile=False)
check(model.input_shape == (None, 224, 224, 3), f"model loads, input {model.input_shape}, {model.count_params():,} parameters")
preds = list(csv.DictReader(open(os.path.join(BASE, "models", "test_predictions.csv"), encoding="utf-8")))
sample = os.path.join(BASE, "data", "raw", *preds[0]["file"].split("/"))
t0 = time.perf_counter()
heat, p = compute_gradcam(model, preprocess(sample)[None])
check(heat.shape == (7, 7) and 0.0 <= p <= 1.0, f"Grad-CAM heatmap {heat.shape}, P(glaucoma) = {p:.4f} for {preds[0]['file']} "
                                                f"({time.perf_counter() - t0:.2f} s incl. first-call overhead)")

print("4. Reproduce the saved test metrics from the saved model")
x = np.stack([preprocess(os.path.join(BASE, "data", "raw", *r["file"].split("/"))) for r in preds])
y = np.array([int(r["label"]) for r in preds])
probs = model.predict(x, batch_size=32, verbose=0).ravel()
auc = roc_auc_score(y, probs)
saved = json.load(open(os.path.join(BASE, "models", "metrics.json"), encoding="utf-8"))
check(abs(auc - saved["auc"]) < 1e-3, f"test AUC recomputed = {auc:.4f}, saved = {saved['auc']:.4f}")
acc = float(((probs >= 0.5).astype(int) == y).mean())
check(abs(acc - saved["at_threshold_0.5"]["accuracy"]) < 1e-3, f"test accuracy recomputed = {acc:.4f}, saved = {saved['at_threshold_0.5']['accuracy']:.4f}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
