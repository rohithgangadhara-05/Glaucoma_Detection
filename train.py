"""
train.py
========
Training and evaluation pipeline for Glaucoma Detection on the RIM-ONE DL database.

Pipeline:
  1. Read the OFFICIAL RIM-ONE DL partition from data/splits.csv (written by fetch_data.py)
       --split random   : the randomly partitioned train / test sets (default)
       --split hospital : train on one hospital, test on the other two (generalisation experiment)
     A stratified 15 % of the training set is held out for validation; the test set is used only once, at the end.
  2. Preprocess every image with utils/preprocessing.py (CLAHE on L channel -> 224 x 224 -> [0, 1])
  3. MobileNetV2 (ImageNet) transfer learning: [0,1] -> Rescaling to [-1,1] -> backbone -> GAP -> Dropout(0.5)
     -> BatchNorm -> Dense(1, sigmoid)
  4. Phase 1: frozen backbone, head only (Adam 1e-3).  Phase 2: unfreeze top 30 backbone layers (Adam 1e-4).
     EarlyStopping on validation AUC, ReduceLROnPlateau, class weights for the 313 / 172 class imbalance.
  5. Evaluate on the untouched test set and save every result to the output folder:
       glaucoma_model.h5, metrics.json, test_predictions.csv, training_history.png, roc_curve.png,
       confusion_matrix.png, gradcam_samples.png
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from utils.gradcam import compute_gradcam, heatmap_centre_share, overlay_heatmap
from utils.preprocessing import IMG_SIZE, preprocess, preprocess_bgr, read_bgr

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
SPLITS_CSV = os.path.join(BASE_DIR, "data", "splits.csv")
BATCH_SIZE = 16
SEED = 42
VAL_FRACTION = 0.15
FINE_TUNE_LAYERS = 30
BENCHMARK_SPECIFICITY = 0.85   # RIM-ONE DL reports sensitivity at 85 % specificity


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ── data ──────────────────────────────────────────────────────────────────────
def load_split(split_name):
    if not os.path.isfile(SPLITS_CSV):
        sys.exit("ERROR: data/splits.csv not found. Run: python fetch_data.py")
    column = f"{split_name}_split"
    rows = list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))
    train = [(os.path.join(DATA_RAW, r["file"]), int(r["label"])) for r in rows if r[column] == "train"]
    test = [(os.path.join(DATA_RAW, r["file"]), int(r["label"])) for r in rows if r[column] == "test"]
    if not train or not test:
        sys.exit(f"ERROR: no '{split_name}' partition in {SPLITS_CSV}")
    tr, va = train_test_split(train, test_size=VAL_FRACTION, stratify=[l for _, l in train], random_state=SEED)
    return tr, va, test


def to_arrays(items):
    x = np.stack([preprocess(p) for p, _ in items]).astype(np.float32)
    y = np.array([l for _, l in items], dtype=np.float32)
    return x, y


def augment(image, label):
    """Light, anatomy-preserving augmentation (horizontal flip mirrors left/right eyes; no vertical flip)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.08)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    return tf.clip_by_value(image, 0.0, 1.0), label


def make_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ── model ─────────────────────────────────────────────────────────────────────
def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="fundus_0_1")
    x = layers.Rescaling(2.0, offset=-1.0, name="to_minus1_plus1")(inputs)   # MobileNetV2 expects [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.BatchNormalization(name="bn")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="glaucoma_prob")(x)
    model = Model(inputs, outputs, name="GlaucomaDetector")
    return model, base


def compile_model(model, lr):
    model.compile(optimizer=Adam(learning_rate=lr), loss=BinaryCrossentropy(),
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def val_auc(model, x, y):
    return float(roc_auc_score(y, model.predict(x, batch_size=32, verbose=0).ravel()))


# ── evaluation helpers ────────────────────────────────────────────────────────
def threshold_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.astype(int), pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {"threshold": thr, "accuracy": (tp + tn) / len(y), "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1, "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def sensitivity_at_specificity(y, p, target=BENCHMARK_SPECIFICITY):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= (1 - target) + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_auc_ci(y, p, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            scores.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_history(h1, h2, path):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    auc = h1.history["auc"] + h2.history["auc"]
    val_a = h1.history["val_auc"] + h2.history["val_auc"]
    ep = range(1, len(acc) + 1)
    split = len(h1.history["accuracy"]) + 0.5
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (tr, va, title) in zip(ax, [(acc, val_acc, "Accuracy"), (loss, val_loss, "Binary cross-entropy loss"), (auc, val_a, "AUC")]):
        a.plot(ep, tr, "b-o", ms=3, label="Train")
        a.plot(ep, va, "r-o", ms=3, label="Validation")
        a.axvline(split, color="grey", ls="--", label="Fine-tuning starts")
        a.set_title(title)
        a.set_xlabel("Epoch")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Glaucoma Detection - Training History (MobileNetV2, two-phase transfer learning)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, p, auc, ci, se85, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color="#1F3864", lw=2, label=f"MobileNetV2 + CLAHE (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    ax.axvline(1 - BENCHMARK_SPECIFICITY, color="#B7791F", ls=":", lw=1.2, label=f"Specificity 0.85: sensitivity {se85:.3f}")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC curve on the test set\n95 % bootstrap CI for AUC: {ci[0]:.3f} - {ci[1]:.3f}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(m, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted glaucoma"])
    ax.set_yticks([0, 1], ["Actual normal", "Actual glaucoma"])
    ax.set_title(f"Confusion matrix (threshold {m['threshold']:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam_samples(model, items, probs, path, n_per_group=3):
    """Grad-CAM for real test images: correct glaucoma, correct normal, and misclassified examples."""
    labels = np.array([l for _, l in items])
    pred = (probs >= 0.5).astype(int)
    groups = [("Glaucoma, correct", np.where((labels == 1) & (pred == 1))[0]),
              ("Normal, correct", np.where((labels == 0) & (pred == 0))[0]),
              ("Misclassified", np.where(labels != pred)[0])]
    chosen = [(g, i) for g, idx in groups for i in idx[np.argsort(-np.abs(probs[idx] - 0.5))][:n_per_group]]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3.0 * len(chosen)))
    axes = np.atleast_2d(axes)
    for row, (group, i) in enumerate(chosen):
        bgr = read_bgr(items[i][0])
        tensor, enhanced = preprocess_bgr(bgr)
        heat, prob = compute_gradcam(model, tensor[None])
        raw = cv2_rgb_resized(bgr)
        for col, (img, title) in enumerate([(raw, "Original"), (enhanced, "CLAHE"), (overlay_heatmap(raw, heat), "Grad-CAM")]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            axes[row, col].set_title(title if row == 0 else "", fontsize=10)
        truth = "glaucoma" if labels[i] else "normal"
        axes[row, 0].text(-8, IMG_SIZE / 2, f"{group}\ntrue: {truth}\nP(glaucoma) = {prob:.2f}", ha="right", va="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cv2_rgb_resized(bgr):
    import cv2
    return cv2.cvtColor(cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Train and evaluate the glaucoma detector on RIM-ONE DL.")
    ap.add_argument("--split", choices=["random", "hospital"], default="random")
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=30)
    args = ap.parse_args()
    out = os.path.join(BASE_DIR, "models") if args.split == "random" else os.path.join(BASE_DIR, "models", "hospital_split")
    os.makedirs(out, exist_ok=True)
    set_seeds()
    started = time.time()
    log.info("=== Glaucoma Detection - training on RIM-ONE DL (%s partition) ===", args.split)
    log.info("TensorFlow %s | GPU: %s", tf.__version__, tf.config.list_physical_devices("GPU") or "none (CPU)")

    tr, va, te = load_split(args.split)
    log.info("Train %d | Validation %d | Test %d", len(tr), len(va), len(te))
    x_tr, y_tr = to_arrays(tr)
    x_va, y_va = to_arrays(va)
    x_te, y_te = to_arrays(te)
    n_pos, n_neg = y_tr.sum(), len(y_tr) - y_tr.sum()
    class_weight = {0: len(y_tr) / (2 * n_neg), 1: len(y_tr) / (2 * n_pos)}
    log.info("Class weights: %s", {k: round(v, 3) for k, v in class_weight.items()})
    train_ds, val_ds = make_ds(x_tr, y_tr, training=True), make_ds(x_va, y_va)

    model, base = build_model()
    compile_model(model, 1e-3)
    log.info("Phase 1 - training the classification head (backbone frozen) ...")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, class_weight=class_weight, verbose=2,
                   callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
                              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)])
    best_p1, auc_p1 = model.get_weights(), val_auc(model, x_va, y_va)
    log.info("Phase 1 best validation AUC: %.4f", auc_p1)

    log.info("Phase 2 - fine-tuning the top %d backbone layers ...", FINE_TUNE_LAYERS)
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:                      # keep BatchNorm statistics frozen (small dataset)
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-4)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, class_weight=class_weight, verbose=2,
                   callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True),
                              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)])
    auc_p2 = val_auc(model, x_va, y_va)
    log.info("Phase 2 best validation AUC: %.4f", auc_p2)
    if auc_p2 < auc_p1:
        log.info("Fine-tuning did not improve validation AUC; keeping the phase-1 weights.")
        model.set_weights(best_p1)
    kept = "phase 2 (fine-tuned)" if auc_p2 >= auc_p1 else "phase 1 (head only)"
    model_path = os.path.join(out, "glaucoma_model.h5")
    model.save(model_path)
    log.info("Model saved -> %s", model_path)

    # ── final, single evaluation on the untouched test set ────────────────
    probs = model.predict(x_te, batch_size=32, verbose=0).ravel()
    auc = float(roc_auc_score(y_te, probs))
    ci = bootstrap_auc_ci(y_te, probs)
    se85 = sensitivity_at_specificity(y_te, probs)
    m50 = threshold_metrics(y_te, probs, 0.5)

    # Grad-CAM localisation statistic and CPU latency on real test images
    shares, times = [], []
    for k, (path, _) in enumerate(te):
        t0 = time.perf_counter()
        tensor, _ = preprocess_bgr(read_bgr(path))
        heat, _ = compute_gradcam(model, tensor[None])
        times.append(time.perf_counter() - t0)
        shares.append(heatmap_centre_share(heat))
    shares = np.array(shares)
    metrics = {
        "dataset": "RIM-ONE DL", "partition": args.split, "train_images": len(tr), "validation_images": len(va),
        "test_images": len(te), "test_glaucoma": int(y_te.sum()), "test_normal": int(len(y_te) - y_te.sum()),
        "auc": auc, "auc_95ci": ci, "sensitivity_at_specificity_0.85": se85, "at_threshold_0.5": m50,
        "validation_auc_phase1": auc_p1, "validation_auc_phase2": auc_p2, "weights_kept": kept,
        "epochs_phase1": len(h1.history["loss"]), "epochs_phase2": len(h2.history["loss"]),
        "parameters_total": int(model.count_params()),
        "parameters_trainable_phase2": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "gradcam_centre_share_mean": float(shares.mean()),
        "gradcam_centre_share_correct_glaucoma": float(shares[(y_te == 1) & (probs >= 0.5)].mean()) if ((y_te == 1) & (probs >= 0.5)).any() else None,
        "cpu_seconds_per_image_preprocess_predict_gradcam_median": float(np.median(times[1:] if len(times) > 1 else times)),
        "training_minutes": (time.time() - started) / 60, "tensorflow": tf.__version__, "seed": SEED,
    }
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(out, "test_predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "p_glaucoma", "predicted_0.5"])
        for (path, label), p in zip(te, probs):
            w.writerow([os.path.relpath(path, DATA_RAW).replace(os.sep, "/"), label, f"{p:.6f}", int(p >= 0.5)])

    plot_history(h1, h2, os.path.join(out, "training_history.png"))
    plot_roc(y_te, probs, auc, ci, se85, os.path.join(out, "roc_curve.png"))
    plot_confusion(m50, os.path.join(out, "confusion_matrix.png"))
    plot_gradcam_samples(model, te, probs, os.path.join(out, "gradcam_samples.png"))

    log.info("TEST  AUC %.4f (95%% CI %.4f-%.4f) | Se@Sp0.85 %.4f | Acc %.4f | Se %.4f | Sp %.4f (threshold 0.5)",
             auc, ci[0], ci[1], se85, m50["accuracy"], m50["sensitivity"], m50["specificity"])
    log.info("All results written to %s", out)


if __name__ == "__main__":
    main()
