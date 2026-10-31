"""
app.py
======
Streamlit web application for Glaucoma Detection with Grad-CAM explainability.

  - Upload a retinal fundus photograph (.jpg, .jpeg, .png) or pick a sample from the held-out TEST set.
  - The same preprocessing as training (utils/preprocessing.py): CLAHE on the L channel -> 224 x 224 -> [0, 1].
  - MobileNetV2 transfer-learning model (models/glaucoma_model.h5) gives P(glaucoma).
  - Grad-CAM from scratch with tf.GradientTape (utils/gradcam.py) shows which region drove the score.
  - Three-panel view (original | CLAHE | Grad-CAM), the model's measured test performance, clinical guidance
    and a downloadable three-panel report.
This is a screening aid for research and education, not a diagnostic device.
"""

import csv
import io
import json
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="GlaucomaScan AI | Clinical Decision Support", page_icon="👁️", layout="wide",
                   initial_sidebar_state="expanded")

import tensorflow as tf  # noqa: E402  (imported after set_page_config so the page appears quickly)

from utils.gradcam import compute_gradcam, overlay_heatmap  # noqa: E402
from utils.preprocessing import IMG_SIZE, preprocess_bgr  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "glaucoma_model.h5")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
TEST_PRED_PATH = os.path.join(BASE_DIR, "models", "test_predictions.csv")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")

st.markdown(
    """
    <style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0f4c81; margin-bottom: 0.1rem; letter-spacing: -0.5px; }
    .sub-title { font-size: 1.05rem; color: #4a5568; margin-bottom: 1.6rem; }
    .alert-glaucoma { background: #fff5f5; border-left: 6px solid #e53e3e; color: #9b2c2c; padding: 1.2rem;
                      border-radius: 8px; font-size: 1.25rem; font-weight: 700; margin: 1rem 0; }
    .alert-healthy { background: #f0fff4; border-left: 6px solid #38a169; color: #22543d; padding: 1.2rem;
                     border-radius: 8px; font-size: 1.25rem; font-weight: 700; margin: 1rem 0; }
    .alert-uncertain { background: #fffaf0; border-left: 6px solid #dd6b20; color: #7b341e; padding: 1.2rem;
                       border-radius: 8px; font-size: 1.25rem; font-weight: 700; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained MobileNetV2 model ...")
def load_detection_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_metrics():
    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return None


@st.cache_data
def load_test_samples(max_per_class=6):
    """Sample images come ONLY from the held-out test set, so the demo never shows images the model was trained on."""
    if not os.path.isfile(TEST_PRED_PATH):
        return []
    rows = list(csv.DictReader(open(TEST_PRED_PATH, encoding="utf-8")))
    samples = []
    for label, name in ((0, "Normal"), (1, "Glaucoma")):
        for r in [r for r in rows if int(r["label"]) == label][:max_per_class]:
            path = os.path.join(DATA_RAW, *r["file"].split("/"))
            if os.path.isfile(path):
                samples.append((f"Test set - {name}: {os.path.basename(path)}", path, name))
    return samples


def build_report(original, enhanced, overlay, title, subtitle):
    """Compose the three panels and the result into one PNG report."""
    w = IMG_SIZE
    canvas = Image.new("RGB", (3 * w + 40, w + 120), "white")
    for k, img in enumerate((original, enhanced, overlay)):
        canvas.paste(Image.fromarray(img), (10 + k * (w + 10), 80))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(15, 76, 129), font=font)
    draw.text((10, 30), subtitle, fill=(60, 60, 60), font=font)
    draw.text((10, 50), "Screening aid only - confirmation by an ophthalmologist is required.", fill=(155, 44, 44), font=font)
    for k, label in enumerate(("Original fundus", "CLAHE enhanced", "Grad-CAM overlay")):
        draw.text((10 + k * (w + 10), 80 + w + 8), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def build_sidebar(metrics):
    with st.sidebar:
        st.markdown("## ⚙️ Model Settings")
        threshold = st.slider("Decision threshold for P(glaucoma)", 0.30, 0.85, 0.50, 0.05,
                              help="Results within 0.10 of the threshold are shown as uncertain.")
        st.markdown("---")
        st.markdown("### 📈 Measured test performance")
        if metrics:
            m = metrics["at_threshold_0.5"]
            st.markdown(
                f"- **Dataset**: {metrics['dataset']} ({metrics['partition']} partition)\n"
                f"- **Test images**: {metrics['test_images']} ({metrics['test_glaucoma']} glaucoma, {metrics['test_normal']} normal)\n"
                f"- **AUC**: {metrics['auc']:.3f} (95 % CI {metrics['auc_95ci'][0]:.3f}-{metrics['auc_95ci'][1]:.3f})\n"
                f"- **Sensitivity @ specificity 0.85**: {metrics['sensitivity_at_specificity_0.85']:.3f}\n"
                f"- **At threshold 0.5**: accuracy {m['accuracy']:.3f}, sensitivity {m['sensitivity']:.3f}, "
                f"specificity {m['specificity']:.3f}"
            )
        else:
            st.caption("models/metrics.json not found - run `python train.py`.")
        st.markdown("---")
        st.markdown("### 🧬 Model")
        st.markdown("- **Backbone**: MobileNetV2 (ImageNet)\n- **Input**: 224 × 224 × 3, CLAHE-enhanced\n"
                    "- **Head**: GAP + Dropout(0.5) + BatchNorm + Dense(1, σ)\n- **Explainability**: Grad-CAM (tf.GradientTape)")
        return threshold


def main():
    st.markdown('<div class="main-title">👁️ Glaucoma Detection AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Deep transfer learning with <b>MobileNetV2</b>, <b>OpenCV CLAHE</b> and '
                '<b>Grad-CAM</b> explainability, trained on the public <b>RIM-ONE DL</b> database.</div>',
                unsafe_allow_html=True)
    metrics = load_metrics()
    threshold = build_sidebar(metrics)
    model = load_detection_model()
    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Run `python fetch_data.py` and then `python train.py`.")
        return

    col_upload, col_sample = st.columns([1.5, 1.2])
    with col_upload:
        st.markdown("### 📤 Upload Retinal Fundus Image")
        uploaded = st.file_uploader("Fundus photograph centred on the optic disc (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
    with col_sample:
        st.markdown("### 🧪 Or Select a Test-Set Sample")
        samples = load_test_samples()
        chosen_path = chosen_truth = None
        if samples:
            options = ["-- None (upload your own image) --"] + [s[0] for s in samples]
            choice = st.selectbox("Images the model has never seen during training:", options)
            if choice != options[0]:
                _, chosen_path, chosen_truth = samples[options.index(choice) - 1]
        else:
            st.caption("No test samples found. Run `python fetch_data.py` and `python train.py`.")

    if uploaded is None and not chosen_path:
        st.info("👋 Upload a retinal image or choose a test-set sample to run the analysis.")
        return
    source = uploaded.name if uploaded is not None else os.path.basename(chosen_path)
    try:
        pil = Image.open(uploaded if uploaded is not None else chosen_path).convert("RGB")
    except Exception:  # corrupted file or not really an image
        st.error(f"`{source}` could not be read as an image. Please upload a valid JPG or PNG fundus photograph.")
        return
    if min(pil.size) < 64:
        st.error(f"`{source}` is too small ({pil.size[0]} × {pil.size[1]} px). Please upload a fundus photograph of at least 64 × 64 px.")
        return
    if uploaded is not None:
        chosen_truth = None

    st.markdown("---")
    with st.spinner("Running CLAHE enhancement, MobileNetV2 inference and Grad-CAM ..."):
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        tensor, enhanced = preprocess_bgr(bgr)
        heatmap, prob = compute_gradcam(model, tensor[None], target_class=1)
        original = cv2.resize(np.array(pil), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        overlay = overlay_heatmap(original, heatmap)

    is_glaucoma = prob >= threshold
    uncertain = abs(prob - threshold) < 0.10
    confidence = (prob if is_glaucoma else 1.0 - prob) * 100.0
    st.markdown("## 📊 Screening Result")
    if uncertain:
        st.markdown(f'<div class="alert-uncertain">⚠️ Uncertain result (P(glaucoma) = {prob:.3f}, close to the threshold '
                    f'{threshold:.2f}) - specialist review recommended</div>', unsafe_allow_html=True)
    elif is_glaucoma:
        st.markdown(f'<div class="alert-glaucoma">⚠️ Glaucoma suspected: {confidence:.1f}% '
                    f'<span style="font-size:0.95rem;font-weight:normal">(P(glaucoma) = {prob:.4f})</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-healthy">✅ No glaucoma suspected: {confidence:.1f}% '
                    f'<span style="font-size:0.95rem;font-weight:normal">(P(glaucoma) = {prob:.4f})</span></div>',
                    unsafe_allow_html=True)
    if chosen_truth:
        agree = (chosen_truth == "Glaucoma") == bool(is_glaucoma)
        st.caption(f"Expert label in RIM-ONE DL: **{chosen_truth}** - the model {'agrees' if agree else 'DISAGREES'}.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Output", "Glaucoma" if is_glaucoma else "Normal")
    m2.metric("P(glaucoma)", f"{prob:.3f}")
    m3.metric("Threshold", f"{threshold:.2f}")
    st.caption(f"Analysed image: {source}")

    st.markdown("---")
    st.markdown("### 🔬 Grad-CAM Explainability")
    st.caption("Grad-CAM weights the last 7 × 7 MobileNetV2 feature maps by the gradient of the glaucoma score. Warm colours "
               "(red/yellow) mark the regions that increased the score. Check that they lie on the optic disc and cup.")
    c1, c2, c3 = st.columns(3)
    c1.image(original, caption=f"1. Original: {source}", width="stretch")
    c2.image(enhanced, caption="2. CLAHE enhanced (clipLimit 2.0, 8 × 8 tiles)", width="stretch")
    c3.image(overlay, caption="3. Grad-CAM overlay", width="stretch")
    result_text = ("Uncertain" if uncertain else "Glaucoma suspected" if is_glaucoma else "No glaucoma suspected")
    st.download_button("📥 Download three-panel report (PNG)",
                       data=build_report(original, enhanced, overlay, f"GlaucomaScan AI - {result_text}",
                                         f"Image: {source} | P(glaucoma) = {prob:.4f} | threshold {threshold:.2f}"),
                       file_name=f"glaucoma_report_{os.path.splitext(source)[0]}.png", mime="image/png")

    st.markdown("---")
    st.markdown("### 🩺 Clinical Decision Support")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("#### What the clinician should verify on the image")
        st.info("- **Cup-to-disc ratio (CDR)**: a vertical CDR above about 0.6, or a difference between eyes, is suspicious.\n"
                "- **ISNT rule**: a normal rim is usually thickest Inferiorly, then Superiorly, Nasally and Temporally; "
                "rim thinning that breaks this order is a warning sign.\n"
                "- **Grad-CAM focus**: the warm region should lie on the optic disc / cup. If it lies on the image border or "
                "an artefact, do not trust the result.\n"
                "_The model does not measure the CDR; these are the signs an expert checks._")
    with g2:
        st.markdown("#### Suggested next step")
        if is_glaucoma or uncertain:
            st.warning("1. Refer to an ophthalmologist.\n2. Measure intraocular pressure (tonometry).\n"
                       "3. Optical coherence tomography (RNFL thickness).\n4. Visual-field test (automated perimetry).")
        else:
            st.success("1. Routine eye examination as usual.\n2. Earlier review if there are risk factors "
                       "(family history, high myopia, diabetes, age over 40).")
    st.caption("This tool is a research and education prototype trained on RIM-ONE DL. It is not a medical device.")


if __name__ == "__main__":
    main()
