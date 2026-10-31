# Identification and Analysis of Glaucoma Using Advanced Deep Learning Techniques

MobileNetV2 transfer learning + OpenCV CLAHE + Grad-CAM explainability, trained and evaluated on the public
**RIM-ONE DL** retinal image database, with a Streamlit web application.

> B.Tech CSE (Major Project) · Malla Reddy University · Guide: KVN Sreedhar, Assistant Professor
> Team: Gangadhara Rohith (2311CS010193) · Gummadi Jayanth Reddy (2311CS010246) · Kanukanti Sanvi (2311CS010302) · Shilam Goutham (2311CS010622)

---

## 1. Quick start (Windows)

```bat
run_project.bat
```

The script creates a virtual environment in `.venv`, installs the pinned packages, downloads RIM-ONE DL (`fetch_data.py`),
uses the trained model in `models/` (or trains one if it is missing), runs `verify_project.py` and opens the app at
<http://localhost:8501>. To train again from scratch: `run_project.bat retrain` (about 5 minutes on a laptop CPU).

Linux / macOS: `chmod +x run_project.sh && ./run_project.sh`

**Requirements:** Python **3.10–3.12** on PATH, about 2 GB free disk space and an internet connection for the first run
(packages ~600 MB, dataset ~210 MB). No GPU is needed.

### Manual steps (same result)
```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python fetch_data.py        :: downloads and prepares RIM-ONE DL
.venv\Scripts\python train.py             :: optional - models\ already contains the trained model
.venv\Scripts\python verify_project.py    :: self-check (files, data, model, Grad-CAM, metrics)
.venv\Scripts\python -m pytest tests -v   :: 25 automated tests (preprocessing, Grad-CAM, data, model, metrics, app)
.venv\Scripts\python -m streamlit run app.py
```
If the automatic download fails (Google Drive quota / no internet), open <https://bit.ly/rim-one-dl-images>, save the
ZIP as `data\RIM-ONE_DL_images.zip` and run `fetch_data.py` again.

---

## 2. Dataset: RIM-ONE DL

| Item | Value |
|---|---|
| Source | Fumero Batista et al., *Image Analysis & Stereology* 39(3):161–167, 2020, doi:10.5566/ias.2346 — <https://github.com/miag-ull/rim-one-dl> |
| Images | 485 retinal fundus images cropped around the optic nerve head: **313 normal, 172 glaucoma**, labelled by glaucoma experts |
| Hospitals | Hospital Universitario de Canarias, Hospital Universitario Miguel Servet, Hospital Clínico San Carlos (Spain) |
| Official partition used (random) | train 339 (219 normal / 120 glaucoma) · test **146 (94 / 52)** |
| Official partition (by hospital) | train 311 (one hospital) · test **174 (118 / 56)** from the other two hospitals |
| Licence | Free for research and education; **copying and redistribution are prohibited** — so the images are *not* in this ZIP, `fetch_data.py` downloads them from the official link |

`fetch_data.py` writes `data/raw/0_healthy`, `data/raw/1_glaucoma` and `data/splits.csv` (both official partitions).
From each training set, a stratified 15 % is held out for validation; the test set is used **once**, at the end.

---

## 3. Method

```
fundus image ─► CLAHE on L channel (L*a*b*, clip 2.0, 8×8 tiles) ─► 224×224, [0,1]
             ─► Rescaling to [-1,1] ─► MobileNetV2 (ImageNet) ─► 7×7×1280 feature maps
             ─► GlobalAveragePooling ─► Dropout(0.5) ─► BatchNorm ─► Dense(1, sigmoid) = P(glaucoma)
             └► Grad-CAM: α_k = mean(∂y/∂A^k),  heatmap = ReLU(Σ α_k A^k) ─► JET overlay
```

* **Phase 1** – backbone frozen, head trained (Adam 1e-3, ≤ 15 epochs).
* **Phase 2** – top 30 backbone layers unfrozen (BatchNorm statistics kept frozen), Adam 1e-4, ≤ 30 epochs.
* EarlyStopping on validation AUC, ReduceLROnPlateau, class weights for the 313 / 172 imbalance, seed 42.
* Augmentation: horizontal flip, ±8 % brightness, ±10 % contrast (no vertical flip, to keep superior/inferior anatomy).
* The same `utils/preprocessing.py` is used for training and in the app; Grad-CAM lives in `utils/gradcam.py`.

---

## 4. Results (measured; reproduced by `verify_project.py`)

### 4.1 Random partition — `models/`

| Metric (146 test images, never seen in training) | This project | RIM-ONE DL paper, MobileNetV2 |
|---|---|---|
| AUC | **0.938** (95 % bootstrap CI 0.899–0.970) | 0.974 |
| Sensitivity at specificity 0.85 | **0.885** | 0.942 |
| Accuracy (threshold 0.5) | **0.849** | 0.904 |
| Sensitivity / specificity (threshold 0.5) | 0.846 / 0.851 | – |
| Confusion matrix (threshold 0.5) | TP 44 · TN 80 · FP 14 · FN 8 | – |

Training time ≈ 5 min on a laptop CPU · 2,264,385 parameters · ≈ 0.42 s per image on CPU for preprocessing + prediction + Grad-CAM.

### 4.2 By-hospital partition (generalisation) — `models/hospital_split/`

| Metric (174 test images from two *other* hospitals) | This project | RIM-ONE DL paper, MobileNetV2 |
|---|---|---|
| AUC | **0.779** (95 % CI 0.702–0.845) | 0.777 |
| Sensitivity at specificity 0.85 | **0.464** | 0.446 |
| Accuracy / sensitivity / specificity (0.5) | 0.736 / 0.357 / 0.915 | – |

### 4.3 Insights
1. **Transfer learning works with little data** – 288 training images reach AUC 0.94; fine-tuning raised validation AUC
   from 0.865 (head only) to 0.926.
2. **The model overfits quickly** – training AUC reaches ≈ 1.0 in fine-tuning while validation AUC levels off at ≈ 0.92;
   early stopping is essential (see `models/training_history.png`).
3. **Generalisation across hospitals is the main weakness** – AUC drops from 0.94 to 0.78 when the test images come from
   different hospitals/cameras, matching the published benchmark. Validation AUC on the training hospital (0.94) badly
   over-estimates this, so external validation is required before any clinical use.
4. **Grad-CAM points at the optic disc for true glaucoma** – for correctly detected glaucoma, 81 % of the heatmap mass lies
   in the central (optic-disc) region versus 44 % on average; confident normal images produce almost empty heatmaps.
5. **Typical errors are clinically plausible** – the most confident false positives are normal eyes with large, pale
   physiological cups, and for missed glaucoma cases the heatmap is almost empty (`models/gradcam_samples.png`).

---

## 5. Project structure

```
glaucoma-detection/
├── app.py                 Streamlit app (upload / test-set samples, prediction, Grad-CAM, report download)
├── train.py               training + evaluation (writes everything in models/)
├── fetch_data.py          downloads RIM-ONE DL, writes data/raw and data/splits.csv
├── verify_project.py      self-check for evaluators
├── tests/                 25 pytest tests (preprocessing, Grad-CAM, fetch_data, dataset, model/metrics, Streamlit app)
├── utils/preprocessing.py shared CLAHE preprocessing
├── utils/gradcam.py       Grad-CAM with tf.GradientTape
├── models/                trained model + metrics.json, test_predictions.csv, training_history.png,
│   │                      roc_curve.png, confusion_matrix.png, gradcam_samples.png
│   └── hospital_split/    the same outputs for the by-hospital experiment
├── requirements.txt       pinned versions used for the results above
├── run_project.bat / run_project.sh
└── data/                  created by fetch_data.py (not distributed – licence)
```

## 6. Limitations
* RIM-ONE DL images are cropped around the optic disc; full-field fundus photographs should be cropped similarly.
* 485 images from three Spanish hospitals; performance on other populations and cameras is not established (see 4.2).
* Binary output (glaucoma / normal); the model does not measure the cup-to-disc ratio.
* Research and education prototype — **not** a medical device. Diagnosis requires an ophthalmologist.

## 7. References
1. F. J. Fumero Batista *et al.*, "RIM-ONE DL: A unified retinal image database for assessing glaucoma using deep learning," *Image Anal. Stereol.*, 39(3):161–167, 2020.
2. M. Sandler *et al.*, "MobileNetV2: Inverted residuals and linear bottlenecks," CVPR 2018.
3. R. R. Selvaraju *et al.*, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017.
4. K. Zuiderveld, "Contrast limited adaptive histogram equalization," *Graphics Gems IV*, 1994.
5. Y.-C. Tham *et al.*, "Global prevalence of glaucoma and projections of glaucoma burden through 2040," *Ophthalmology*, 2014.
