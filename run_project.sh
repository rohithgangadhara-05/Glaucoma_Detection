#!/usr/bin/env bash
# ==============================================================================
#  run_project.sh  --  Glaucoma Detection Deep Learning Project Runner (Linux / macOS)
#  Usage:  ./run_project.sh            (uses the included trained model)
#          ./run_project.sh retrain    (trains the model again, ~30-60 min on CPU)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "=== GLAUCOMA DETECTION USING DEEP LEARNING WITH GRAD-CAM EXPLAINABILITY ==="
if command -v python3 &>/dev/null; then BASEPY=python3; elif command -v python &>/dev/null; then BASEPY=python; else
  echo "[ERROR] Python 3.10-3.12 is required."; exit 1; fi
$BASEPY --version

if [ ! -x .venv/bin/python ]; then
  echo "[STEP 1/5] Creating a virtual environment in .venv ..."
  $BASEPY -m venv .venv
fi
PY=.venv/bin/python
echo "[STEP 1/5] Installing dependencies (first run downloads ~600 MB) ..."
$PY -m pip install --upgrade pip >/dev/null
$PY -m pip install -r requirements.txt

echo "[STEP 2/5] Preparing the RIM-ONE DL dataset ..."
$PY fetch_data.py

if [ "${1:-}" != "retrain" ] && [ -f models/glaucoma_model.h5 ] && [ -f models/metrics.json ]; then
  echo "[STEP 3/5] Trained model found in models/ - skipping training (use ./run_project.sh retrain to train again)."
else
  echo "[STEP 3/5] Training and evaluating the model ..."
  $PY train.py
fi

echo "[STEP 4/5] Verifying the project ..."
$PY verify_project.py

echo "[STEP 5/5] Launching the Streamlit app at http://localhost:8501 (Ctrl+C to stop)"
$PY -m streamlit run app.py
