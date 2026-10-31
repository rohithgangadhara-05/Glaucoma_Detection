@echo off
REM ============================================================================
REM  run_project.bat  --  Glaucoma Detection Deep Learning Project Runner (Windows)
REM  Usage:  run_project.bat            (uses the included trained model)
REM          run_project.bat retrain    (trains the model again, ~30-60 min on CPU)
REM ============================================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================================
echo   GLAUCOMA DETECTION USING DEEP LEARNING WITH GRAD-CAM EXPLAINABILITY
echo ============================================================================
echo.

echo [STEP 0/5] Checking Python ...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10-3.12 is required and must be on PATH. Install it from https://python.org
    pause & exit /b 1
)
python --version

if not exist ".venv\Scripts\python.exe" (
    echo [STEP 1/5] Creating a virtual environment in .venv ...
    python -m venv .venv || (echo [ERROR] Could not create the virtual environment. & pause & exit /b 1)
)
set "PY=.venv\Scripts\python.exe"
echo [STEP 1/5] Installing dependencies from requirements.txt (first run downloads ~600 MB) ...
"%PY%" -m pip install --upgrade pip >nul
"%PY%" -m pip install -r requirements.txt || (echo [ERROR] Package installation failed. & pause & exit /b 1)

echo [STEP 2/5] Preparing the RIM-ONE DL dataset (fetch_data.py) ...
"%PY%" fetch_data.py || (echo [ERROR] Dataset not available. See the message above. & pause & exit /b 1)

if /I "%~1"=="retrain" goto train
if exist "models\glaucoma_model.h5" if exist "models\metrics.json" (
    echo [STEP 3/5] Trained model found in models\ - skipping training. Use "run_project.bat retrain" to train again.
    goto verify
)
:train
echo [STEP 3/5] Training and evaluating the model (train.py) ...
"%PY%" train.py || (echo [ERROR] Training failed. & pause & exit /b 1)

:verify
echo [STEP 4/5] Verifying the project (verify_project.py) ...
"%PY%" verify_project.py || (echo [ERROR] Verification failed. & pause & exit /b 1)

echo [STEP 5/5] Launching the Streamlit app at http://localhost:8501  (Ctrl+C to stop)
"%PY%" -m streamlit run app.py
pause
