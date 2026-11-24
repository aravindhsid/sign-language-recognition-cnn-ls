@echo off
setlocal enabledelayedexpansion

echo =====================================================================
echo    Sign-Language Gesture Recognition - CNN-LSTM Hybrid System
echo =====================================================================
echo.

cd /d "%~dp0"

set PYTHON_CMD=C:\Users\JSidd\anaconda3\python.exe
if not exist "%PYTHON_CMD%" (
    set PYTHON_CMD=python
)

echo [1/4] Checking dependencies...
"%PYTHON_CMD%" -c "import torch, cv2, streamlit; print('All core packages verified.')"
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    "%PYTHON_CMD%" -m pip install -r requirements.txt
)

echo.
echo [2/4] Generating synthetic sign language videos (if not present)...
"%PYTHON_CMD%" dataset.py

echo.
echo [3/4] Training CNN-LSTM Hybrid Model...
"%PYTHON_CMD%" train.py --epochs 15 --batch_size 8

echo.
echo [4/4] Evaluating Model Performance...
"%PYTHON_CMD%" evaluate.py

echo.
echo =====================================================================
echo    Starting Accessibility Communication App (Streamlit)...
echo =====================================================================
"%PYTHON_CMD%" -m streamlit run app.py

pause
