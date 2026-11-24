# PowerShell execution script for CNN-LSTM Sign Language Recognition
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$python = "C:\Users\JSidd\anaconda3\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host "   Sign-Language Gesture Recognition - CNN-LSTM Hybrid System" -ForegroundColor Cyan
Write-Host "=====================================================================`n" -ForegroundColor Cyan

Write-Host "[1/4] Verifying dependencies..." -ForegroundColor Yellow
& $python -c "import torch, cv2, streamlit; print('All core packages verified.')"

Write-Host "`n[2/4] Generating gesture dataset samples..." -ForegroundColor Yellow
& $python dataset.py

Write-Host "`n[3/4] Training CNN-LSTM Hybrid Model..." -ForegroundColor Yellow
& $python train.py --epochs 15 --batch_size 8

Write-Host "`n[4/4] Evaluating Model Performance..." -ForegroundColor Yellow
& $python evaluate.py

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host "   Launching Accessibility Communication App (Streamlit)..." -ForegroundColor Green
Write-Host "=====================================================================`n" -ForegroundColor Green
& $python -m streamlit run app.py
