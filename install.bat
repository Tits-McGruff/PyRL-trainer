@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  rem Try to find Python 3.11 or 3.10 specifically for stability
  py -3.11 --version >nul 2>nul
  if !errorlevel!==0 (
    set "PY=py -3.11"
  ) else (
     py -3.10 --version >nul 2>nul
     if !errorlevel!==0 (
       set "PY=py -3.10"
     ) else (
       set "PY=py -3"
     )
  )
) else (
  set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel

if exist "requirements.txt" (
  pip install -r requirements.txt
) else (
  pip install numpy websockets tomli
  echo Installing PyTorch with CUDA 12.6 support...
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
)

echo.
echo Done, venv is .venv
exit /b 0
