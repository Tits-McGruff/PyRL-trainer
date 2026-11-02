@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PY="

where py >nul 2>nul
if !errorlevel!==0 (
  rem Prefer Python versions covered by the repository CI matrix.
  rem Python 3.15 is deliberately excluded because PyTorch 2.14 can crash
  rem with a native access violation when CUDA work runs in asyncio worker threads.
  for %%V in (3.14 3.11 3.10) do (
    if not defined PY (
      py -%%V --version >nul 2>nul
      if !errorlevel!==0 set "PY=py -%%V"
    )
  )
) else (
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11), (3, 14)) else 1)" >nul 2>nul
  if !errorlevel!==0 set "PY=python"
)

if not defined PY (
  echo ERROR: No supported Python interpreter was found.
  echo Install Python 3.14, 3.11, or 3.10, then run install.bat again.
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 10), (3, 11), (3, 14)) else 1)" >nul 2>nul
  if not !errorlevel!==0 (
    echo Existing .venv uses an unsupported Python version; recreating with !PY!...
    rmdir /s /q ".venv"
  )
)

if not exist ".venv\Scripts\python.exe" (
  !PY! -m venv .venv
  if not !errorlevel!==0 exit /b !errorlevel!
)

call ".venv\Scripts\activate.bat"
if not !errorlevel!==0 exit /b !errorlevel!

python -m pip install --upgrade pip setuptools wheel
if not !errorlevel!==0 exit /b !errorlevel!

if exist "requirements.txt" (
  pip install -r requirements.txt
  if not !errorlevel!==0 exit /b !errorlevel!
) else (
  pip install numpy websockets tomli tomli-w
  if not !errorlevel!==0 exit /b !errorlevel!
  echo Installing PyTorch with CUDA 12.6 support...
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  if not !errorlevel!==0 exit /b !errorlevel!
)

echo.
echo Done, venv is .venv using:
python --version
exit /b 0
