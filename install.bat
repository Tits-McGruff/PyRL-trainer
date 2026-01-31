@echo off
setlocal

cd /d "%~dp0"

rem Pick a Python, prefer the Windows launcher if present.
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  set "PY=python"
)

rem Create venv if missing
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv
)

rem Activate venv
call ".venv\Scripts\activate.bat"

rem Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

rem Install dependencies
rem If you have a requirements.txt, it will be used, otherwise we install the known deps.
if exist "requirements.txt" (
  pip install -r requirements.txt
) else (
  rem Core deps for your trainer
  pip install numpy websockets torch

  rem Optional, only needed if you run on Python < 3.11 and want TOML parsing
  pip install toml
)

echo.
echo Done. Virtual environment is in .venv
exit /b 0
