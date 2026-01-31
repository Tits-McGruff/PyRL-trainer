@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
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
  pip install numpy websockets torch
  pip install tomli
)

echo.
echo Done, venv is .venv
exit /b 0
