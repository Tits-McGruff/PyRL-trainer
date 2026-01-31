@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo .venv not found, run install.bat first.
  exit /b 1
)

call ".venv\Scripts\activate.bat"

set "TRAINER=trainer.py"

rem If first arg is a .py file, treat it as the trainer script
if not "%~1"=="" (
  if /I "%~x1"==".py" (
    set "TRAINER=%~1"
    shift
  )
)

rem Collect remaining args after the optional trainer file
set "ARGS="
:collect_args
if "%~1"=="" goto run
set "ARGS=!ARGS! "%~1""
shift
goto collect_args

:run
python "%TRAINER%" %ARGS%
exit /b %errorlevel%
