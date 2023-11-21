@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set PY_VER=3.10

set ENV_NAME=geoapps
set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
set PYTHONUTF8=1

:: all dependencies are installed from conda
set PIP_NO_DEPS=1

set MY_CONDA_ENV_FILE=environments\conda-py-%PY_VER%-win-64.lock.yml
if not exist %MY_CONDA_ENV_FILE% (
  echo "** ERROR: Could not find the conda environment specification file '%MY_CONDA_ENV_FILE%' **"
  pause
  exit /B 0
)

call "!MY_CONDA!" activate base ^
  && call "!MY_CONDA!" env create --force -n %ENV_NAME% --file %MY_CONDA_ENV_FILE% ^
  && call "!MY_CONDA!" run -n %ENV_NAME% pip install -e .[ui]

if !errorlevel! neq 0 (
  echo "** ERROR: Installation failed **"
  pause
  exit /B !errorlevel!
)

pause
cmd /k "!MY_CONDA!" activate base
