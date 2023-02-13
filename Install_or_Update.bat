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
call "!MY_CONDA!" activate
call conda remove --name %ENV_NAME% --all --yes
call conda env create -f environments\conda-py-%PY_VER%-win-64.lock.yml -n %ENV_NAME%
call conda activate %ENV_NAME% && pip install -e . --no-deps

pause
cmd /k
