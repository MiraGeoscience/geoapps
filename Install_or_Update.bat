@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set PY_VER=3.9

set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
set PYTHONUTF8=1
call "!MY_CONDA!" remove --name geoapps --all --yes
call "!MY_CONDA!" env create -f environments\conda-py-%PY_VER%-win-64.lock.yml -n geoapps
call "!MY_CONDA!" activate geoapps && python -m pip install -e . --no-deps
pause
cmd /k
