@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
call "!MY_CONDA!" remove --name geoapps --all --yes
call "!MY_CONDA!" create --name geoapps --yes
call "!MY_CONDA!" activate geoapps
call "!MY_CONDA!" install --yes --quiet matplotlib ipython h5py
call "!MY_CONDA!" env update --file environment.yml --prune
call python -m pip install -e . --no-deps
pause
cmd /k
