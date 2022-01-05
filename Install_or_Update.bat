@echo off
setlocal EnableDelayedExpansion

call %~dp0get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

cd %~dp0
call !MY_CONDA_EXE! remove --name geoapps --all --yes
call !MY_CONDA_EXE! create --name geoapps --yes
call !MY_CONDA_EXE! activate geoapps
call !MY_CONDA_EXE! install --yes --quiet numpy scipy matplotlib ipython h5py
call !MY_CONDA_EXE! env update --file environment.yml --prune
call python -m pip install -e . --no-deps
pause
cmd /k
