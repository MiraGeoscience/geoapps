@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set env_path=%project_dir%\.conda-env
call !MY_CONDA_EXE! env create -p %env_path% --file %project_dir%\conda-py-3.9-win-64-dev.lock.yml
call !MY_CONDA_EXE! activate %env_path% && pip install --upgrade --force-reinstall -e %project_dir%\..\geoh5py --no-deps
pause
cmd /k
