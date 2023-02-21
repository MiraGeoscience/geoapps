:: Setup a local conda dev environment under .conda-dev
::  with all dependencies for the application,
::  and local editable installation of geoh5py.
:: Note: the application itself is not installed in the environment.
::
:: The script has no parameters, and can be executed by double-clicking
:: the .bat file from Windows Explorer.
::
:: Usage: setup-dev.bat

@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set PY_VER=3.10

set env_path=%project_dir%\.conda-env
call !MY_CONDA_EXE! activate base ^
  && call conda env update --solver=libmamba -p %env_path% --file %project_dir%\environments\conda-py-%PY_VER%-win-64-dev.lock.yml

if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

if exist %project_dir%\..\geoh5py\ (
  call conda run -p %env_path% pip install --upgrade --force-reinstall -e %project_dir%\..\geoh5py --no-deps
)
if exist %project_dir%\..\param-sweeps\ (
  call conda run -p %env_path% pip install --upgrade --force-reinstall -e %project_dir%\..\param-sweeps --no-deps
)

if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

pause
cmd /k "conda activate %env_path%"
