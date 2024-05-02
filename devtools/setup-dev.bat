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

set PYTHONUTF8=1
set CONDA_CHANNEL_PRIORITY=strict

:: all dependencies are installed from conda
set PIP_NO_DEPS=1

set PY_VER=3.10

set env_path=%project_dir%\.conda-env
call !MY_CONDA_EXE! activate base ^
  && call !MY_CONDA_EXE! env update --solver libmamba -p %env_path% --file %project_dir%\environments\py-%PY_VER%-win-64-dev.conda.lock.yml

if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

if exist %project_dir%\..\geoh5py\ (
  call !MY_CONDA_EXE! run -p %env_path% pip install --upgrade --force-reinstall -e %project_dir%\..\geoh5py
)
if exist %project_dir%\..\param-sweeps\ (
  call !MY_CONDA_EXE! run -p %env_path% pip install --upgrade --force-reinstall -e %project_dir%\..\param-sweeps
)

if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

pause
cmd /k !MY_CONDA_EXE! activate %env_path%
