@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set env_path=%project_dir%\.conda-env
cd %project_dir%\geoapps
set "PYTHONPATH=%project_dir%;%PYTHONPATH%"
call !MY_CONDA_EXE! run -n %env_path% jupyter notebook Index.ipynb
cmd /k
