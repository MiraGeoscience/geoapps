@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set env_path=%project_dir%\.conda-env
call !MY_CONDA_EXE! env create -p %env_path% python=3.9 --file %project_dir%\environment.yml
call !MY_CONDA_EXE! install -y -p %env_path% --file %project_dir%\dev-extra-requirements.txt
pause
cmd /k
