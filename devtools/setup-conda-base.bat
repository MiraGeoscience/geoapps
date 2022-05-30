@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

call !MY_CONDA_EXE! install -n base -c conda-forge mamba -y
call !MY_CONDA_EXE! activate base
call mamba install -c conda-forge conda-pack -y
call pip install conda-lock[pip_support]

pause
cmd /k
