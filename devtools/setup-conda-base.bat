:: Setup the conda base environment with mamba, conda-lock and conda-pack
::
:: The script has no parameters, and can be executed by double-clicking
:: the .bat file from Windows Explorer.
::
:: Usage: setup-conda-base.bat

@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

call !MY_CONDA_EXE! install -n base -c conda-forge mamba -y
call !MY_CONDA_EXE! activate base
pip install conda-lock[pip_support]

pause
cmd /k
