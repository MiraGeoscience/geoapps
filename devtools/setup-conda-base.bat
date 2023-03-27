:: Setup the conda base environment with the conda libmamba solver, conda-lock and conda-pack
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
  pause
  exit /B !errorlevel!
)

call !MY_CONDA_EXE! install -n base --override-channels -c conda-forge conda-libmamba-solver -y ^
  && call !MY_CONDA_EXE! run -n base pip install conda-lock[pip_support]

if !errorlevel! neq 0 (
  echo "** ERROR: Installation failed **"
  pause
  exit /B !errorlevel!
)

pause
cmd /k !MY_CONDA_EXE! activate base
