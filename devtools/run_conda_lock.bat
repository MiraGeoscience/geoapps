:: Creates conda locked environments from pyproject.toml
:: (see run_conda_lock.py for details).
::
:: The script has no parameters, and can be executed by double-clicking
:: the .bat file from Windows Explorer.
::
:: Usage: run_conda_lock.bat

@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
cd %project_dir%
call get_conda_exec.bat
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

call !MY_CONDA_EXE! activate && python devtools\run_conda_lock.py

pause
cmd /k
