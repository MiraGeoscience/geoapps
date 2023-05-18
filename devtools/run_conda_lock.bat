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

:: use test-pypi to get development versions if needed
:: (Set it through the config command as simply setting
::  POETRY_REPOSITORIES_TEST_PYPI_URL seems to have no effect
::  when running conda-lock, despite it appears in poetry config --list)
poetry config repositories.test_pypi https://test.pypi.org/simple/

call !MY_CONDA_EXE! activate && python devtools\run_conda_lock.py

:: do not keep test-pypi repository in the config
poetry config --unset repositories.test_pypi

pause
cmd /k
