@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set ENV_NAME=geoapps

set MY_CONDA=!MY_CONDA_EXE:"=!
call "!MY_CONDA!" run -n %ENV_NAME% start_notebook
cmd /k "!MY_CONDA!" activate %ENV_NAME%
