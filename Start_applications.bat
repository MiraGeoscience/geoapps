@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set ENV_NAME=geoapps

set MY_CONDA=!MY_CONDA_EXE:"=!
call "!MY_CONDA!" run --live-stream -n %ENV_NAME% python -m geoapps.scripts.start_notebook
cmd /k "!MY_CONDA!" activate %ENV_NAME%
