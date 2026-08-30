@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set env_path="C:\Users\dominiquef\AppData\Local\miniforge3\envs\pro"
call !MY_CONDA_EXE! run -p %env_path% pylint %*
