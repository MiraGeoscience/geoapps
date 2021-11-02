@echo off
setlocal EnableDelayedExpansion

set project_dir=%~dp0..
call %project_dir%\get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set env_path=%project_dir%\.conda-env
call !MY_CONDA_EXE! activate %env_path% && pylint %*
