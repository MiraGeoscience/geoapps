@echo off
setlocal EnableDelayedExpansion

call %~dp0get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

cd %~dp0geoapps\applications
call !MY_CONDA_EXE! activate geoapps && jupyter notebook Index.ipynb
cmd /k
