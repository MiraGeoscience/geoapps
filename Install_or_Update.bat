@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
call "!MY_CONDA!" remove --name geoapps --all --yes
call "!MY_CONDA!" env create -f environment.yml
call "!MY_CONDA!" activate geoapps
call python -m pip install -e . --no-deps
pause
cmd /k
