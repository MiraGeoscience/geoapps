@echo off
setlocal EnableDelayedExpansion

call %~dp0get_conda_exec.bat
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

cd %~dp0
call !MY_CONDA_EXE! remove --name geoapps --all --yes
call !MY_CONDA_EXE! env update --file environment.yml --prune
call !MY_CONDA_EXE! env config vars set KMP_WARNINGS=0 --name geoapps
call !MY_CONDA_EXE! activate geoapps && python -m pip install -e .
pause
cmd /k
