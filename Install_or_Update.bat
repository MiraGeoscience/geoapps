@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  pause
  exit /B !errorlevel!
)

set PY_VER=3.10

set ENV_NAME=geoapps
set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
set PYTHONUTF8=1

:: try installing libmamba solver in base environment (fail silently)
call !MY_CONDA! install -n base --override-channels -c conda-forge conda-libmamba-solver -y > nul 2>&1 ^
  && set "CONDA_SOLVER=libmamba" ^
  || (call )

call "!MY_CONDA!" activate base ^
  && call conda env create --force -n %ENV_NAME% --file environments\conda-py-%PY_VER%-win-64.lock.yml ^
  && call conda run -n %ENV_NAME% pip install -e . --no-deps

if !errorlevel! neq 0 (
  echo "** ERROR: Installation failed **"
  pause
  exit /B !errorlevel!
)

pause
cmd /k "conda activate %ENV_NAME%"
