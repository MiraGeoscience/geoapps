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
call "!MY_CONDA!" activate

IF "%MY_CONDA:mamba.bat=%mamba.bat"=="%MY_CONDA%" (
  set pkg_mgr_exe=mamba
  echo "use mamba"
) ELSE (
  set pkg_mgr_exe=conda
  echo "use conda"
)

call !pkg_mgr_exe! env create --force -n %ENV_NAME% --file environments\conda-py-%PY_VER%-win-64.lock.yml ^
  && call !pkg_mgr_exe! run -n %ENV_NAME% pip install -e . --no-deps

if !errorlevel! neq 0 (
  echo "** ERROR: Installation failed **"
  pause
  exit /B !errorlevel!
)

pause
cmd /k "!pkg_mgr_exe! activate %ENV_NAME%"
