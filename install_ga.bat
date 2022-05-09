@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
call "!MY_CONDA!" remove --name geoapps_ga --all --yes
call "!MY_CONDA!" create --name geoapps_ga --yes
call "!MY_CONDA!" activate geoapps_ga
call "!MY_CONDA!" env update --file environment-GA.yml --prune
call python -m pip install git+https://github.com/MiraGeoscience/simpeg.git@release/geoapps-0.8.0 --no-deps
call python -m pip install git+https://github.com/MiraGeoscience/simpeg.git@v0.9.1.dev1+geoapps.0.6.0 --no-deps
call python -m pip install git+https://github.com/MiraGeoscience/geoh5py.git@release/0.3.0
call python -m pip install git+https://github.com/MiraGeoscience/geoana.git
call python -m pip install "dask[distributed]" --upgrade
call python -m pip install tqdm
call python -m pip install -e . --no-deps
pause
cmd /k
