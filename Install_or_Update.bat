@echo off
setlocal EnableDelayedExpansion

call "%~dp0get_conda_exec.bat"
if !errorlevel! neq 0 (
  exit /B !errorlevel!
)

set MY_CONDA=!MY_CONDA_EXE:"=!
cd %~dp0
call "!MY_CONDA!" remove --name geoapps --all --yes
call "!MY_CONDA!" create --name geoapps --yes
call "!MY_CONDA!" activate geoapps
call "!MY_CONDA!" install --yes --quiet numpy scipy matplotlib ipython h5py
call "!MY_CONDA!" env update --file environment.yml --prune
call python -m pip install -e git+https://github.com/MiraGeoscience/simpeg.git@GEOPY-500#egg=simpeg
call python -m pip install git+https://github.com/MiraGeoscience/simpeg.git@v0.9.1.dev1+geoapps.0.6.0
call python -m pip install -e git+https://github.com/MiraGeoscience/geoh5py.git@release/0.3.0#egg=geoh5py
call python -m pip install -e git+https://github.com/MiraGeoscience/geoana.git@GEOPY-500#egg=geoana
call python -m pip install tqdm
call python -m pip install -e . --no-deps
pause
cmd /k
