@echo off
set project_dir=%~dp0..\
conda env create -p %project_dir%\.conda-env python=3.9 --file %project_dir%\environment.yml --file %project_dir%\environment-dev.yml
