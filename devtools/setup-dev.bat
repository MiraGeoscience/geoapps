@echo off
set project_dir=%~dp0..\
set env_path=%project_dir%\.conda-env
conda env create -p %env_path% python=3.9 --file %project_dir%\environment.yml ^
 & conda install -y -p %env_path% --file %project_dir%\dev-extra-requirements.txt
