set CONDA_EXE=conda
:: if Conda executable is not in the PATH, uncomment and set the executable location below
::set CONDA_EXE=%USERPROFILE%\AppData\Local\Continuum\anaconda3\Library\bin\conda.bat

cd %~dp0
call %CONDA_EXE% remove --name geoapps --all --yes
call %CONDA_EXE% env update --file environment.yml --prune
call %CONDA_EXE% env config vars set KMP_WARNINGS=0 --name geoapps
call %CONDA_EXE% activate geoapps && python -m pip install -e .
pause
cmd /k
