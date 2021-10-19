set CONDA_EXE=conda
:: if Conda executable is not in the PATH, uncomment and set the executable location below
::set CONDA_EXE=%USERPROFILE%\AppData\Local\Continuum\anaconda3\Library\bin\conda.bat

set THIS_DIR=%~dp0
set PYTHONPATH=%THIS_DIR%;%PYTHONPATH%
cd %THIS_DIR%\geoapps\applications
call %CONDA_EXE% activate geoapps && jupyter notebook Index.ipynb
cmd /k
