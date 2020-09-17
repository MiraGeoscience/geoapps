set PATH=%PATH%;%USERPROFILE%\AppData\Local\Continuum\anaconda3\Scripts;%USERPROFILE%\anaconda3\Scripts
call activate.bat
conda env update --file environment.yml  --prune
pause
cmd /k
