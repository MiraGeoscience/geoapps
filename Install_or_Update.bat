set PATH=%PATH%;%USERPROFILE%\anaconda3\Scripts;
call activate.bat
conda env update --file environment.yml  --prune
pause
cmd /k
