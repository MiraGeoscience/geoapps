set PATH=%PATH%;%USERPROFILE%\anaconda3\Scripts;
call activate.bat
call conda remove --name geoapps --all
call conda env update --file environment.yml  --prune
pause
cmd /k
