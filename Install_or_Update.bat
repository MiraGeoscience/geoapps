set PATH=%PATH%;%USERPROFILE%\anaconda3\Scripts;
call activate.bat
call conda remove --name geoapps --all
call conda env update --file environment.yml  --prune
call activate.bat geoapps
call conda env config vars set KMP_WARNINGS=0
pause
cmd /k
