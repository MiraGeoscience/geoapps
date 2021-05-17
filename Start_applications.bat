set PATH=%PATH%;%USERPROFILE%\anaconda3\Scripts;%USERPROFILE%\anaconda3\Library\bin;%USERPROFILE%\anaconda3\envs;%USERPROFILE%\anaconda3\envs\geoapps\Library\bin
set PYTHONPATH=%PYTHONPATH%;%CD%
call activate.bat
call activate geoapps
cd geoapps/applications
jupyter notebook Index.ipynb
cmd /k
