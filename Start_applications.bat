set PATH=%PATH%;%USERPROFILE%\AppData\Local\Continuum\anaconda3\Scripts
call activate.bat
call activate geoapps
cd geoapps/applications
jupyter notebook Index.ipynb
cmd /k
