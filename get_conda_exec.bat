@echo off
setlocal EnableDelayedExpansion

set custom_script="%~dp0get_custom_conda.bat"
if exist !custom_script! (
  call !custom_script!
  if !ERRORLEVEL! neq 0 (
    echo ERROR: calling !custom_script! 1>&2
	exit /B !ERRORLEVEL!
  )
  if [!MY_CONDA_EXE!] == [] (
    echo ERROR: MY_CONDA_EXE not set by !custom_script! 1>&2
	exit /B 1
  )
  call "!MY_CONDA_EXE:"=!" --version 2> NUL
  if !ERRORLEVEL! neq 0 (
	echo ERROR: Failed executing Conda: !MY_CONDA_EXE! 1>&2
	echo Check definition of MY_CONDA_EXE in !custom_script!
	exit /B !ERRORLEVEL!
  )
  goto success
)

:: reset error level
call (exit /B 0)

set usual_conda_paths=^
  "%USERPROFILE%\anaconda3";^
  "%USERPROFILE%\miniconda3";^
  "%LOCALAPPDATA%\Continuum\anaconda3";^
  "%LOCALAPPDATA%\Continuum\miniconda3";^
  "%ProgramData%\anaconda3";^
  "%ProgramData%\miniconda3";

set conda_bat_subpath=Library\bin\conda.bat

for %%p in (%usual_conda_paths%) do (
  set base_path=%%p
  set conda_path="!base_path:"=!\%conda_bat_subpath%"
  if exist !conda_path! (
    set MY_CONDA_EXE=!conda_path!
    goto success
  )
)
echo Error: Failed to find conda.bat 1>&2
echo You can define a custom Conda location with in !custom_script!
exit /B 1

:success
  echo Using Conda: !MY_CONDA_EXE!
  endlocal & set MY_CONDA_EXE=%MY_CONDA_EXE%
  exit /B 0
