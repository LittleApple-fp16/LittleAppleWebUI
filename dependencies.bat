@echo off

@title=依赖安装
SET VENV_NAME=venv
@rem Setting path...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
echo Activating...
call %VENV_NAME%\Scripts\activate.bat
echo Installing requirements
pip install -r requirements.txt
echo Done.
pause
exit /b