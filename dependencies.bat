@echo off

@title=������װ
SET VENV_NAME=venv
@rem ��������·������ǰĿ¼��...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
echo �������⻷��...
call %VENV_NAME%\Scripts\activate.bat
echo ���ڳ��Ը�������
pip install -r requirements.txt
echo �����������
pause
exit /b