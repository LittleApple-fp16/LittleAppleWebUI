@echo off

SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=������һ��3.10.6�汾��python·���������Ļ����Ѿ���3.10.6������python:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo ���ڴ������⻷��...
    echo ����Դ�Ƽ�aliyun
    python -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    echo ���ڰ�װ����...
    pip install --upgrade pip
    pip install python==3.10.6
    pip install -r requirements.txt
    echo ���
) else (
    echo ��⵽���⻷��
    SET PYTHON=python
)

echo �������⻷��...
call %VENV_NAME%\Scripts\activate.bat
echo ����webui...
%PYTHON% webui.py %*
pause
exit /b