@echo off

title=Сƻ��webui
SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=������һ��>=3.10.6�汾��python·���������ǰ���������Ѿ����㣬ֱ�ӻس�:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo [��ʼ��] ���ڴ������⻷��...
    echo [��Ϣ] ����Դ�Ƽ�aliyun
    %PYTHON% -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    @rem [��ʼ��] ��������·������ǰĿ¼��...
    set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
    echo [��ʼ��] ���ڰ�װ����...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [��Ϣ] ��ɣ����ٴ�����
    pause
) else (
    echo [�Լ�] ��⵽���⻷��
    SET PYTHON=python
)

@rem [�Լ�] ��������·������ǰĿ¼��...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

echo [�Լ�] �������⻷��...
call %VENV_NAME%\Scripts\activate.bat
echo [�Լ�] �Զ�����...
git pull
echo [�Լ�] ����webui...
%PYTHON% webui.py %*
pause
exit /b
