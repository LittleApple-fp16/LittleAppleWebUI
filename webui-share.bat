@echo off

title=Сƻ��webui
SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=������һ��>=3.10.6�汾��python·���������ǰ���������Ѿ����㣬ֱ�ӻس�:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo ���ڴ������⻷��...
    echo ����Դ�Ƽ�aliyun
    %PYTHON% -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    @rem ��������·������ǰĿ¼��...
    set "path %cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
    echo ���ڰ�װ����...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo ��ɣ����ֶ���װpytorch��
    echo ���environment.bat����torch�İ�װ
    echo torch�İ�װ��Ҫ������������
    echo ��װ���� pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pause
) else (
    echo ��⵽���⻷��
    SET PYTHON=python
)

@rem ��������·������ǰĿ¼��...
set "path %cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

echo �������⻷��...
call %VENV_NAME%\Scripts\activate.bat
echo �Զ�����...
git pull
echo ����webui...
%PYTHON% webui.py --share %*
pause
exit /b
