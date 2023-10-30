@echo off

SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=请输入一个3.10.6版本的python路径，如果你的环境已经是3.10.6，输入python:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo 正在创建虚拟环境...
    echo 依赖源推荐aliyun
    python -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    echo 正在安装依赖...
    pip install --upgrade pip
    pip install python==3.10.6
    pip install -r requirements.txt
    echo 完成
) else (
    echo 检测到虚拟环境
    SET PYTHON=python
)

echo 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo 启动webui...
%PYTHON% webui.py %*
pause
exit /b