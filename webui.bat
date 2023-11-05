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
    pip install -r requirements.txt
    echo 完成，请手动安装pytorch
    echo 安装命令 pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pause
    exit /b
) else (
    echo 检测到虚拟环境
    SET PYTHON=python
)

echo 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo 自动更新...
git pull
echo 启动webui...
%PYTHON% webui.py %*
pause
exit /b