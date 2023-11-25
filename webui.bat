@echo off

@rem 为中文路径切换UNICODE模式...
@chcp 65001>nul
title=小苹果webui
SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=请输入一个>=3.10.6版本的python路径，如果当前环境变量已经满足，直接回车:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo [初始化] 正在创建虚拟环境...
    echo [信息] 依赖源推荐aliyun
    %PYTHON% -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    @rem [初始化] 设置依赖路径到当前目录内...
    set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
    echo [初始化] 正在安装依赖...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [信息] 完成，请再次启动
    pause
) else (
    echo [自检] 检测到虚拟环境
    SET PYTHON=python
)

@rem [自检] 设置依赖路径到当前目录内...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

echo [自检] 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo [自检] 自动更新...
git pull
echo [自检] 启动webui...
%PYTHON% webui.py %*
pause
exit /b
