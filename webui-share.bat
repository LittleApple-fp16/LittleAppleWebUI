@echo off

title=小苹果webui
SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=请输入一个>=3.10.6版本的python路径，如果当前环境变量已经满足，直接回车:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
    echo 正在创建虚拟环境...
    echo 依赖源推荐aliyun
    %PYTHON% -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    @rem 设置依赖路径到当前目录内...
    set "path %cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
    echo 正在安装依赖...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo 完成，请手动安装pytorch：
    echo 请打开environment.bat运行torch的安装
    echo torch的安装需要良好网络连接
    echo 安装命令 pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pause
) else (
    echo 检测到虚拟环境
    SET PYTHON=python
)

@rem 设置依赖路径到当前目录内...
set "path %cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

echo 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo 自动更新...
git pull
echo 启动webui...
%PYTHON% webui.py --share %*
pause
exit /b
