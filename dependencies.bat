@echo off

@title=依赖安装
SET VENV_NAME=venv
@rem 设置依赖路径到当前目录内...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
echo 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo 正在尝试更新依赖
pip install -r requirements.txt
echo 依赖更新完成
pause
exit /b