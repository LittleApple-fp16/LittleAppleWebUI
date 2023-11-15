@echo off

title=强制更新
SET VENV_NAME=venv

@rem 设置依赖路径到当前目录内...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
echo 激活虚拟环境...
call %VENV_NAME%\Scripts\activate.bat
echo 自动更新...
git pull
echo 更新完成!
pause
exit /b