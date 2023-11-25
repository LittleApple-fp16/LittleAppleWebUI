@echo off

title=LittleAppleWebui
SET VENV_NAME=venv
SET PYTHON=python

if not exist %VENV_NAME% (
    echo [init] Creating venv...
    %PYTHON% -m venv %VENV_NAME%
    call %VENV_NAME%\Scripts\activate.bat
    @rem [init] Setting path...
    set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"
    echo [init] Installing deps...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    echo [info] Done. Please restart.
    pause
) else (
    echo [info] Detected venv
)

@rem [info] Setting path...
set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

echo [info] Activating...
call %VENV_NAME%\Scripts\activate.bat
echo [info] Auto update...
git pull
echo [info] Starting webui...
%PYTHON% webui.py %*
pause
exit /b
