@echo off

title=LittleAppleWebui
SET VENV_NAME=venv

if not exist %VENV_NAME% (
set /p userinput=Please enter a Python path for version>=3.10.6. If the current environment variable is already satisfied, press Enter directly:
if "%userinput%"=="" (
    set userinput=python
)
SET PYTHON=%userinput%
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
    SET PYTHON=python
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
