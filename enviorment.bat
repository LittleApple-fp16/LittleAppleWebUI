@SET VENV_NAME=venv
@set "path=%cd%\%VENV_NAME%\scripts;%cd%\%VENV_NAME%\dep\python;%cd%\%VENV_NAME%\dep\python\scripts;%cd%\%VENV_NAME%\dep\git\bin;%cd%;%path%"

@title=Enviorment Commandline
@if not defined PROMPT set PROMPT=$P$G
@set PROMPT=(venv) %PROMPT%
@cmd /k
