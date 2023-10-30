@rem 设置依赖路径到当前目录内...
@path=%cd%\venv\scripts;%cd%\venv\dep\python;%cd%\venv\dep\python\scripts;%cd%\venv\dep\git\bin;%cd%;%path%

@rem 为中文路径切换UNICODE模式...
@chcp 65001>nul
@if not defined PROMPT set PROMPT=$P$G
@set PROMPT=(venv) %PROMPT%
@cmd /k
