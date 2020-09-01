@echo off
set LOGFILE=batch.log
call :LOG > %LOGFILE%
exit /B

:LOG

sphinx-build -b html source build