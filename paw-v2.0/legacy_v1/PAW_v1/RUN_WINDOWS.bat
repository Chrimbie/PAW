@echo off
cd /d "%~dp0"
py -3 paw.py
if errorlevel 1 python paw.py
pause
