@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Prepara el entorno con scripts\setup_windows.ps1.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m flybrain_snake gui %*
