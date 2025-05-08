@echo off
SET SCRIPT_DIR=%~dp0
python -m pip install --upgrade pip
pip install -r "%SCRIPT_DIR%requirements.txt"
echo Libraries installed.
pause