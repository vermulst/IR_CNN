@echo off
SET SCRIPT_DIR=%~dp0
SET PARENT_DIR=%SCRIPT_DIR%..

REM Normalize path
PUSHD %PARENT_DIR%
SET PARENT_DIR=%CD%
POPD

SET VENV_DIR=%PARENT_DIR%\venv

REM Check if venv exists
IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate the virtual environment
CALL "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r "%SCRIPT_DIR%requirements.txt"

echo Libraries installed in virtual environment at %VENV_DIR%.
pause
