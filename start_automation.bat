@echo off
REM PerryPicks v3 - Automation Startup (Windows Double-Click)
REM 
REM Double-click this file to start the complete automation system
REM 
REM This will open a new Command Prompt window and start both backend and frontend

REM Get the directory where this file is located
cd /d "%~dp0"

REM Clear screen
cls

REM Print banner
echo.
echo ============================================================
echo.
echo    ╔═════════════════════════════════════════════════════════════╗
echo    ║                                                               ║
echo    ║    🤖 PerryPicks v3 - Automation System 🤖                  ║
echo    ║                                                               ║
echo    ║    Complete social media automation for NBA predictions            ║
echo    ║                                                               ║
echo    ╚═════════════════════════════════════════════════════════════╝
echo.
echo ============================================================
echo.
echo Starting automation system...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    REM Try python3
    python3 --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ❌ Error: Python is not installed or not in PATH
        echo Please install Python 3.8 or later from https://python.org
        echo.
        pause
        exit /b 1
    )
    REM Set python3 as the command
    set PYTHON_CMD=python3
    echo ✅ Python3 found
) else (
    set PYTHON_CMD=python
    echo ✅ Python found
)
echo.

REM Start automation (Python script)
echo Starting automation...
echo.

if exist start_automation.py (
    echo Using Python startup script...
    echo.
    %PYTHON_CMD% start_automation.py
) else (
    echo ❌ Error: start_automation.py not found!
    echo.
    pause
    exit /b 1
)

REM If script exits, keep window open
echo.
echo ============================================================
echo.
echo Automation stopped.
echo Press any key to close this window.
echo.
pause
