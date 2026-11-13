@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".env.bifrost" (
    echo [ERROR] .env.bifrost not found in project root.
    exit /b 1
)

echo [INFO] Switching environment: copying .env.bifrost to .env...
copy /Y ".env.bifrost" ".env" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy .env.bifrost to .env.
    exit /b 1
)

echo [INFO] Launching Bifrost AI Gateway in a new window...
if defined BIFROST_SETUP_TOKEN (
    start "Bifrost Gateway" cmd /k "set BIFROST_SETUP_TOKEN=%BIFROST_SETUP_TOKEN% && bifrost"
) else (
    start "Bifrost Gateway" cmd /k "bifrost"
)

echo [INFO] Starting Stale-Answer Detector in Bifrost Gateway mode...
python -m streamlit run app.py
