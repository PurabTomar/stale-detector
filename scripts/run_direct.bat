@echo off
setlocal
cd /d "%~dp0\.."

if not exist ".env.direct" (
    echo [ERROR] .env.direct not found in project root.
    exit /b 1
)

echo [INFO] Switching environment: copying .env.direct to .env...
copy /Y ".env.direct" ".env" >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy .env.direct to .env.
    exit /b 1
)

echo [INFO] Starting Stale-Answer Detector in Direct Gemini mode...
python -m streamlit run app.py
