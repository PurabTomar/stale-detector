@echo off
setlocal
cd /d "%~dp0\.."

if "%~1"=="" (
    echo [ERROR] Commit message is required.
    echo Usage: push.bat "Your commit message"
    exit /b 1
)

echo [INFO] Staging all tracked and untracked changes...
git add .

echo [INFO] Running staged-file safety check...
python "%~dp0check_secrets.py"
if errorlevel 1 (
    echo [ERROR] Safety check failed! Sensitive files were detected and unstaged. Commit aborted.
    exit /b 1
)

echo [INFO] Committing changes with message: %~1
git commit -m "%~1"
if errorlevel 1 (
    echo [ERROR] Git commit failed or nothing to commit.
    exit /b 1
)

echo [INFO] Pushing changes to remote main branch...
git push origin main
if errorlevel 1 (
    echo [ERROR] Git push failed. Please check network connection or remote status.
    exit /b 1
)

echo [SUCCESS] Changes committed and pushed successfully!
