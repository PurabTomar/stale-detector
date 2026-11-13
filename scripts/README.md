# Automation & Helper Scripts

### run_direct.bat
Switches the active configuration by copying `.env.direct` to `.env`.
Ensures the app runs in Direct Gemini mode without requiring a local gateway.
Validates that `.env.direct` exists in the repository root before copying.
Launches the Streamlit application using `python -m streamlit run app.py`.
Recommended for quick local verification or cloud deployment testing.

### run_bifrost.bat
Switches the active configuration by copying `.env.bifrost` to `.env`.
Reads the `BIFROST_SETUP_TOKEN` from the ambient environment if configured.
Spawns the Bifrost AI Gateway service in a separate console window.
Starts the main Streamlit application in Bifrost multi-provider mode.
Enables task routing, native cross-provider fallbacks, and outage testing.

### push.bat
Accepts a commit message argument and stages changes with `git add .`.
Runs pre-commit validation using `check_secrets.py` on all staged files.
Automatically unstages any forbidden secret files, databases, or API keys.
Commits verified clean changes using the user-provided commit message.
Pushes the verified commit to `origin main` for automated CI evaluation.

### check_secrets.py
Inspects `git diff --cached --name-only` for sensitive file patterns.
Blocks `.env`, `.env.direct`, `.env.bifrost`, `*.db`, and related secret paths.
Scans staged diff contents for exposed API keys (`tvly-*`, `AIza*`, `sk-*`).
Automatically unstages violating files via `git reset HEAD <file>`.
Returns exit code 0 when all staged changes are clean, or 1 to abort commits.
