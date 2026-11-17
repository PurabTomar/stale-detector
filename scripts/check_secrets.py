"""
scripts/check_secrets.py - Secret and credential scanner for Stale-Answer Detector.

Scans the repository, results folders, and git staged files for exposed API keys
and forbidden secret files (.env, *.db, private keys).

Usage:
    python scripts/check_secrets.py [--staged-only]
Exits with 0 if clean, or 1 if any exposed secret or forbidden file is detected.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys

FORBIDDEN_NAME_PATTERNS = [
    ".env",
    ".env.direct",
    ".env.bifrost",
    ".env.*",
    ".tavily_cache.db",
    ".tavily_cache.db*",
    "*.db",
]

ALLOWED_EXCEPTIONS = {
    ".env.example",
}

KEY_PATTERNS = [
    ("Tavily API Key", re.compile(r"\btvly-[A-Za-z0-9_-]{18,}")),
    ("Google Gemini Key", re.compile(r"\bAIza[0-9A-Za-z-_]{35}\b")),
    ("OpenAI / Bifrost Key", re.compile(r"\bsk-[A-Za-z0-9_-]{18,}")),
    ("Google OAuth / Token", re.compile(r"\bAQ\.[0-9A-Za-z-_]{20,}")),
    ("Bifrost Virtual Key", re.compile(r"\bbf-vk-[A-Za-z0-9_-]{8,}")),
]

# Known harmless mock/test placeholders used in documentation, tests, and CI
KNOWN_MOCKS = {
    "tvly-mock-",
    "tvly-your-",
    "tvly-test-",
    "tvly-worker-",
    "tvly-thread-",
    "tvly-cloud-",
    "tvly-1234567890",
    "sk-bf-stale-detector-",
    "sk-bf-test-",
    "sk-1234567890",
    "sk-and-all-other",
    "AIza-thread-",
    "mock-ci-key",
    "your_gemini_api_key",
    "your-tavily-api-key",
}

IGNORED_DIRS = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
}


def get_staged_files() -> list[str]:
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return []
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def is_forbidden_name(filepath: str) -> bool:
    basename = os.path.basename(filepath)
    if basename in ALLOWED_EXCEPTIONS:
        return False
    for pat in FORBIDDEN_NAME_PATTERNS:
        if fnmatch.fnmatch(basename, pat) or fnmatch.fnmatch(filepath, pat):
            return True
    return False


def unstage_file(filepath: str) -> None:
    subprocess.run(["git", "reset", "HEAD", "--", filepath], capture_output=True, check=False)


def scan_file_for_secrets(filepath: str) -> list[tuple[int, str, str]]:
    """Scan a single file for exposed keys. Returns list of (line_no, rule_name, masked_snippet)."""
    hits = []
    # Skip test files containing synthetic test fixtures
    basename = os.path.basename(filepath)
    if basename.startswith("test_") or "/tests/" in filepath.replace("\\", "/"):
        return []

    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, 1):
                # Skip comments explaining regexes
                if "re.compile" in line or line.strip().startswith("#"):
                    continue
                for rule_name, pat in KEY_PATTERNS:
                    match = pat.search(line)
                    if match:
                        val = match.group(0)
                        if any(mock in val for mock in KNOWN_MOCKS):
                            continue
                        masked = val[:6] + "..." + val[-2:] if len(val) > 8 else "***"
                        hits.append((line_no, rule_name, masked))
    except Exception:
        pass
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Check repository and staged files for exposed secrets.")
    parser.add_argument("--staged-only", action="store_true", help="Only check git staged files.")
    args = parser.parse_args()

    violations_found = False

    # 1. Check staged files if git is available
    staged = get_staged_files()
    if staged:
        print(f"[check_secrets] Checking {len(staged)} staged file(s)...")
        for f in staged:
            if is_forbidden_name(f):
                print(f"[SECURITY ALERT] Forbidden file staged: {f}")
                unstage_file(f)
                print(f"[ACTION] Unstaged: {f}")
                violations_found = True
                continue
            hits = scan_file_for_secrets(f)
            if hits:
                for line_no, rule, masked in hits:
                    print(f"[SECURITY ALERT] {f}:{line_no} matches {rule} ({masked})! Unstaging.")
                unstage_file(f)
                violations_found = True

    # 2. Check full repo and results folders unless --staged-only
    if not args.staged_only:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print(f"[check_secrets] Scanning repository and results folders under {root_dir}...")
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for fn in filenames:
                full_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(full_path, root_dir)
                if is_forbidden_name(rel_path):
                    # Check if file is tracked by git
                    res = subprocess.run(["git", "ls-files", rel_path], capture_output=True, text=True, check=False)
                    if res.stdout.strip():
                        print(f"[SECURITY ALERT] Forbidden file is tracked in git: {rel_path}")
                        violations_found = True
                    continue
                if fn.endswith((".py", ".json", ".md", ".yml", ".yaml", ".toml", ".txt", ".csv")):
                    hits = scan_file_for_secrets(full_path)
                    if hits:
                        for line_no, rule, masked in hits:
                            print(f"[SECURITY ALERT] Potential secret in {rel_path}:{line_no} [{rule}]: {masked}")
                        violations_found = True

    if violations_found:
        print("\n[BLOCKED] Secrets check failed: exposed key patterns or forbidden files detected.")
        return 1

    print("[OK] Zero secrets detected across repository and results folders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
