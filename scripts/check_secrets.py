"""
scripts/check_secrets.py - Staged-file security check for Stale-Answer Detector.

Inspects staged git files:
1. Disallows staging of .env, .env.direct, .env.bifrost, any .env.* (except .env.example),
   .tavily_cache.db, or any *.db files.
2. Inspects staged diffs for accidental API key patterns (e.g., tvly-*, AIza*, sk-*, bf-vk-*).
3. Automatically unstages any violating file using `git reset HEAD <path>`.
4. Exits with 0 if clean, or 1 if any secret/forbidden file was staged.
"""

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
    re.compile(r"tvly-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z-_]{35}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"bf-vk-[A-Za-z0-9_-]{8,}"),
]


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


def check_staged_content_for_keys(filepath: str) -> list[str]:
    # Check added lines in git diff --cached for this file
    res = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--", filepath],
        capture_output=True,
        text=True,
        check=False,
    )
    violations = []
    for line in res.stdout.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            for pat in KEY_PATTERNS:
                if pat.search(line):
                    violations.append(pat.pattern)
    return violations


def main() -> int:
    staged = get_staged_files()
    if not staged:
        print("[INFO] No staged files found.")
        return 0

    violations_found = False

    for f in staged:
        # Check forbidden names
        if is_forbidden_name(f):
            print(f"[SECURITY ALERT] Forbidden file staged: {f}")
            unstage_file(f)
            print(f"[ACTION] Unstaged: {f}")
            violations_found = True
            continue

        # Check content for exposed keys
        key_hits = check_staged_content_for_keys(f)
        if key_hits:
            print(f"[SECURITY ALERT] File {f} contains potential API key pattern! Unstaging.")
            unstage_file(f)
            print(f"[ACTION] Unstaged: {f}")
            violations_found = True

    if violations_found:
        print("\n[BLOCKED] Commit blocked because sensitive files/keys were detected and unstaged.")
        print("Please review `git status` before trying again.")
        return 1

    print(f"[OK] Safety check passed: {len(staged)} staged file(s) verified clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
