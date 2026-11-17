"""
scripts/pre_push.py - Pre-push verification helper for Stale-Answer Detector.

Runs offline security and test checks before pushing to remote:
1. Scans repository and results folders for exposed secrets with `scripts/check_secrets.py`.
2. Runs the full offline unit test suite (with all network calls mocked).

Usage:
    python scripts/pre_push.py

Note: This helper script does NOT automatically install git hooks into .git/hooks/.
It can be run manually or invoked from custom developer workflows.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 60)
    print("Stale-Answer Detector: Pre-Push Verification")
    print("=" * 60)

    # Step 1: Scan for secrets
    print("\n[Step 1/2] Scanning for exposed secrets and forbidden files...")
    check_script = os.path.join(root_dir, "scripts", "check_secrets.py")
    res_secrets = subprocess.run([sys.executable, check_script], cwd=root_dir)
    if res_secrets.returncode != 0:
        print("\n❌ [FAIL] Secrets scanner detected issues. Aborting pre-push.")
        return 1

    # Step 2: Run offline unit tests
    print("\n[Step 2/2] Running offline unit tests...")
    env = dict(os.environ)
    env["CI"] = "true"
    env["LLM_API_KEY"] = "mock-ci-key"
    env["GEMINI_API_KEY"] = "mock-ci-key"
    env["TAVILY_API_KEY"] = "tvly-mock-ci-key"

    res_tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
        cwd=root_dir,
        env=env,
    )
    if res_tests.returncode != 0:
        print("\n❌ [FAIL] Unit tests failed. Aborting pre-push.")
        return 1

    print("\n" + "=" * 60)
    print("✅ [PASS] Pre-push verification succeeded: 0 secrets, all tests passing.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
