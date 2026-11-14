"""
scripts/check_evidence_clean.py - Security check verifying evidence/ contains zero secrets or API keys.

Scans all files in `evidence/` (specifically `evidence/bifrost_evidence.json`).
Fails with exit code 1 if any API key, virtual key, secret token, or sensitive header is found.
Exits with 0 if completely clean.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = ROOT_DIR / "evidence"
EVIDENCE_FILE = EVIDENCE_DIR / "bifrost_evidence.json"

KEY_PATTERNS = [
    ("Tavily API Key", re.compile(r"tvly-[A-Za-z0-9_-]{10,}", re.IGNORECASE)),
    ("Google Gemini Key", re.compile(r"AIza[0-9A-Za-z-_]{30,}", re.IGNORECASE)),
    ("OpenAI / General sk- Key", re.compile(r"sk-[A-Za-z0-9_-]{15,}", re.IGNORECASE)),
    ("Bifrost Virtual Key", re.compile(r"bf-vk-[A-Za-z0-9_-]{6,}", re.IGNORECASE)),
    ("Authorization Bearer Header", re.compile(r"Bearer\s+[A-Za-z0-9_.-]{15,}", re.IGNORECASE)),
]


def scan_obj_for_keys(obj: Any, path: str = "root") -> list[str]:
    """Recursively search for sensitive key patterns in any JSON object."""
    violations: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}"
            # Check key name itself
            k_lower = str(k).lower()
            if any(s in k_lower for s in ("virtual_key", "api_key", "apikey")) and v != "[REDACTED]":
                violations.append(f"Sensitive key name '{current_path}' containing unredacted value: {v}")
            violations.extend(scan_obj_for_keys(v, current_path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            violations.extend(scan_obj_for_keys(item, f"{path}[{idx}]"))
    elif isinstance(obj, str):
        for name, regex in KEY_PATTERNS:
            if regex.search(obj):
                violations.append(f"Pattern match '{name}' at {path}: {obj[:40]}...")
    return violations


def check_evidence_directory() -> int:
    """Validate all files in evidence/ directory."""
    if not EVIDENCE_DIR.is_dir():
        print(f"[ERROR] Evidence directory not found at: {EVIDENCE_DIR}")
        return 1

    if not EVIDENCE_FILE.is_file():
        print(f"[ERROR] Evidence file not found at: {EVIDENCE_FILE}")
        return 1

    print(f"[SCAN] Auditing files in {EVIDENCE_DIR} for secrets and API keys...")
    total_files = 0
    all_violations: list[str] = []

    for file_path in EVIDENCE_DIR.glob("*"):
        if file_path.is_file():
            total_files += 1
            print(f"       Inspecting {file_path.name} ({file_path.stat().st_size:,} bytes)...")
            try:
                raw_text = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                print(f"[ERROR] Failed to read {file_path}: {exc}")
                return 1

            # 1. Regex scan raw text
            for name, regex in KEY_PATTERNS:
                matches = regex.findall(raw_text)
                if matches:
                    for m in matches:
                        all_violations.append(f"Exposed key in {file_path.name} matching '{name}': {m[:10]}...")

            # 2. JSON structured scan if JSON
            if file_path.suffix.lower() == ".json":
                try:
                    data = json.loads(raw_text)
                    all_violations.extend(scan_obj_for_keys(data, path=file_path.name))
                except Exception as exc:
                    all_violations.append(f"Malformed JSON in {file_path.name}: {exc}")

    if all_violations:
        print(f"\n[FAILED] Security check failed! Found {len(all_violations)} violation(s):")
        for v in all_violations:
            print(f"  ❌ {v}")
        return 1

    print(f"\n[PASSED] All {total_files} file(s) in evidence/ are clean. Zero secrets, keys, or sensitive headers detected.")
    return 0


if __name__ == "__main__":
    sys.exit(check_evidence_directory())
