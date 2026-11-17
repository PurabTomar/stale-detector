"""
stale_check/__main__.py - CLI entrypoint for stale_check package.

Usage:
    python -m stale_check FILE.md [--max-claims 8] [--format md|json]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from settings import MAX_CLAIMS_PER_AUDIT
from stale_check.checker import (
    audit_markdown_document,
    format_json_report,
    format_markdown_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stale_check",
        description="Audit Markdown or documentation files for outdated and stale facts.",
    )
    parser.add_argument(
        "file_path",
        help="Path to the Markdown or text file to audit (e.g. README.md)",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=MAX_CLAIMS_PER_AUDIT,
        help=f"Maximum claims to audit from document (default: {MAX_CLAIMS_PER_AUDIT})",
    )
    parser.add_argument(
        "--format",
        choices=["md", "json"],
        default="md",
        help="Output report format: 'md' (Markdown table) or 'json' (default: md)",
    )

    args = parser.parse_args(argv)

    path = Path(args.file_path)
    if not path.is_file():
        print(f"Error: File not found: {args.file_path}", file=sys.stderr)
        return 2

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1")

    result = audit_markdown_document(
        doc_text=content,
        max_claims=args.max_claims,
        filename=path.name,
    )

    if args.format == "json":
        print(format_json_report(result))
    else:
        print(format_markdown_report(result))

    # Return exit code 1 if stale/outdated claims found, otherwise 0
    if result.get("outdated_count", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
