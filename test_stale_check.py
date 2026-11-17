"""
test_stale_check.py - Offline mocked unit tests for stale_check CLI package.

Run with:
    python -m unittest test_stale_check.py -v
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from stale_check.__main__ import main as cli_main
from stale_check.checker import (
    audit_markdown_document,
    find_line_number,
    format_json_report,
    format_markdown_report,
)


class TestStaleCheck(unittest.TestCase):
    def test_find_line_number(self):
        doc = (
            "# Title\n"
            "Introduction line.\n"
            "\n"
            "The iPhone 15 was released in 2023.\n"
            "Another line here.\n"
        )
        self.assertEqual(find_line_number(doc, "iPhone 15"), 4)
        self.assertEqual(find_line_number(doc, "Introduction line"), 2)
        self.assertEqual(find_line_number(doc, "Non-existent phrase"), 1)

    @patch("stale_check.checker.verify_claim")
    @patch("stale_check.checker.extract_claims")
    def test_audit_markdown_document_mocked(self, mock_extract, mock_verify):
        mock_extract.return_value = (
            [
                {
                    "claim": "iPhone 15 starts at $799",
                    "span": "iPhone 15 starts at $799",
                    "category": "price",
                    "search_query": "iPhone 15 price",
                }
            ],
            "mock-gemini-flash",
        )
        mock_verify.return_value = {
            "claim": "iPhone 15 starts at $799",
            "status": "outdated",
            "score": 25,
            "current_info": "iPhone 18 is current model",
            "sources": [{"title": "Apple", "url": "https://apple.com"}],
        }

        doc = "# Pricing\n\nThe iPhone 15 starts at $799.\n"
        result = audit_markdown_document(doc, max_claims=5, filename="test.md")

        self.assertEqual(result["filename"], "test.md")
        self.assertEqual(result["outdated_count"], 1)
        self.assertEqual(result["fresh_count"], 0)
        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(result["claims"][0]["line_number"], 3)

        # Test Markdown formatting
        md_report = format_markdown_report(result)
        self.assertIn("# Stale-Answer Detector: Document Freshness Audit", md_report)
        self.assertIn("L3", md_report)
        self.assertIn("OUTDATED", md_report)

        # Test JSON formatting
        json_report = format_json_report(result)
        data = json.loads(json_report)
        self.assertEqual(data["outdated_count"], 1)

    @patch("stale_check.__main__.audit_markdown_document")
    def test_cli_main_returns_1_when_outdated_claims_found(self, mock_audit):
        mock_audit.return_value = {
            "filename": "sample.md",
            "overall_score": 40,
            "claims": [{"claim": "stale fact", "status": "outdated", "line_number": 2}],
            "fresh_count": 0,
            "outdated_count": 1,
            "unverifiable_count": 0,
        }

        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# Sample Doc\nSome stale fact here.\n")
            tmp_path = f.name

        try:
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                ret = cli_main([tmp_path, "--format", "json"])
                self.assertEqual(ret, 1)
                self.assertIn("stale fact", fake_out.getvalue())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_cli_main_returns_2_for_missing_file(self):
        with patch("sys.stderr", new=io.StringIO()) as fake_err:
            ret = cli_main(["non_existent_file_xyz_123.md"])
            self.assertEqual(ret, 2)
            self.assertIn("File not found", fake_err.getvalue())


if __name__ == "__main__":
    unittest.main()
