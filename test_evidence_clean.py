"""
test_evidence_clean.py - Unit test verifying evidence/bifrost_evidence.json is clean and structured.
"""

import json
from pathlib import Path
import unittest

from scripts.check_evidence_clean import check_evidence_directory, scan_obj_for_keys
from bifrost_evidence import load_bifrost_evidence, render_bifrost_evidence_panel
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parent
EVIDENCE_FILE = ROOT_DIR / "evidence" / "bifrost_evidence.json"


class TestBifrostEvidence(unittest.TestCase):
    """Test suite for evidence/bifrost_evidence.json and bifrost_evidence.py."""

    def test_evidence_file_exists_and_is_valid_json(self):
        """Ensure evidence/bifrost_evidence.json exists and parses cleanly."""
        self.assertTrue(EVIDENCE_FILE.is_file(), f"Evidence file not found at {EVIDENCE_FILE}")
        data = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        self.assertIn("total_requests", data)
        self.assertGreater(data["total_requests"], 0)
        self.assertIn("fallback_events", data)
        self.assertIn("note", data)
        self.assertIn("recorded_date", data)

    def test_evidence_contains_zero_secrets_or_keys(self):
        """Ensure evidence directory passes strict check_evidence_directory security audit."""
        result = check_evidence_directory()
        self.assertEqual(result, 0, "Security check found keys or secrets in evidence/")

    def test_scan_obj_for_keys_detects_deliberate_key(self):
        """Verify the secret scanner actively catches key patterns if present."""
        bad_sample = {
            "auth": "sk-1234567890abcdef1234567890",
            "nested": {"token": "tvly-1234567890abcdef1234"},
        }
        violations = scan_obj_for_keys(bad_sample)
        self.assertGreater(len(violations), 0)

    def test_load_bifrost_evidence_returns_valid_structure(self):
        """Verify load_bifrost_evidence returns parsed data correctly."""
        ev = load_bifrost_evidence()
        self.assertIsNotNone(ev)
        self.assertIsInstance(ev, dict)
        self.assertEqual(ev.get("recorded_date"), "Nov 13, 2026")
        self.assertEqual(ev.get("total_requests"), 490)
        self.assertEqual(ev.get("total_fallbacks"), 10)
        self.assertIn("models_breakdown", ev)
        self.assertIn("requests", ev)

    @patch("streamlit.markdown")
    @patch("streamlit.metric")
    @patch("streamlit.dataframe")
    @patch("streamlit.expander")
    @patch("streamlit.columns")
    def test_render_bifrost_evidence_panel_runs_cleanly(
        self, mock_columns, mock_expander, mock_dataframe, mock_metric, mock_markdown
    ):
        """Verify render_bifrost_evidence_panel executes with zero exceptions."""
        col_mock = MagicMock()
        mock_columns.return_value = [col_mock, col_mock, col_mock, col_mock]
        mock_expander.return_value.__enter__.return_value = MagicMock()
        mock_expander.return_value.__exit__.return_value = None

        # Call with arch key prefix
        render_bifrost_evidence_panel(key_prefix="test_arch")
        self.assertTrue(mock_markdown.called)
        self.assertTrue(mock_metric.called)


if __name__ == "__main__":
    unittest.main()
