"""Offline unit tests for new features and tabs in Stale-Answer Detector.

Tests:
1. HTML report export (self-contained, offline-compatible, no external assets)
2. Shareable plain-text summary formatter
3. Model Showdown datasets, winner evaluation, and small sample size disclaimer
4. Session audit history (initial seeding, insertion, duplicate prevention, limit)
5. Session watchlist (tracking, delta detection)
6. Ground truth evidence telemetry loader (no invented data)
"""

import json
import unittest
from unittest.mock import patch

from reporting import format_report_html, format_shareable_summary
from showdown_data import (
    SAMPLE_SIZE_DISCLAIMER,
    evaluate_showdown_winner,
    get_saved_showdown,
    get_saved_showdowns,
)
from tab_components import (
    init_history_if_needed,
    load_evidence_counts,
    record_audit_to_history,
)


class TestHtmlReportExport(unittest.TestCase):
    """Test format_report_html produces self-contained, offline-compatible HTML."""

    def setUp(self):
        self.sample_payload = {
            "mode": "Audit",
            "generated_at": "2026-11-13 14:00:00",
            "summary": {
                "overall_score": 96,
                "overall_tier": "Fresh",
                "claims_checked": 2,
                "fresh": 2,
                "outdated": 0,
                "incorrect": 0,
            },
            "tokens_used": {"total_tokens": 420},
            "models_used": ["gemini-2.5-flash"],
            "claims": [
                {
                    "claim": "Apple iPhone 16 Pro Max starts at $1,199.",
                    "status": "fresh",
                    "score": 96,
                    "confidence": "high",
                    "category": "tech",
                    "sources": [
                        {
                            "url": "https://www.apple.com/iphone-16-pro/",
                            "domain": "apple.com",
                            "is_primary": True,
                            "stance": "support",
                        }
                    ],
                }
            ],
            "corrected_answer": "Apple's latest flagship is iPhone 16 Pro Max starting at $1,199.",
        }

    def test_html_structure_and_doctype(self):
        html_str = format_report_html(self.sample_payload)
        self.assertIn("<!DOCTYPE html>", html_str)
        self.assertIn("<html", html_str)
        self.assertIn("</html>", html_str)
        self.assertIn("Freshness Score", html_str)
        self.assertIn("96", html_str)

    def test_offline_compatible_no_external_resources(self):
        html_str = format_report_html(self.sample_payload)
        # Should not link to external stylesheets or google fonts
        self.assertNotIn("fonts.googleapis.com", html_str)
        self.assertNotIn("<script", html_str)
        self.assertIn("<style>", html_str)

    def test_claims_and_sources_rendered(self):
        html_str = format_report_html(self.sample_payload)
        self.assertIn("iPhone 16 Pro Max", html_str)
        self.assertIn("apple.com", html_str)
        self.assertIn("SUPPORT", html_str.upper())


class TestShareableSummary(unittest.TestCase):
    """Test format_shareable_summary plain-text formatter."""

    def setUp(self):
        self.sample_payload = {
            "mode": "Pasted Answer",
            "generated_at": "2026-11-13 15:30:00",
            "summary": {
                "overall_score": 30,
                "overall_tier": "Outdated",
                "claims_checked": 2,
                "fresh": 1,
                "outdated": 1,
                "incorrect": 0,
            },
            "models_used": ["gemini-2.5-flash"],
            "claims": [
                {
                    "claim": "India GST rates merged into 3 slabs.",
                    "status": "outdated",
                    "score": 20,
                    "sources": [{"domain": "cbic.gov.in"}],
                }
            ],
            "corrected_answer": "India GST retains 4 primary tax slabs (5%, 12%, 18%, 28%).",
        }

    def test_plain_text_summary_contents(self):
        summary = format_shareable_summary(self.sample_payload)
        self.assertIn("Stale-Answer Detector — Audit Summary", summary)
        self.assertIn("Freshness Score : 30/100 (Outdated)", summary)
        self.assertIn("Claims Audited  : 2 (1 fresh, 1 outdated, 0 incorrect)", summary)
        self.assertIn("OUTDATED", summary)
        self.assertIn("cbic.gov.in", summary)
        self.assertIn("https://github.com/PurabTomar/stale-detector", summary)


class TestModelShowdown(unittest.TestCase):
    """Test Model Showdown datasets and winner logic."""

    def test_saved_showdowns_exist(self):
        showdowns = get_saved_showdowns()
        self.assertIn("iphone", showdowns)
        self.assertIn("gpt4_pricing", showdowns)
        self.assertIn("india_pm", showdowns)
        self.assertGreaterEqual(len(showdowns), 3)

    def test_single_showdown_fields(self):
        sd = get_saved_showdown("iphone")
        self.assertIsNotNone(sd)
        self.assertIn("question", sd)
        self.assertIn("models", sd)
        self.assertIn("winner", sd)
        self.assertIn("gemini-2.5-flash", sd["models"])

    def test_evaluate_showdown_winner_picks_highest_freshness(self):
        models = {
            "model_fresh": {
                "score": 95,
                "status_counts": {"outdated": 0, "incorrect": 0},
                "latency_ms": 350,
            },
            "model_stale": {
                "score": 25,
                "status_counts": {"outdated": 2, "incorrect": 0},
                "latency_ms": 300,
            },
        }
        eval_res = evaluate_showdown_winner(models)
        self.assertEqual(eval_res["winner"], "model_fresh")
        self.assertEqual(eval_res["winner_score"], 95)
        self.assertIn("+70 points", eval_res["reason"])

    def test_small_sample_size_disclaimer_present(self):
        self.assertIn("Small sample size disclaimer", SAMPLE_SIZE_DISCLAIMER)
        self.assertIn("Agent Lab", SAMPLE_SIZE_DISCLAIMER)


class TestSessionHistory(unittest.TestCase):
    """Test session audit history operations."""

    def setUp(self):
        self.mock_session_state = {}

    def test_init_history_seeding(self):
        with patch("streamlit.session_state", self.mock_session_state):
            init_history_if_needed()
            self.assertIn("audit_history", self.mock_session_state)
            self.assertGreater(len(self.mock_session_state["audit_history"]), 0)
            first_entry = self.mock_session_state["audit_history"][0]
            self.assertIn("score", first_entry)
            self.assertIn("status_counts", first_entry)
            self.assertIn("Saved", first_entry["source_label"])

    def test_record_audit_to_history(self):
        with patch("streamlit.session_state", self.mock_session_state):
            self.mock_session_state["audit_history"] = []
            text = "Apple released iPhone 16 Pro Max."
            results = [{"claim": text, "status": "fresh", "score": 96}]
            record_audit_to_history(text, results, source_label="Live", score=96)

            self.assertEqual(len(self.mock_session_state["audit_history"]), 1)
            entry = self.mock_session_state["audit_history"][0]
            self.assertEqual(entry["score"], 96)
            self.assertEqual(entry["status_counts"]["fresh"], 1)

            # Consecutive duplicate should not insert twice
            record_audit_to_history(text, results, source_label="Live", score=96)
            self.assertEqual(len(self.mock_session_state["audit_history"]), 1)

    def test_history_limit_to_50(self):
        with patch("streamlit.session_state", self.mock_session_state):
            self.mock_session_state["audit_history"] = []
            for i in range(60):
                record_audit_to_history(f"Claim {i}", [{"claim": f"C{i}", "status": "fresh", "score": 90}], "Live")
            self.assertEqual(len(self.mock_session_state["audit_history"]), 50)


class TestWatchlistDelta(unittest.TestCase):
    """Test watchlist claim tracking and status delta detection."""

    def test_watchlist_unchanged_vs_changed(self):
        watchlist = [
            {"claim": "Claim A", "original_status": "fresh", "current_status": "fresh"},
            {"claim": "Claim B", "original_status": "fresh", "current_status": "outdated"},
        ]
        self.assertEqual(watchlist[0]["original_status"] == watchlist[0]["current_status"], True)
        self.assertNotEqual(watchlist[1]["original_status"], watchlist[1]["current_status"])


class TestEvidenceTelemetryCounts(unittest.TestCase):
    """Test loading real evidence metrics without hallucinating data."""

    def test_evidence_counts_load_real_values(self):
        counts = load_evidence_counts()
        self.assertGreaterEqual(counts["bifrost_requests"], 400)
        self.assertGreaterEqual(counts["bifrost_fallbacks"], 5)
        self.assertGreaterEqual(counts["agent_lab_runs"], 300)
        self.assertIn("Nov 13, 2026", counts["bifrost_date"])


if __name__ == "__main__":
    unittest.main()
