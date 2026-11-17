"""
test_settings_panel.py - Unit tests for settings resource estimation logic.

Run with:
    python -m unittest test_settings_panel.py -v
"""

from __future__ import annotations

import unittest

from settings_panel import (
    SUPPORTED_CATEGORIES,
    SUPPORTED_LANGUAGES,
    estimate_audit_resources,
)


class TestSettingsPanel(unittest.TestCase):
    def test_constants_defined(self):
        self.assertIn("price", SUPPORTED_CATEGORIES)
        self.assertIn("regulation", SUPPORTED_CATEGORIES)
        self.assertIn("Hindi (हिन्दी)", SUPPORTED_LANGUAGES)
        self.assertIn("Hinglish", SUPPORTED_LANGUAGES)

    def test_estimate_resources_empty_text(self):
        est = estimate_audit_resources("")
        self.assertEqual(est["estimated_claims"], 0)
        self.assertEqual(est["estimated_llm_calls"], 0)
        self.assertEqual(est["estimated_tavily_credits"], 0.0)

    def test_estimate_resources_single_judge(self):
        sample_text = "Apple released iPhone 15 Pro Max at $1199 starting price."
        est = estimate_audit_resources(sample_text, max_claims=4, dual_judge=False)
        self.assertGreaterEqual(est["estimated_claims"], 1)
        self.assertEqual(est["estimated_llm_calls"], 1 + est["estimated_claims"])
        self.assertEqual(est["estimated_tavily_credits"], float(est["estimated_claims"]))

    def test_estimate_resources_dual_judge(self):
        sample_text = "Apple released iPhone 15 Pro Max at $1199 starting price with A17 Pro."
        est = estimate_audit_resources(sample_text, max_claims=4, dual_judge=True)
        # Dual-judge requires 1 extract + (claims * 2) calls
        self.assertEqual(est["estimated_llm_calls"], 1 + (est["estimated_claims"] * 2))

    def test_estimate_resources_url_adds_credit(self):
        sample_text = "https://example.com/article"
        est = estimate_audit_resources(sample_text, is_url=True)
        self.assertGreaterEqual(est["estimated_tavily_credits"], 2.0)  # at least 1 extract + 1 search


if __name__ == "__main__":
    unittest.main()
