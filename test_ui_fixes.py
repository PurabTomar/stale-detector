"""
test_ui_fixes.py - Unit tests for recent visual and verdict alignment fixes:
1. Model & Fallback row rendering (chips, arrow, no glued text).
2. Status metric top-border CSS selectors (all 6 distinct status colors).
3. Verdict sentence & tier alignment for 'incorrect' claims.
"""

import os
import unittest
from app import (
    _build_compact_model_row_html,
    get_overall_tier,
    build_overall_verdict_sentence,
    render_score_ring_html,
)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(WORKSPACE, "assets", "style.css")


class TestUiFixes(unittest.TestCase):
    def test_model_row_renders_separate_chips_and_fallback_arrow(self):
        """Ensure models are separate chips, spaced, and fallback has an arrow."""
        models = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
        fallbacks = ["gemini-3.1-flash-lite -> gemini-3.5-flash"]

        html = _build_compact_model_row_html(models, fallback_routes=fallbacks)

        # Chips must have separate spans
        self.assertIn('<span class="sad-model-tag">gemini-3.1-flash-lite</span>', html)
        self.assertIn('<span class="sad-model-tag">gemini-3.5-flash</span>', html)
        # Text must not be glued together
        self.assertNotIn("gemini-3.1-flash-litegemini-3.5-flash", html)

        # Fallback chip with arrow
        self.assertIn('class="sad-fallback-pill"', html)
        self.assertIn('<span class="sad-fallback-arrow">↳</span>', html)
        self.assertIn("gemini-3.1-flash-lite → gemini-3.5-flash", html)

    def test_verdict_sentence_matches_incorrect_claim(self):
        """Ensure single incorrect claim mentions contradicted by official sources."""
        results = [
            {
                "claim": "The Prime Minister of India is John Doe",
                "status": "incorrect",
                "score": 0,
            }
        ]
        sentence = build_overall_verdict_sentence(results)
        self.assertIn("1 incorrect claim found", sentence)
        self.assertIn("contradicted by official sources", sentence)

    def test_overall_tier_returns_incorrect_for_never_true_claim(self):
        """Ensure score < 40 with incorrect claims returns Incorrect and contradicted description."""
        results = [
            {
                "claim": "The Prime Minister of India is John Doe",
                "status": "incorrect",
                "score": 0,
            }
        ]
        tier_label, ring_color, tier_desc = get_overall_tier(0, results=results)
        self.assertEqual(tier_label, "Incorrect")
        self.assertEqual(ring_color, "#DC2626")
        self.assertIn("Contradicted by official sources", tier_desc)

    def test_overall_tier_returns_stale_for_outdated_claim(self):
        """Ensure score < 40 with only outdated claims returns Stale."""
        results = [
            {
                "claim": "GPT-4 was released yesterday",
                "status": "outdated",
                "score": 15,
            }
        ]
        tier_label, ring_color, tier_desc = get_overall_tier(15, results=results)
        self.assertEqual(tier_label, "Stale")
        self.assertIn("outdated", tier_desc.lower())

    def test_css_contains_all_six_status_colors_and_model_chip_styles(self):
        """Ensure assets/style.css has proper rules for metric cards and model tags."""
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()

        # Model and Fallback rules
        self.assertIn(".sad-model-row", css)
        self.assertIn(".sad-model-tag", css)
        self.assertIn(".sad-fallback-pill", css)
        self.assertIn(".sad-fallback-arrow", css)

        # Metric borders for all 6 statuses
        self.assertIn("var(--status-fresh-text)", css)
        self.assertIn("var(--status-outdated-text)", css)
        self.assertIn("var(--status-incorrect-text)", css)
        self.assertIn("var(--status-partial-text)", css)
        self.assertIn("var(--status-uncertain-text)", css)
        self.assertIn("var(--status-unverifiable-text)", css)

        # Ensure the old buggy blanket selector is completely removed
        self.assertNotIn(
            'div[data-testid="column"]:nth-child(2) [data-testid="stMetric"]::before {',
            css,
        )


if __name__ == "__main__":
    unittest.main()
