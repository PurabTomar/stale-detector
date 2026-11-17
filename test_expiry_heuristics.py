"""
test_expiry_heuristics.py - Unit tests for category-based heuristic change cycle estimation.

Run with:
    python -m unittest test_expiry_heuristics.py -v
"""

from __future__ import annotations

from datetime import datetime
import unittest

from expiry_heuristics import get_likely_expiry_info, load_expiry_rules


class TestExpiryHeuristics(unittest.TestCase):
    def test_rules_file_loads_correctly(self):
        rules = load_expiry_rules()
        self.assertIn("rules", rules)
        self.assertIn("price", rules["rules"])
        self.assertIn("regulation", rules["rules"])
        self.assertIn("stable_fact", rules["rules"])

    def test_price_cycle_is_3_months(self):
        base = datetime(2026, 11, 15)
        info = get_likely_expiry_info(category="price", check_date=base)
        self.assertEqual(info["typical_cycle"], "3 months")
        self.assertEqual(info["days_until_review"], 90)
        self.assertEqual(info["heuristic_tag"], "Typical change cycle: price (heuristic)")
        self.assertIn("heuristic, not a prediction", info["review_label"])
        self.assertEqual(info["estimated_review_date"], "Feb 13, 2027")

    def test_leadership_cycle(self):
        info = get_likely_expiry_info(category="company", volatility_class="leadership")
        self.assertEqual(info["volatility_class"], "leadership")
        self.assertIn("12 months", info["typical_cycle"])
        self.assertEqual(info["days_until_review"], 365)
        self.assertEqual(info["heuristic_tag"], "Typical change cycle: leadership (heuristic)")

    def test_stable_fact_has_none_cycle(self):
        info = get_likely_expiry_info(category="other", volatility_class="stable_fact")
        self.assertEqual(info["typical_cycle"], "none")
        self.assertIsNone(info["days_until_review"])
        self.assertEqual(info["estimated_review_date"], "None (Stable fact)")
        self.assertIn("No routine review required", info["review_label"])


if __name__ == "__main__":
    unittest.main()
