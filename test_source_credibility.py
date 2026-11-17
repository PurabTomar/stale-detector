"""
test_source_credibility.py - Unit tests for source credibility scoring and rule evaluation.

Run with:
    python -m unittest test_source_credibility.py -v
"""

from __future__ import annotations

import unittest

from source_credibility import get_source_credibility, load_credibility_rules, render_credibility_bar_html


class TestSourceCredibility(unittest.TestCase):
    def test_rules_load_successfully(self):
        rules = load_credibility_rules()
        self.assertIn("rules", rules)
        self.assertIn("primary", rules["rules"])
        self.assertIn("news", rules["rules"])
        self.assertIn("blog", rules["rules"])
        self.assertIn("forum", rules["rules"])

    def test_primary_credibility_scoring(self):
        cred = get_source_credibility("Primary")
        self.assertEqual(cred["credibility_score"], 95)
        self.assertEqual(cred["tier_label"], "Primary Official Domain")
        self.assertIn("+35% confidence boost", cred["rule_contribution"])

    def test_news_credibility_scoring(self):
        cred = get_source_credibility("news")
        self.assertEqual(cred["credibility_score"], 80)
        self.assertIn("+20% confidence boost", cred["rule_contribution"])

    def test_blog_credibility_scoring(self):
        cred = get_source_credibility("blog")
        self.assertEqual(cred["credibility_score"], 50)
        self.assertIn("+5% baseline", cred["rule_contribution"])

    def test_forum_credibility_scoring(self):
        cred = get_source_credibility("forum")
        self.assertEqual(cred["credibility_score"], 30)
        self.assertIn("-10% downweight", cred["rule_contribution"])

    def test_render_credibility_bar_html(self):
        cred = get_source_credibility("Primary")
        html = render_credibility_bar_html(cred)
        self.assertIn("95% Credibility", html)
        self.assertIn("Primary Official Domain", html)
        self.assertIn("+35% confidence boost", html)


if __name__ == "__main__":
    unittest.main()
