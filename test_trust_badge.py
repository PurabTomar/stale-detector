"""
test_trust_badge.py - Unit tests for SVG trust badge generation and Markdown snippets.

Run with:
    python -m unittest test_trust_badge.py -v
"""

from __future__ import annotations

import unittest

from trust_badge import (
    generate_trust_badge_markdown,
    generate_trust_badge_svg,
    get_trust_badge_snapshot_notice,
)


class TestTrustBadge(unittest.TestCase):
    def test_generate_trust_badge_svg_structure(self):
        svg = generate_trust_badge_svg(fresh_count=4, stale_count=1, check_date="Nov 17, 2026")
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn("Checked Nov 17, 2026", svg)
        self.assertIn("4 fresh, 1 stale", svg)
        self.assertIn("<path d=\"M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z\"", svg)

    def test_generate_trust_badge_markdown(self):
        md = generate_trust_badge_markdown(fresh_count=3, stale_count=0, check_date="Nov 17, 2026")
        self.assertIn("[![Freshness Audit: 3 fresh, 0 stale](freshness-badge.svg)]", md)
        self.assertIn("Snapshot in time", md)

    def test_snapshot_notice(self):
        notice = get_trust_badge_snapshot_notice("Nov 17, 2026")
        self.assertIn("Snapshot Notice", notice)
        self.assertIn("as of Nov 17, 2026", notice)
        self.assertIn("always run a fresh audit", notice)


if __name__ == "__main__":
    unittest.main()
