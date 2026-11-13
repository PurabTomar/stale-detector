"""
test_theme.py - Unit tests for Light and Dark mode appearance switch and styling.
"""

import os
import re
import unittest
from streamlit.testing.v1 import AppTest

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(WORKSPACE, "app.py")


class TestThemeSwitch(unittest.TestCase):
    def test_css_braces_and_tokens_balanced(self):
        """Verify CUSTOM_CSS and LIGHT_THEME_CSS have strictly balanced braces and key tokens."""
        with open(APP_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        custom_css_match = re.search(r'CUSTOM_CSS = """(.*?)"""', content, re.DOTALL)
        self.assertIsNotNone(custom_css_match, "CUSTOM_CSS block not found in app.py")
        custom_css = custom_css_match.group(1)
        self.assertEqual(
            custom_css.count("{"),
            custom_css.count("}"),
            f"Unbalanced braces in CUSTOM_CSS: {custom_css.count('{')} vs {custom_css.count('}')}",
        )

        light_css_match = re.search(r'LIGHT_THEME_CSS = """(.*?)"""', content, re.DOTALL)
        self.assertIsNotNone(light_css_match, "LIGHT_THEME_CSS block not found in app.py")
        light_css = light_css_match.group(1)
        self.assertEqual(
            light_css.count("{"),
            light_css.count("}"),
            f"Unbalanced braces in LIGHT_THEME_CSS: {light_css.count('{')} vs {light_css.count('}')}",
        )

        required_selectors = [
            ":root",
            "color-scheme: light !important;",
            ".stApp",
            'section[data-testid="stSidebar"]',
            ".sad-hero",
            ".sad-score-card",
            ".sad-claim-card",
            ".sad-compare-box",
            ".sad-source-chip",
            ".sad-snippet-card",
            ".sad-empty-state",
            ".sad-diff-card",
            ".pill-fresh",
            ".pill-outdated",
            ".pill-uncertain",
            ".pill-unverifiable",
            'div.stButton > button[data-testid="stBaseButton-secondary"]',
            "div.stDownloadButton > button",
        ]
        for sel in required_selectors:
            self.assertIn(sel, light_css, f"Missing required selector/rule in LIGHT_THEME_CSS: {sel}")

    def test_apptest_theme_toggle_interaction(self):
        """Verify AppTest starts with dark mode ON, toggles to light mode, and toggles back."""
        at = AppTest.from_file(APP_PATH, default_timeout=30).run()
        self.assertFalse(at.exception, f"App startup exception: {at.exception}")

        toggles = [t for t in at.sidebar.toggle if t.key == "dark_mode"]
        self.assertEqual(len(toggles), 1, "Expected exactly 1 dark_mode toggle in sidebar")
        dark_toggle = toggles[0]
        self.assertEqual(dark_toggle.label, "Dark mode")
        self.assertTrue(dark_toggle.value, "Default dark_mode value should be True")
        self.assertTrue(at.session_state["dark_mode"])

        # Switch to Light mode
        dark_toggle.set_value(False).run()
        self.assertFalse(at.exception, f"Exception after switching to Light mode: {at.exception}")
        self.assertFalse(at.session_state["dark_mode"], "Session state dark_mode should be False in Light mode")

        # Switch back to Dark mode
        dark_toggle.set_value(True).run()
        self.assertFalse(at.exception, f"Exception after switching back to Dark mode: {at.exception}")
        self.assertTrue(at.session_state["dark_mode"], "Session state dark_mode should be True in Dark mode")


if __name__ == "__main__":
    unittest.main()
