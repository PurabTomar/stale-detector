"""
test_reporting.py - Unit tests for reporting.py (word diff, changelog, Markdown & JSON reports).

Run with:
    python -m unittest test_reporting.py
"""

from datetime import datetime, timezone
import json
import unittest

import reporting


class TestReporting(unittest.TestCase):
    def test_side_by_side_word_diff_marks_red_deletions_and_green_additions(self):
        orig = "Apple's flagship is the iPhone 15 Pro Max starting at $1,199 and runs iOS 17."
        corr = "Apple's flagship is the iPhone 17 Pro Max starting at $1,299 and runs iOS 26."

        diff = reporting.build_side_by_side_word_diff(orig, corr)
        orig_html = diff["original_html"]
        corr_html = diff["corrected_html"]
        stats = diff["stats"]

        # Removed tokens struck through in red on original side; green additions on corrected side.
        self.assertIn('<del class="sad-diff-del">15</del>', orig_html)
        self.assertIn('<del class="sad-diff-del">&#36;1,199</del>', orig_html)
        self.assertIn('<del class="sad-diff-del">17</del>', orig_html)
        self.assertNotIn("sad-diff-ins", orig_html)

        self.assertIn('<ins class="sad-diff-ins">17</ins>', corr_html)
        self.assertIn('<ins class="sad-diff-ins">&#36;1,299</ins>', corr_html)
        self.assertIn('<ins class="sad-diff-ins">26</ins>', corr_html)
        self.assertNotIn("sad-diff-del", corr_html)

        # Currency '$' must be escaped to avoid Streamlit KaTeX rendering bugs.
        self.assertNotIn("$1,199", orig_html)
        self.assertNotIn("$1,299", corr_html)

        self.assertTrue(stats["has_changes"])
        self.assertEqual(stats["removed_words"], 3)
        self.assertEqual(stats["added_words"], 3)
        self.assertGreater(stats["unchanged_words"], 5)

    def test_build_change_log_prioritizes_outdated_then_uncertain_with_sources(self):
        results = [
            {
                "claim": "Sam Altman is CEO of OpenAI.",
                "category": "company",
                "risk_level": "moderate",
                "status": "fresh",
                "score": 95,
                "current_info": "Sam Altman remains CEO.",
                "reason": "Confirmed by OpenAI.",
                "sources": [{"title": "OpenAI", "url": "https://openai.com/about"}],
            },
            {
                "claim": "India GST exemption threshold is Rs 20 lakh everywhere.",
                "category": "regulation",
                "risk_level": "critical",
                "status": "uncertain",
                "score": 55,
                "current_info": "Threshold differs between goods (40 lakh) and services (20 lakh).",
                "reason": "Depends on state and goods vs. services.",
                "sources": [
                    {"title": "GST Portal", "url": "https://www.gst.gov.in/rules"},
                ],
            },
            {
                "claim": "OpenAI reached a valuation of $86 billion.",
                "category": "company",
                "risk_level": "moderate",
                "status": "outdated",
                "score": 15,
                "current_info": "Higher recent valuation rounds reported.",
                "reason": "Newer funding rounds superseded the early 2024 figure.",
                "sources": [
                    {"title": "Reuters", "url": "https://www.reuters.com/tech/openai"},
                    {"title": "Reuters Duplicate", "url": "https://www.reuters.com/tech/openai"},
                    {"title": "Bloomberg", "url": "https://www.bloomberg.com/openai"},
                ],
            },
        ]

        changes = reporting.build_change_log(results)
        # Fresh claim is excluded; outdated comes first, uncertain second.
        self.assertEqual(len(changes), 2)
        self.assertEqual(changes[0]["action"], "Replaced")
        self.assertEqual(changes[0]["claim_index"], 3)
        self.assertEqual(len(changes[0]["sources"]), 2)  # duplicate URL deduplicated
        self.assertEqual(changes[1]["action"], "Qualified")
        self.assertEqual(changes[1]["claim_index"], 2)

    def test_report_payload_markdown_and_json_include_all_required_metadata(self):
        fixed_dt = datetime(2026, 11, 11, 14, 31, 52, tzinfo=timezone.utc)
        results = [
            {
                "claim": "iOS 17 is the latest iOS.",
                "category": "version",
                "search_query": "latest iOS version",
                "status": "outdated",
                "score": 12,
                "confidence": "high",
                "risk_level": "moderate",
                "current_info": "iOS 26 is current.",
                "reason": "Apple released newer iOS versions.",
                "why_this_score": "Score 12/100: Outdated version.",
                "score_breakdown": {"final_score": 12},
                "sources": [{"title": "Apple Support", "url": "https://support.apple.com/ios"}],
                "tavily_credits_used": 2.2,
                "model": "anthropic/claude-3-5-sonnet",
                "primary_model": "openai/gpt-4o-mini",
                "fallback_used": True,
                "gateway_metrics": {
                    "operation": "Verify claim #1",
                    "timestamp": "14:31:45",
                    "primary_model": "openai/gpt-4o-mini",
                    "model_used": "anthropic/claude-3-5-sonnet",
                    "fallback_used": True,
                    "latency_ms": 210.5,
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "total_tokens": 160,
                },
            }
        ]
        extract_meta = {
            "operation": "Extract claims",
            "timestamp": "14:31:40",
            "primary_model": "openai/gpt-4o-mini",
            "model_used": "openai/gpt-4o-mini",
            "fallback_used": False,
            "latency_ms": 95.0,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        }
        rewrite_meta = {
            "operation": "Rewrite answer",
            "timestamp": "14:31:50",
            "primary_model": "openai/gpt-4o-mini",
            "model_used": "anthropic/claude-3-5-sonnet",
            "fallback_used": True,
            "latency_ms": 180.0,
            "prompt_tokens": 100,
            "completion_tokens": 35,
            "total_tokens": 135,
        }

        payload = reporting.build_report_payload(
            mode="Ask & Audit",
            question="What is the latest iOS version?",
            original_answer="iOS 17 is the latest iOS.",
            corrected_answer="iOS 26 is the latest iOS.",
            verification_results=results,
            extract_model="openai/gpt-4o-mini",
            extract_meta=extract_meta,
            corrected_model="anthropic/claude-3-5-sonnet",
            corrected_meta=rewrite_meta,
            generated_at=fixed_dt,
        )

        # Verify JSON round-trip and required fields.
        json_str = reporting.format_report_json(payload)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["generated_at"], "2026-11-11T14:31:52+00:00")
        self.assertEqual(parsed["timestamps"]["date"], "2026-11-11")
        self.assertEqual(len(parsed["timestamps"]["request_timestamps"]), 3)
        self.assertEqual(parsed["summary"]["overall_score"], 12)
        self.assertEqual(parsed["summary"]["tavily_credits_used"], 2.2)
        self.assertEqual(parsed["models_used"], ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"])
        self.assertEqual(parsed["fallback_routes"], ["openai/gpt-4o-mini -> anthropic/claude-3-5-sonnet"])
        self.assertEqual(parsed["tokens_used"]["prompt_tokens"], 270)
        self.assertEqual(parsed["tokens_used"]["completion_tokens"], 95)
        self.assertEqual(parsed["tokens_used"]["total_tokens"], 365)
        self.assertEqual(parsed["diff_summary"]["removed_words"], 1)
        self.assertEqual(parsed["diff_summary"]["added_words"], 1)
        self.assertEqual(parsed["changes_made"][0]["sources"][0]["url"], "https://support.apple.com/ios")
        self.assertEqual(parsed["claims"][0]["sources"][0]["url"], "https://support.apple.com/ios")

        # Verify Markdown content contains claims, scores, sources, timestamps, models, and tokens.
        md_str = reporting.format_report_markdown(payload)
        self.assertIn("# Stale-Answer Detector Report", md_str)
        self.assertIn("2026-11-11T14:31:52+00:00", md_str)
        self.assertIn("12/100 (Stale)", md_str)
        self.assertIn("openai/gpt-4o-mini, anthropic/claude-3-5-sonnet", md_str)
        self.assertIn("365 total (270 prompt / 95 completion)", md_str)
        self.assertIn("[Apple Support](https://support.apple.com/ios)", md_str)
        self.assertIn("## What Changed & Why", md_str)
        self.assertIn("iOS 26 is the latest iOS.", md_str)
        self.assertIn("| 3 | Rewrite answer | 14:31:50 |", md_str)


if __name__ == "__main__":
    unittest.main()
