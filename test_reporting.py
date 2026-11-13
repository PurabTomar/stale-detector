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
        self.assertIn(changes[0]["action"], {"Updated", "Replaced"})
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

    def test_build_draft_corrected_answer_replaces_outdated_claims_and_keeps_fresh_ones(self):
        original = "Sam Altman is CEO of OpenAI. Python 3.11 is the latest Python release."
        results = [
            {
                "claim": "Sam Altman is CEO of OpenAI.",
                "status": "fresh",
                "score": 92,
                "current_info": "Sam Altman remains CEO of OpenAI.",
            },
            {
                "claim": "Python 3.11 is the latest Python release.",
                "status": "outdated",
                "score": 12,
                "current_info": "Python 3.14 is the latest stable Python release",
            },
        ]

        draft = reporting.build_draft_corrected_answer(original, results)
        self.assertIn("Sam Altman is CEO of OpenAI.", draft)
        self.assertIn("Python 3.14 is the latest stable Python release.", draft)
        self.assertNotIn("Python 3.11", draft)

        # When explicit matched_spans are provided, replace only that exact span.
        span_start = original.index("Python 3.11")
        draft_with_spans = reporting.build_draft_corrected_answer(
            original,
            results,
            matched_spans=[{"claim_index": 1, "start": span_start, "end": len(original) - 1}],
        )
        self.assertIn("Python 3.14 is the latest stable Python release.", draft_with_spans)

        # Fresh-only or empty answers return unchanged text.
        fresh_only = [results[0]]
        self.assertEqual(reporting.build_draft_corrected_answer(original, fresh_only), original)
        self.assertEqual(reporting.build_draft_corrected_answer("", results), "")

    def test_report_handles_unverifiable_risk_weighted_score_and_structured_fields(self):
        results = [
            {
                "claim": "India GST slab includes 12% and 28%.",
                "category": "regulation",
                "risk_level": "critical",
                "status": "fresh",
                "score": 90,
                "confidence": "high",
                "current_info": "Verified GST slabs match.",
                "reason": "Government portal confirms.",
                "sources": [{"title": "GST Council", "url": "https://gstcouncil.gov.in/rates"}],
            },
            {
                "claim": "iOS 17 is the latest iOS.",
                "category": "version",
                "risk_level": "moderate",
                "status": "outdated",
                "score": 10,
                "confidence": "high",
                "current_info": "iOS 26 is current.",
                "current_value": "iOS 26",
                "effective_date": "2026-09-15",
                "change_summary": "iOS 18 and iOS 26 superseded iOS 17",
                "stale_since": "2024-09",
                "reason": "Newer major OS versions released.",
                "sources": [{"title": "Apple Support", "url": "https://support.apple.com/ios"}],
            },
            {
                "claim": "Internal Codename X ships in Q5.",
                "category": "other",
                "risk_level": "low",
                "status": "unverifiable",
                "score": None,
                "confidence": "low",
                "current_info": "Unverified (no authoritative live web sources found).",
                "reason": "No public evidence found.",
                "sources": [],
            },
        ]

        # Change log includes outdated AND unverifiable (marked Flagged (unverifiable)) with structured fields.
        changes = reporting.build_change_log(results)
        self.assertIn(changes[0]["action"], {"Updated", "Replaced"})
        self.assertEqual(changes[1]["action"], "Flagged (unverifiable)")
        self.assertEqual(changes[0]["current_value"], "iOS 26")
        self.assertEqual(changes[0]["effective_date"], "2026-09-15")
        self.assertEqual(changes[0]["change_summary"], "iOS 18 and iOS 26 superseded iOS 17")
        self.assertEqual(changes[0]["stale_since"], "2024-09")
        self.assertIsNone(changes[1]["score"])

        # Overall score is risk-weighted: (90*3.0 + 10*1.5) / (3.0 + 1.5) = 63, skipping unverifiable.
        payload = reporting.build_report_payload(
            mode="Audit pasted answer",
            original_answer="Test original.",
            corrected_answer="Test corrected.",
            verification_results=results,
        )
        self.assertEqual(payload["summary"]["overall_score"], 63)
        self.assertEqual(payload["summary"]["overall_tier"], "Mixed")
        self.assertEqual(payload["summary"]["unverifiable"], 1)
        self.assertIsNone(payload["claims"][2]["score"])
        self.assertEqual(payload["claims"][1]["change_summary"], "iOS 18 and iOS 26 superseded iOS 17")

        md_str = reporting.format_report_markdown(payload)
        self.assertIn("63/100 (Mixed)", md_str)
        self.assertIn("1 unverifiable", md_str)
        self.assertIn("[UNVERIFIABLE — N/A]", md_str)
        self.assertIn("**Change summary:** iOS 18 and iOS 26 superseded iOS 17", md_str)
        self.assertIn("**Effective date:** 2026-09-15", md_str)
        self.assertIn("**Stale since:** 2024-09", md_str)

        # When ALL claims are unverifiable, overall score is None and overall_tier is "Unverifiable".
        all_unverifiable_payload = reporting.build_report_payload(
            mode="Audit pasted answer",
            original_answer="Internal claim.",
            verification_results=[results[2]],
        )
        self.assertIsNone(all_unverifiable_payload["summary"]["overall_score"])
        self.assertEqual(all_unverifiable_payload["summary"]["overall_tier"], "Unverifiable")
        self.assertIn("Overall freshness score:** Unverifiable", reporting.format_report_markdown(all_unverifiable_payload))

    def test_evidence_layer_fields_in_json_and_markdown(self):
        long_snippet = " ".join(f"tok{i}" for i in range(1, 40))
        results = [
            {
                "claim": "Acme SDK costs $50/month.",
                "category": "price",
                "risk_level": "critical",
                "status": "outdated",
                "score": 15,
                "confidence": "high",
                "current_info": "Acme SDK costs $99/month.",
                "current_value": "$99/month",
                "stale_since": "2026-05-01",
                "stale_for_months": 6,
                "stale_duration_label": "stale for ~6 months",
                "entity_domains": ["acme.dev"],
                "sources_agreeing": 2,
                "sources_total": 3,
                "agreement_summary": "2 of 3 sources agree",
                "conflict_detected": False,
                "conflict_note": None,
                "reason": "Official pricing page increased to $99/month in May 2026.",
                "sources": [
                    {
                        "title": "Acme Pricing",
                        "url": "https://acme.dev/pricing",
                        "source_type": "Primary",
                        "published_date": "2026-05-01",
                        "snippet": long_snippet,
                    },
                    {
                        "title": "Reuters Tech",
                        "url": "https://www.reuters.com/technology/acme-price",
                        "published_date": "2026-05-03",
                        "snippet": "Acme raised SDK tier pricing.",
                    },
                ],
            },
            {
                "claim": "Acme CEO is Jordan Lee.",
                "category": "company",
                "risk_level": "moderate",
                "status": "uncertain",
                "score": 52,
                "confidence": "low",
                "current_info": "Sources disagree on Acme CEO.",
                "sources_agreeing": 1,
                "sources_total": 2,
                "agreement_summary": "1 of 2 sources agree",
                "conflict_detected": True,
                "conflict_note": "Sources disagree on the current fact (1 of 2 sources agree).",
                "reason": "Sources disagree between two reports.",
                "sources": [
                    {
                        "title": "Tech Forum",
                        "url": "https://www.reddit.com/r/tech/comments/1",
                        "source_type": "Forum",
                        "published_date": "2026-11-02",
                        "snippet": "Mixed reports on leadership.",
                    }
                ],
            },
        ]

        payload = reporting.build_report_payload(
            mode="Audit pasted answer",
            original_answer="Acme SDK costs $50/month and CEO is Jordan Lee.",
            verification_results=results,
        )
        c0 = payload["claims"][0]
        self.assertEqual(c0["stale_for_months"], 6)
        self.assertEqual(c0["stale_duration_label"], "stale for ~6 months")
        self.assertEqual(c0["entity_domains"], ["acme.dev"])
        self.assertEqual(c0["agreement_summary"], "2 of 3 sources agree")
        self.assertEqual(c0["sources"][0]["source_type"], "Primary")
        self.assertEqual(c0["sources"][0]["published_date"], "2026-05-01")
        self.assertLessEqual(len(c0["sources"][0]["snippet"].split()), 25)
        self.assertEqual(c0["sources"][1]["source_type"], "News")

        c1 = payload["claims"][1]
        self.assertTrue(c1["conflict_detected"])
        self.assertIn("Sources disagree", c1["conflict_note"])
        self.assertEqual(c1["sources"][0]["source_type"], "Forum")

        md_str = reporting.format_report_markdown(payload)
        self.assertIn("stale for ~6 months", md_str)
        self.assertIn("2 of 3 sources agree", md_str)
        self.assertIn("1 of 2 sources agree", md_str)
        self.assertIn("[Primary]", md_str)
        self.assertIn("[News]", md_str)
        self.assertIn("[Forum]", md_str)
        self.assertIn("2026-05-01", md_str)

    def test_dual_judge_and_session_latency_fallback_metrics(self):
        extract_meta = {
            "operation": "Extract claims",
            "task": "extract",
            "timestamp": "15:00:00",
            "primary_model": "gemini/gemini-2.5-flash-lite",
            "model_used": "gemini/gemini-2.5-flash-lite",
            "fallback_used": False,
            "latency_ms": 90.0,
            "prompt_tokens": 40,
            "completion_tokens": 15,
            "total_tokens": 55,
        }
        results = [
            {
                "claim": "Acme SDK costs $50/month.",
                "category": "price",
                "risk_level": "critical",
                "status": "uncertain",
                "score": 55,
                "confidence": "low",
                "current_info": "Judges disagreed on exact plan tier.",
                "reason": "Models disagree across billing pages.",
                "dual_judge_used": True,
                "judge_models": ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"],
                "judge_verdicts": [
                    {"model": "openai/gpt-4o-mini", "status": "fresh", "score": 88, "current_info": "$50"},
                    {"model": "anthropic/claude-3-5-sonnet-20241022", "status": "outdated", "score": 22, "current_info": "$79"},
                ],
                "model_disagreement": True,
                "model_agreement_summary": "Models disagree",
                "judge_telemetry": [
                    {
                        "operation": "Verify claim (Judge 1)",
                        "task": "judge",
                        "timestamp": "15:00:01",
                        "primary_model": "openai/gpt-4o-mini",
                        "model_used": "groq/llama-3.3-70b-versatile",
                        "fallback_used": True,
                        "latency_ms": 150.0,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "total_tokens": 100,
                    },
                    {
                        "operation": "Verify claim (Judge 2)",
                        "task": "judge",
                        "timestamp": "15:00:02",
                        "primary_model": "anthropic/claude-3-5-sonnet-20241022",
                        "model_used": "anthropic/claude-3-5-sonnet-20241022",
                        "fallback_used": False,
                        "latency_ms": 210.0,
                        "prompt_tokens": 90,
                        "completion_tokens": 30,
                        "total_tokens": 120,
                    },
                ],
                "sources": [{"title": "Acme Pricing", "url": "https://acme.dev/pricing"}],
            }
        ]

        payload = reporting.build_report_payload(
            mode="Audit pasted answer",
            original_answer="Acme SDK costs $50/month.",
            verification_results=results,
            extract_model="gemini/gemini-2.5-flash-lite",
            extract_meta=extract_meta,
            audit_duration_ms=950.0,
        )

        self.assertEqual(payload["tokens_used"]["requests_count"], 3)
        self.assertEqual(payload["tokens_used"]["fallback_count"], 1)
        self.assertEqual(payload["tokens_used"]["avg_latency_ms"], 150.0)
        self.assertEqual(payload["tokens_used"]["audit_duration_ms"], 950.0)
        self.assertEqual(payload["tokens_used"]["total_tokens"], 275)
        self.assertTrue(payload["claims"][0]["dual_judge_used"])
        self.assertEqual(payload["claims"][0]["model_agreement_summary"], "Models disagree")

        md_str = reporting.format_report_markdown(payload)
        self.assertIn("Dual-judge consensus:** Models disagree", md_str)
        self.assertIn("avg latency 150 ms", md_str)
        self.assertIn("audit time 950 ms", md_str)
        self.assertIn("1 fallback(s)", md_str)


    def test_draft_corrected_answer_no_broken_sentence_splicing(self):
        """Regression: replacing a sub-sentence span must not leave dangling fragments.

        The original bug produced text like:
            "Goods and services fall under The GST structure has been simplified..."
        because the replacement was spliced mid-sentence.  The fix expands every
        outdated span to its full enclosing sentence boundary before substituting.
        """
        # --- Case 1: sub-sentence span (the original bug) ---
        original = (
            "Goods and services fall under four GST tax slabs."
            " The government taxes luxury goods at 28%."
        )
        results = [
            {
                "claim": "Goods and services fall under four GST tax slabs",
                "status": "outdated",
                "score": 10,
                "current_info": "The GST structure has been simplified to three slabs: 5%, 12%, and 18%.",
            }
        ]
        draft = reporting.build_draft_corrected_answer(original, results)
        # Dangling fragment must be gone.
        self.assertNotIn("fall under The GST", draft)
        # Replacement text must be present.
        self.assertIn("three slabs", draft)
        # Second sentence must be preserved verbatim.
        self.assertIn("The government taxes luxury goods at 28%.", draft)
        # No double spaces.
        self.assertNotIn("  ", draft)

        # --- Case 2: version numbers must not cause mis-splits ---
        original2 = (
            "Sam Altman is CEO of OpenAI."
            " Python 3.11 is the latest release."
            " iOS 17 is the latest iOS."
        )
        results2 = [
            {
                "claim": "Sam Altman is CEO of OpenAI.",
                "status": "fresh",
                "score": 92,
                "current_info": "Sam Altman remains CEO of OpenAI.",
            },
            {
                "claim": "Python 3.11 is the latest release.",
                "status": "outdated",
                "score": 10,
                "current_info": "Python 3.14 is the latest stable release.",
            },
            {
                "claim": "iOS 17 is the latest iOS.",
                "status": "outdated",
                "score": 5,
                "current_info": "iOS 26 is the current release.",
            },
        ]
        draft2 = reporting.build_draft_corrected_answer(original2, results2)
        self.assertIn("Sam Altman is CEO of OpenAI.", draft2)
        self.assertIn("Python 3.14", draft2)
        self.assertIn("iOS 26", draft2)
        self.assertNotIn("Python 3.11", draft2)
        self.assertNotIn("iOS 17", draft2)
        # No broken splicing (the mis-split "3.Python" bug).
        self.assertNotIn("3.Python", draft2)
        self.assertNotIn("  ", draft2)

        # --- Case 3: exact bug from user report (GST bracket) ---
        original3 = (
            "India uses a 4-tier GST structure with slabs of 5%, 12%, 18%, and 28%."
            " Many household goods are exempt from GST."
        )
        results3 = [
            {
                "claim": "India uses a 4-tier GST structure with slabs of 5%, 12%, 18%, and 28%.",
                "status": "outdated",
                "score": 15,
                "current_info": "The GST Council has rationalised slabs to 5%, 12%, and 18%.",
            }
        ]
        draft3 = reporting.build_draft_corrected_answer(original3, results3)
        self.assertNotIn("uses a 4-tier GST", draft3)
        self.assertIn("rationalised slabs", draft3)
        self.assertIn("Many household goods are exempt from GST.", draft3)
        self.assertNotIn("  ", draft3)

    def test_draft_corrected_answer_edge_cases(self):
        """Edge cases: empty inputs, all-fresh claims, missing current_info."""
        any_outdated = [
            {
                "claim": "X is Y.",
                "status": "outdated",
                "score": 5,
                "current_info": "X is now Z.",
            }
        ]
        # Empty original
        self.assertEqual(reporting.build_draft_corrected_answer("", any_outdated), "")
        # No results
        self.assertEqual(reporting.build_draft_corrected_answer("Some text.", []), "Some text.")
        # All fresh — text unchanged
        fresh_only = [{"claim": "Sam is CEO.", "status": "fresh", "current_info": "Sam is CEO."}]
        self.assertEqual(
            reporting.build_draft_corrected_answer("Sam is CEO.", fresh_only), "Sam is CEO."
        )
        # Outdated but no current_info — text unchanged (no replacement to make)
        no_info = [{"claim": "Old fact.", "status": "outdated", "current_info": ""}]
        self.assertEqual(reporting.build_draft_corrected_answer("Old fact.", no_info), "Old fact.")

    def test_draft_corrected_answer_numeric_endings_bullets_compound_and_mixed_statuses(self):
        # 1. Sentence ending with a number splits cleanly without affecting decimals like 3.11
        num_orig = "Our stack runs Python 3.11. It targets iOS 17. OpenAI GPT-4o API costs $2.50 per million input tokens."
        num_results = [
            {
                "claim": "Our stack runs Python 3.11.",
                "status": "outdated",
                "score": 15,
                "current_info": "Our stack runs Python 3.15.",
            },
            {
                "claim": "It targets iOS 17.",
                "status": "outdated",
                "score": 10,
                "current_info": "It targets iOS 26.",
            },
            {
                "claim": "OpenAI GPT-4o API costs $2.50 per million input tokens.",
                "status": "fresh",
                "score": 92,
                "current_info": "Verified fresh.",
            },
        ]
        num_draft = reporting.build_draft_corrected_answer(num_orig, num_results)
        self.assertIn("Our stack runs Python 3.15.", num_draft)
        self.assertIn("It targets iOS 26.", num_draft)
        self.assertIn("OpenAI GPT-4o API costs $2.50 per million input tokens.", num_draft)
        self.assertNotIn("3.11", num_draft)
        self.assertNotIn("iOS 17", num_draft)

        # 2. Multiline bullet lists stay separated and keep bullets
        bullet_orig = "- Python 3.11 is the latest release\n- iOS 17 is the newest version\n- ChatGPT Plus costs $20 per month"
        bullet_results = [
            {
                "claim": "Python 3.11 is the latest release",
                "span": "Python 3.11 is the latest release",
                "status": "outdated",
                "score": 12,
                "current_info": "Python 3.15 is the latest stable release.",
            },
            {
                "claim": "iOS 17 is the newest version",
                "span": "iOS 17 is the newest version",
                "status": "outdated",
                "score": 10,
                "current_info": "iOS 26 is the current version.",
            },
            {
                "claim": "ChatGPT Plus costs $20 per month",
                "span": "ChatGPT Plus costs $20 per month",
                "status": "fresh",
                "score": 95,
                "current_info": "Remains $20/month.",
            },
        ]
        bullet_draft = reporting.build_draft_corrected_answer(bullet_orig, bullet_results)
        self.assertIn("- Python 3.15 is the latest stable release.", bullet_draft)
        self.assertIn("- iOS 26 is the current version.", bullet_draft)
        self.assertIn("- ChatGPT Plus costs $20 per month", bullet_draft)
        self.assertEqual(len(bullet_draft.splitlines()), 3)

        # 3. Compound sentence preserves fresh sibling clause while fixing outdated clause
        comp_orig = "Sam Altman is CEO of OpenAI, and Python 3.11 is the latest release."
        comp_results = [
            {
                "claim": "Sam Altman is CEO of OpenAI",
                "span": "Sam Altman is CEO of OpenAI",
                "status": "fresh",
                "score": 95,
                "current_info": "Confirmed.",
            },
            {
                "claim": "Python 3.11 is the latest release",
                "span": "Python 3.11 is the latest release",
                "status": "outdated",
                "score": 10,
                "current_info": "Python 3.15 is the latest stable release.",
            },
        ]
        comp_draft = reporting.build_draft_corrected_answer(comp_orig, comp_results)
        self.assertIn("Sam Altman is CEO of OpenAI", comp_draft)
        self.assertIn("Python 3.15 is the latest stable release", comp_draft)
        self.assertNotIn("Python 3.11", comp_draft)

        # 4. Mixed fresh + outdated + uncertain (conflict) + unverifiable + abbreviations (Mr., U.S.)
        mixed_orig = (
            "Mr. Altman leads OpenAI in the U.S. market. "
            "iOS 17 is Apple's newest OS. "
            "Acme Plan X costs $49 per seat. "
            "Project Orion launched internally on Mars."
        )
        mixed_results = [
            {
                "claim": "Sam Altman leads OpenAI in the U.S. market.",
                "span": "Mr. Altman leads OpenAI in the U.S. market",
                "status": "fresh",
                "score": 93,
                "current_info": "Confirmed.",
            },
            {
                "claim": "iOS 17 is Apple's newest OS.",
                "span": "iOS 17 is Apple's newest OS",
                "status": "outdated",
                "score": 12,
                "current_info": "iOS 26 is Apple's newest OS.",
            },
            {
                "claim": "Acme Plan X costs $49 per seat.",
                "span": "Acme Plan X costs $49 per seat",
                "status": "uncertain",
                "conflict_detected": True,
                "score": 52,
                "current_info": "Sources disagree on Acme Plan X pricing.",
            },
            {
                "claim": "Project Orion launched internally on Mars.",
                "span": "Project Orion launched internally on Mars",
                "status": "unverifiable",
                "score": None,
                "current_info": "No authoritative public sources found.",
            },
        ]
        mixed_draft = reporting.build_draft_corrected_answer(mixed_orig, mixed_results)
        self.assertIn("Mr. Altman leads OpenAI in the U.S. market.", mixed_draft)
        self.assertIn("iOS 26 is Apple's newest OS.", mixed_draft)
        self.assertIn("Acme Plan X costs $49 per seat (needs verification — live sources disagree).", mixed_draft)
        self.assertIn("Project Orion launched internally on Mars.", mixed_draft)
        self.assertNotIn("No authoritative public sources found", mixed_draft)


if __name__ == "__main__":
    unittest.main()
