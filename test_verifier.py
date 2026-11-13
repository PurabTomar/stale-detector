"""
test_verifier.py - Unit tests for explainable freshness scoring in verifier.py.

Run with:
    python -m unittest test_verifier.py
"""

from datetime import date
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tavily.errors import (
    InvalidAPIKeyError,
    TimeoutError as TavilyTimeoutError,
    UsageLimitExceededError,
)

import llm
import verifier


class TestExplainableScoring(unittest.TestCase):
    """Unit tests for score_breakdown, recency, agreement, authority, confidence, and risk_level."""

    def test_risk_level_by_category(self):
        # Regulation and price claims carry critical risk if stale.
        self.assertEqual(verifier.get_risk_level("regulation"), "critical")
        self.assertEqual(verifier.get_risk_level("price"), "critical")
        self.assertEqual(verifier.get_risk_level("PRICE"), "critical")

        # Software versions, company info, and statistics carry moderate risk.
        self.assertEqual(verifier.get_risk_level("version"), "moderate")
        self.assertEqual(verifier.get_risk_level("company"), "moderate")
        self.assertEqual(verifier.get_risk_level("statistic"), "moderate")

        # General / unknown categories are low risk.
        self.assertEqual(verifier.get_risk_level("other"), "low")
        self.assertEqual(verifier.get_risk_level("unknown"), "low")

    def test_source_recency_brackets_and_missing_dates(self):
        today = date(2026, 11, 11)

        # Recent sources (<= 30 days) score 100.
        recent_results = [
            {"url": "https://example.com/1", "published_date": "2026-11-01"},
            {"url": "https://example.com/2", "published_date": "2026-10-20T10:00:00Z"},
        ]
        score, detail = verifier._score_source_recency(recent_results, today=today)
        self.assertEqual(score, 100)
        self.assertIn("2/2 sources dated", detail)

        # Very old sources (> 2 years) score 15.
        old_results = [{"url": "https://example.com/old", "published_date": "2022-01-01"}]
        old_score, _ = verifier._score_source_recency(old_results, today=today)
        self.assertEqual(old_score, 15)

        # Undated sources fall back cleanly to neutral 50/100.
        undated_results = [{"url": "https://docs.python.org/3/", "published_date": None}]
        neutral_score, neutral_detail = verifier._score_source_recency(undated_results, today=today)
        self.assertEqual(neutral_score, 50)
        self.assertIn("neutral", neutral_detail.lower())

    def test_source_authority_prefers_official_over_blogs_and_forums(self):
        official_results = [
            {"url": "https://www.gst.gov.in/rules"},
            {"url": "https://docs.python.org/3/whatsnew/"},
            {"url": "https://openai.com/api/pricing"},
        ]
        blog_forum_results = [
            {"url": "https://www.reddit.com/r/python/comments/123"},
            {"url": "https://medium.com/@user/some-article"},
            {"url": "https://www.quora.com/What-is-the-latest"},
        ]

        official_score, _ = verifier._score_source_authority(official_results)
        low_score, _ = verifier._score_source_authority(blog_forum_results)

        self.assertGreaterEqual(official_score, 90)
        self.assertLessEqual(low_score, 35)
        self.assertGreater(official_score, low_score)

    def test_source_agreement_llm_count_and_fallback_heuristic(self):
        results = [
            {"title": "Python 3.14 Release", "content": "Python 3.14 is the latest stable version."},
            {"title": "Python Downloads", "content": "Download Python 3.14 today."},
            {"title": "Old Guide", "content": "Installing Python 3.11 on Ubuntu."},
            {"title": "Unrelated Post", "content": "JavaScript frameworks overview."},
        ]

        # When LLM supplies agreeing_sources, use it clamped to total results.
        score_llm, detail_llm, count_llm = verifier._score_source_agreement(
            results, "Python 3.14 is the latest stable version.", llm_agreeing_count=3
        )
        self.assertEqual(count_llm, 3)
        self.assertEqual(score_llm, 75)
        self.assertIn("3/4", detail_llm)

        # When LLM count is None, keyword/version overlap counts matching snippets.
        score_fb, _, count_fb = verifier._score_source_agreement(
            results, "Python 3.14 is the latest stable version.", llm_agreeing_count=None
        )
        self.assertEqual(count_fb, 2)
        self.assertEqual(score_fb, 50)

    def test_weighted_score_breakdown_and_confidence(self):
        today = date(2026, 11, 11)
        search_results = [
            {
                "title": "Official Docs",
                "url": "https://docs.python.org/3/",
                "published_date": "2026-11-05",
                "content": "Python 3.14 is the latest stable release.",
            },
            {
                "title": "Python Downloads",
                "url": "https://www.python.org/downloads/",
                "published_date": "2026-11-01",
                "content": "Latest Python 3.14.0 release notes.",
            },
            {
                "title": "PEP Release",
                "url": "https://peps.python.org/pep-0745/",
                "published_date": "2026-10-28",
                "content": "Python 3.14 release schedule.",
            },
            {
                "title": "Tech News",
                "url": "https://github.com/python/cpython",
                "published_date": "2026-11-02",
                "content": "CPython 3.14 branch.",
            },
        ]

        # Test exact weighted sum: 50% LLM + 20% recency + 20% agreement + 10% authority.
        res = verifier.compute_freshness_score(
            status="fresh",
            llm_score=90,
            current_info="Python 3.14 is latest.",
            search_results=search_results,
            llm_agreeing_count=4,
            today=today,
        )

        breakdown = res["score_breakdown"]
        self.assertEqual(breakdown["llm_verdict"]["weight"], 0.50)
        self.assertEqual(breakdown["source_recency"]["weight"], 0.20)
        self.assertEqual(breakdown["source_agreement"]["weight"], 0.20)
        self.assertEqual(breakdown["source_authority"]["weight"], 0.10)

        expected_total = round(
            breakdown["llm_verdict"]["weighted_points"]
            + breakdown["source_recency"]["weighted_points"]
            + breakdown["source_agreement"]["weighted_points"]
            + breakdown["source_authority"]["weighted_points"]
        )
        self.assertEqual(res["score"], expected_total)
        self.assertEqual(breakdown["final_score"], expected_total)
        self.assertEqual(res["confidence"], "high")
        self.assertIn(f"Score {expected_total}/100", res["why_this_score"])

    def test_verify_claim_returns_all_explainable_fields(self):
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "OpenAI Pricing",
                    "url": "https://openai.com/api/pricing",
                    "published_date": date.today().isoformat(),
                    "content": "Current API pricing is $5 per 1M input tokens.",
                },
                {
                    "title": "OpenAI Docs",
                    "url": "https://platform.openai.com/docs/models",
                    "published_date": date.today().isoformat(),
                    "content": "Latest models pricing details.",
                },
                {
                    "title": "Forum post",
                    "url": "https://www.reddit.com/r/OpenAI/comments/abc",
                    "published_date": date.today().isoformat(),
                    "content": "Pricing changed to $5.",
                },
            ],
            "usage": {"credits": 2},
        }
        mock_tavily.extract.return_value = {
            "results": [
                {
                    "url": "https://openai.com/api/pricing",
                    "raw_content": "Official pricing page: GPT input tokens cost $5.00 / 1M tokens.",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 0.2},
        }

        llm_payload = (
            '{"status": "outdated", "score": 15, "agreeing_sources": 3, '
            '"evidence_stance": "contradicts", '
            '"supporting_quote": "Official pricing page: GPT input tokens cost $5.00 / 1M tokens.", '
            '"current_info": "Pricing is $5 per 1M input tokens.", '
            '"reason": "The $10 price was lowered."}'
        )

        with patch("verifier.call_llm", return_value=(llm_payload, "gpt-4o-mini")):
            result = verifier.verify_claim(
                {
                    "claim": "API costs $10 per 1M input tokens.",
                    "category": "price",
                    "search_query": "OpenAI API pricing",
                },
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(result["status"], "outdated")
        self.assertEqual(result["risk_level"], "critical")
        self.assertIn(result["confidence"], ("high", "medium", "low"))
        self.assertIn("score_breakdown", result)
        self.assertIn("why_this_score", result)
        self.assertEqual(len(result["sources"]), 3)
        self.assertEqual(result["model"], "gpt-4o-mini")
        self.assertEqual(result["extracted_url"], "https://openai.com/api/pricing")
        self.assertAlmostEqual(result["tavily_credits_used"], 4.2)

    def test_verify_claim_error_paths_include_explainable_fields(self):
        # 1. Tavily failure returns 'unverifiable' (score=None) + low confidence + risk_level.
        failing_tavily = MagicMock()
        failing_tavily.search.side_effect = RuntimeError("Network timeout")
        res_tavily_err = verifier.verify_claim(
            {"claim": "GST rate is 18%.", "category": "regulation", "search_query": "GST rate"},
            tavily_client=failing_tavily,
            use_cache=False,
        )
        self.assertEqual(res_tavily_err["status"], "unverifiable")
        self.assertIsNone(res_tavily_err["score"])
        self.assertEqual(res_tavily_err["confidence"], "low")
        self.assertEqual(res_tavily_err["risk_level"], "critical")
        self.assertIn("score_breakdown", res_tavily_err)
        self.assertIn("why_this_score", res_tavily_err)
        self.assertIn("current_value", res_tavily_err)
        self.assertIn("effective_date", res_tavily_err)
        self.assertIn("change_summary", res_tavily_err)
        self.assertIn("stale_since", res_tavily_err)
        self.assertEqual(res_tavily_err["tavily_credits_used"], 0.0)

        # 2. Invalid JSON from LLM still scores source signals and attaches top 3 URLs.
        ok_tavily = MagicMock()
        ok_tavily.search.return_value = {
            "results": [
                {"title": "Gov", "url": "https://gst.gov.in/rates", "published_date": date.today().isoformat(), "content": "GST"}
            ]
        }
        with patch("verifier.call_llm", return_value=("not valid json", "gpt-4o-mini")):
            res_llm_err = verifier.verify_claim(
                {"claim": "GST rate is 18%.", "category": "regulation", "search_query": "GST rate"},
                tavily_client=ok_tavily,
                use_cache=False,
            )
        self.assertEqual(res_llm_err["status"], "uncertain")
        self.assertGreaterEqual(res_llm_err["score"], 40)
        self.assertLessEqual(res_llm_err["score"], 74)
        self.assertEqual(res_llm_err["confidence"], "low")
        self.assertEqual(res_llm_err["risk_level"], "critical")
        self.assertEqual(len(res_llm_err["sources"]), 1)
        self.assertIn("score_breakdown", res_llm_err)
        self.assertIn("why_this_score", res_llm_err)
        self.assertNotIn("Traceback", res_llm_err["error"])

    def test_score_clamping_empty_sources_and_confidence_rules(self):
        today = date(2026, 11, 11)

        # Out-of-range LLM scores clamp cleanly, and status bands (fresh 75-100, outdated 0-39) are enforced.
        empty_high = verifier.compute_freshness_score(
            status="fresh",
            llm_score=140,
            current_info="Confirmed.",
            search_results=[],
            llm_agreeing_count=3,
            today=today,
        )
        self.assertEqual(empty_high["score"], 75)
        self.assertEqual(empty_high["score_breakdown"]["llm_verdict"]["score"], 100)
        self.assertEqual(empty_high["score_breakdown"]["evidence_strength"], 0)
        # No sources caps confidence at low even if status is fresh
        self.assertEqual(empty_high["confidence"], "low")

        empty_low = verifier.compute_freshness_score(
            status="outdated",
            llm_score=-25,
            current_info="Updated.",
            search_results=[],
            llm_agreeing_count=0,
            today=today,
        )
        self.assertGreaterEqual(empty_low["score"], 0)
        self.assertLessEqual(empty_low["score"], 39)
        self.assertEqual(empty_low["score_breakdown"]["llm_verdict"]["score"], 0)

        # Same-domain URLs count as only 1 independent source -> low confidence.
        same_domain_urls = [
            {
                "title": f"Gov {i}",
                "url": f"https://www.sec.gov/rule-{i}",
                "published_date": "2026-11-10",
                "content": "Filing rule update",
            }
            for i in range(4)
        ]
        single_domain_cap = verifier.compute_freshness_score(
            status="uncertain",
            llm_score=70,
            current_info="Filing rule update",
            search_results=same_domain_urls,
            llm_agreeing_count=4,
            today=today,
        )
        self.assertEqual(single_domain_cap["confidence"], "low")

        # Uncertain status with multiple independent official sources caps confidence at 'medium' (never 'high').
        independent_official = [
            {"title": "SEC", "url": "https://www.sec.gov/rule", "published_date": "2026-11-10", "content": "Filing rule update"},
            {"title": "Treasury", "url": "https://home.treasury.gov/rule", "published_date": "2026-11-10", "content": "Filing rule update"},
            {"title": "Federal Register", "url": "https://www.federalregister.gov/rule", "published_date": "2026-11-10", "content": "Filing rule update"},
        ]
        uncertain_cap = verifier.compute_freshness_score(
            status="uncertain",
            llm_score=70,
            current_info="Filing rule update",
            search_results=independent_official,
            llm_agreeing_count=3,
            today=today,
        )
        self.assertEqual(uncertain_cap["confidence"], "medium")

    def test_friendly_tavily_invalid_key_and_limit_errors(self):
        # 1. Invalid Tavily API key returns unverifiable (score=None) + crisp error without tracebacks
        bad_key_client = MagicMock()
        bad_key_client.search.side_effect = InvalidAPIKeyError("Invalid API key")
        res_bad_key = verifier.verify_claim(
            {"claim": "GPT-4 costs $10.", "category": "price", "search_query": "GPT-4 pricing"},
            tavily_client=bad_key_client,
            use_cache=False,
        )
        self.assertEqual(res_bad_key["status"], "unverifiable")
        self.assertIsNone(res_bad_key["score"])
        self.assertIn("TAVILY_API_KEY", res_bad_key["error"])
        self.assertNotIn("Traceback", res_bad_key["error"])

        # 2. Exhausted Tavily limit returns friendly quota/limit message + unverifiable status
        limit_client = MagicMock()
        limit_client.search.side_effect = UsageLimitExceededError("Monthly credit limit exceeded")
        with patch("verifier.time.sleep"):
            res_limit = verifier.verify_claim(
                {"claim": "GPT-4 costs $10.", "category": "statistic", "search_query": "GPT-4 stats"},
                tavily_client=limit_client,
                use_cache=False,
            )
        self.assertEqual(res_limit["status"], "unverifiable")
        self.assertIsNone(res_limit["score"])
        self.assertIn("429", res_limit["error"])
        self.assertIn("limit", res_limit["error"].lower())
        self.assertNotIn("Traceback", res_limit["error"])

    def test_score_bands_enforced_regardless_of_evidence_strength(self):
        today = date(2026, 11, 11)
        weak_old_blog = [
            {
                "title": "Old personal post",
                "url": "https://medium.com/@writer/some-note",
                "published_date": "2020-01-01",
                "content": "Unrelated chatter",
            }
        ]
        # Fresh claim stays strictly in 75-100 even when evidence strength is low.
        fresh_low_ev = verifier.compute_freshness_score(
            status="fresh",
            llm_score=76,
            current_info="Exact current match",
            search_results=weak_old_blog,
            llm_agreeing_count=0,
            today=today,
        )
        self.assertGreaterEqual(fresh_low_ev["score"], 75)
        self.assertLessEqual(fresh_low_ev["score"], 100)
        self.assertLess(fresh_low_ev["score_breakdown"]["evidence_strength"], 40)

        strong_official = [
            {
                "title": "GST Portal",
                "url": "https://www.gst.gov.in/rates",
                "published_date": "2026-11-10",
                "content": "GST slabs 5% and 18% current notice",
            },
            {
                "title": "CBIC Official",
                "url": "https://www.cbic.gov.in/notifications",
                "published_date": "2026-11-09",
                "content": "GST slabs 5% and 18% circular",
            },
            {
                "title": "PIB Gov",
                "url": "https://pib.gov.in/release",
                "published_date": "2026-11-08",
                "content": "GST slabs 5% and 18% press release",
            },
        ]
        # Uncertain stays strictly in 40-74 even with 100% recency/agreement/authority.
        uncertain_high_ev = verifier.compute_freshness_score(
            status="uncertain",
            llm_score=74,
            current_info="GST slabs 5% and 18%",
            search_results=strong_official,
            llm_agreeing_count=3,
            today=today,
        )
        self.assertGreaterEqual(uncertain_high_ev["score"], 40)
        self.assertLessEqual(uncertain_high_ev["score"], 74)
        self.assertGreaterEqual(uncertain_high_ev["score_breakdown"]["evidence_strength"], 90)

        # Outdated stays strictly in 10-39 even with 100% recency/agreement/authority.
        outdated_high_ev = verifier.compute_freshness_score(
            status="outdated",
            llm_score=15,
            current_info="GST slabs 5% and 18%",
            search_results=strong_official,
            llm_agreeing_count=3,
            today=today,
        )
        self.assertGreaterEqual(outdated_high_ev["score"], 10)
        self.assertLessEqual(outdated_high_ev["score"], 39)
        self.assertEqual(outdated_high_ev["score_breakdown"]["status_band"], "10-39")

        # Unverifiable returns score=None and status_band='N/A'.
        unverif = verifier.compute_freshness_score(
            status="unverifiable",
            llm_score=None,
            current_info="No evidence",
            search_results=[],
            today=today,
        )
        self.assertIsNone(unverif["score"])
        self.assertEqual(unverif["score_breakdown"]["status_band"], "N/A")
        self.assertEqual(unverif["confidence"], "low")

    def test_risk_weighted_overall_freshness_score(self):
        # Critical (price, weight 3.0) score=90 + Moderate (version, weight 1.5) score=10 -> 63 (not 50)
        results_critical_fresh = [
            {"category": "price", "risk_level": "critical", "status": "fresh", "score": 90},
            {"category": "version", "risk_level": "moderate", "status": "outdated", "score": 10},
        ]
        self.assertEqual(verifier.compute_risk_weighted_overall_score(results_critical_fresh), 63)

        # Critical (regulation, weight 3.0) score=10 + Moderate (version, weight 1.5) score=90 -> 37
        results_critical_stale = [
            {"category": "regulation", "risk_level": "critical", "status": "outdated", "score": 10},
            {"category": "version", "risk_level": "moderate", "status": "fresh", "score": 90},
        ]
        self.assertEqual(verifier.compute_risk_weighted_overall_score(results_critical_stale), 37)

        # Unverifiable claims (score=None) are ignored when computing the overall percentage.
        with_unverifiable = [
            {"category": "price", "risk_level": "critical", "status": "fresh", "score": 88},
            {"category": "other", "risk_level": "low", "status": "unverifiable", "score": None},
        ]
        self.assertEqual(verifier.compute_risk_weighted_overall_score(with_unverifiable), 88)
        self.assertIsNone(
            verifier.compute_risk_weighted_overall_score(
                [{"category": "other", "status": "unverifiable", "score": None}]
            )
        )
        self.assertIsNone(verifier.compute_risk_weighted_overall_score([]))

    def test_calibrated_confidence_rules(self):
        today = date(2026, 11, 11)

        # 1. Single source -> low (even if official)
        one_official = [
            {
                "title": "Official Portal",
                "url": "https://www.gst.gov.in/rates",
                "published_date": "2026-11-10",
                "content": "GST rates notice",
            }
        ]
        res_one = verifier.compute_freshness_score(
            status="fresh",
            llm_score=95,
            current_info="GST rates notice",
            search_results=one_official,
            llm_agreeing_count=1,
            today=today,
        )
        self.assertEqual(res_one["confidence"], "low")

        # 2. Multiple pages from same domain (docs.python.org + peps.python.org) -> low
        same_org = [
            {"title": "Docs", "url": "https://docs.python.org/3/", "published_date": "2026-11-10", "content": "Python 3.14"},
            {"title": "PEPs", "url": "https://peps.python.org/pep-0745/", "published_date": "2026-11-09", "content": "Python 3.14"},
        ]
        res_same_org = verifier.compute_freshness_score(
            status="fresh",
            llm_score=95,
            current_info="Python 3.14",
            search_results=same_org,
            llm_agreeing_count=2,
            today=today,
        )
        self.assertEqual(res_same_org["confidence"], "low")

        # 3. Conflicting sources (1/3 agree, or split) -> low
        conflicting = [
            {"title": "A", "url": "https://apple.com/news", "published_date": "2026-11-10", "content": "Price $999"},
            {"title": "B", "url": "https://reuters.com/tech", "published_date": "2026-11-10", "content": "Price $1099"},
            {"title": "C", "url": "https://bloomberg.com/tech", "published_date": "2026-11-10", "content": "Price $1199"},
        ]
        res_conflict = verifier.compute_freshness_score(
            status="uncertain",
            llm_score=55,
            current_info="Price is uncertain due to conflicting reports.",
            search_results=conflicting,
            llm_agreeing_count=1,
            reason="Sources contradict each other on price.",
            today=today,
        )
        self.assertEqual(res_conflict["confidence"], "low")

        # 4. Two independent non-authoritative domains agreeing -> medium (not high)
        non_auth_two = [
            {"title": "Site 1", "url": "https://newsone.example/tech", "published_date": "2026-11-10", "content": "Latest widget v4"},
            {"title": "Site 2", "url": "https://newstwo.example/tech", "published_date": "2026-11-10", "content": "Latest widget v4"},
        ]
        res_non_auth = verifier.compute_freshness_score(
            status="fresh",
            llm_score=90,
            current_info="Latest widget v4",
            search_results=non_auth_two,
            llm_agreeing_count=2,
            today=today,
        )
        self.assertEqual(res_non_auth["confidence"], "medium")

        # 5. Two independent sources agreeing with at least one authoritative -> high
        auth_plus_indep = [
            {"title": "GST Gov", "url": "https://www.gst.gov.in/rates", "published_date": "2026-11-10", "content": "5% and 18% GST rates"},
            {"title": "CBIC", "url": "https://www.cbic.gov.in/rates", "published_date": "2026-11-09", "content": "5% and 18% GST rates"},
        ]
        res_high = verifier.compute_freshness_score(
            status="outdated",
            llm_score=10,
            current_info="5% and 18% GST rates",
            search_results=auth_plus_indep,
            llm_agreeing_count=2,
            today=today,
        )
        self.assertEqual(res_high["confidence"], "high")
        self.assertLessEqual(res_high["score"], 39)

    def test_unverifiable_zero_or_empty_results_and_structured_fields(self):
        # 1. Zero Tavily search results -> unverifiable without guessing or calling LLM
        empty_tavily = MagicMock()
        empty_tavily.search.return_value = {"results": [], "usage": {"credits": 1}}
        with patch("verifier.call_llm") as mock_llm:
            res_empty = verifier.verify_claim(
                {"claim": "Obscure private metric is 42.", "category": "statistic", "search_query": "obscure metric"},
                tavily_client=empty_tavily,
                use_cache=False,
            )
        mock_llm.assert_not_called()
        self.assertEqual(res_empty["status"], "unverifiable")
        self.assertIsNone(res_empty["score"])
        self.assertEqual(res_empty["confidence"], "low")
        self.assertIsNone(res_empty["current_value"])

        # 2. Blank search content + empty extract -> unverifiable without calling LLM
        blank_tavily = MagicMock()
        blank_tavily.search.return_value = {
            "results": [{"title": "Empty Page", "url": "https://example.com/empty", "content": "   "}],
            "usage": {"credits": 1},
        }
        blank_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        with patch("verifier.call_llm") as mock_llm_blank:
            res_blank = verifier.verify_claim(
                {"claim": "Opaque internal policy is X.", "category": "regulation", "search_query": "policy X"},
                tavily_client=blank_tavily,
                use_cache=False,
            )
        mock_llm_blank.assert_not_called()
        self.assertEqual(res_blank["status"], "unverifiable")
        self.assertIsNone(res_blank["score"])

        # 3. Structured fields extracted and propagated on verify_claim
        ok_tavily = MagicMock()
        ok_tavily.search.return_value = {
            "results": [
                {"title": "GST Gov", "url": "https://gst.gov.in/rates", "published_date": date.today().isoformat(), "content": "5% and 18% GST slabs"},
                {"title": "CBIC", "url": "https://cbic.gov.in/notice", "published_date": date.today().isoformat(), "content": "GST 12% and 28% slabs removed, 40% demerit rate added effective 2025-09-22 (September 2025)"},
            ],
            "usage": {"credits": 2},
        }
        ok_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        structured_json = (
            '{"status": "outdated", "score": 85, "agreeing_sources": 2, '
            '"evidence_stance": "contradicts", '
            '"supporting_quote": "GST 12% and 28% slabs removed, 40% demerit rate added", '
            '"current_info": "India GST uses 5% and 18% main slabs plus a 40% demerit rate.", '
            '"current_value": "5% and 18% (+ 40% demerit rate)", '
            '"effective_date": "2025-09-22", '
            '"change_summary": "12% and 28% slabs removed, 40% added", '
            '"stale_since": "September 2025", '
            '"reason": "Four-slab 5/12/18/28% structure was replaced in GST 2.0 reform."}'
        )
        with patch("verifier.call_llm", return_value=(structured_json, "gpt-4o-mini")):
            res_struct = verifier.verify_claim(
                {"claim": "India GST has four main slabs: 5%, 12%, 18%, 28%.", "category": "regulation", "search_query": "India GST slabs"},
                tavily_client=ok_tavily,
                use_cache=False,
            )
        self.assertEqual(res_struct["status"], "outdated")
        # Even though mock LLM returned an inconsistent score 85 with outdated status, band clamping holds 0-39!
        self.assertGreaterEqual(res_struct["score"], 0)
        self.assertLessEqual(res_struct["score"], 39)
        self.assertEqual(res_struct["confidence"], "high")
        self.assertEqual(res_struct["current_value"], "5% and 18% (+ 40% demerit rate)")
        self.assertEqual(res_struct["effective_date"], "2025-09-22")
        self.assertEqual(res_struct["change_summary"], "12% and 28% slabs removed, 40% added")
        self.assertEqual(res_struct["stale_since"], "September 2025")


class TestTavilyUpgrade(unittest.TestCase):
    """Tests for category-specific search strategies, Tavily Extract, 6h cache, and rate limits."""

    def test_category_search_strategies_and_advanced_depth_gating(self):
        # 1a. Entity-mapped price (high-risk) -> primary-first (include_domains) + general, BOTH search_depth="advanced"
        apple_price_client = MagicMock()
        apple_price_client.search.side_effect = [
            {
                "results": [
                    {"title": "Apple Store", "url": "https://apple.com/iphone", "content": "iPhone 15 Pro pricing"},
                    {"title": "Apple Compare", "url": "https://apple.com/iphone/compare", "content": "iPhone 15 Pro specs and pricing"},
                ],
                "usage": {"credits": 2},
            },
            {"results": [{"title": "News Price", "url": "https://reuters.com/apple-price", "content": "iPhone 15 Pro price update"}], "usage": {"credits": 2}},
        ]
        res_apple_price = verifier.search_tavily_for_claim(
            query="iPhone 15 Pro price",
            category="price",
            client=apple_price_client,
            use_cache=False,
        )
        self.assertEqual(apple_price_client.search.call_count, 2)
        apple_first = apple_price_client.search.call_args_list[0].kwargs
        apple_second = apple_price_client.search.call_args_list[1].kwargs
        self.assertEqual(apple_first["topic"], "general")
        self.assertEqual(apple_first["search_depth"], "advanced")
        self.assertIn("apple.com", apple_first["include_domains"])
        self.assertEqual(apple_second["topic"], "general")
        self.assertEqual(apple_second["search_depth"], "advanced")
        self.assertEqual(res_apple_price["credits_used"], 4.0)

        # 1b. Unmapped price (high-risk) -> news + general, BOTH search_depth="advanced"
        price_client = MagicMock()
        price_client.search.side_effect = [
            {"results": [{"title": "News Price", "url": "https://reuters.com/widget-price", "content": "Price update"}], "usage": {"credits": 2}},
            {"results": [{"title": "Vendor Store", "url": "https://acme-widgets.example/pro", "content": "Widget pricing"}], "usage": {"credits": 2}},
        ]
        res_price = verifier.search_tavily_for_claim(
            query="Acme Widget Pro RTX price",
            category="price",
            client=price_client,
            use_cache=False,
        )
        self.assertEqual(price_client.search.call_count, 2)
        first_call = price_client.search.call_args_list[0].kwargs
        second_call = price_client.search.call_args_list[1].kwargs
        self.assertEqual(first_call["topic"], "news")
        self.assertEqual(first_call["time_range"], verifier.RECENT_NEWS_TIME_RANGE)
        self.assertEqual(first_call["search_depth"], "advanced")
        self.assertEqual(second_call["topic"], "general")
        self.assertEqual(second_call["search_depth"], "advanced")
        self.assertEqual(res_price["credits_used"], 4.0)

        # 2. Unmapped version & company (moderate risk) -> news + general, search_depth="basic" (NOT advanced)
        for cat in ("version", "company"):
            mod_client = MagicMock()
            mod_client.search.return_value = {
                "results": [{"title": "Release", "url": "https://example.org/release", "content": "Latest info"}],
                "usage": {"credits": 1},
            }
            res_mod = verifier.search_tavily_for_claim(
                query=f"latest {cat}",
                category=cat,
                client=mod_client,
                use_cache=False,
            )
            self.assertEqual(mod_client.search.call_count, 2)
            self.assertEqual(mod_client.search.call_args_list[0].kwargs["topic"], "news")
            self.assertEqual(mod_client.search.call_args_list[0].kwargs["search_depth"], "basic")
            self.assertEqual(mod_client.search.call_args_list[1].kwargs["topic"], "general")
            self.assertEqual(mod_client.search.call_args_list[1].kwargs["search_depth"], "basic")
            self.assertEqual(res_mod["credits_used"], 2.0)

        # 3a. Entity-mapped regulation (GST / India, high-risk) -> primary-first (official entity domains) + general [advanced]
        gst_client = MagicMock()
        gst_client.search.return_value = {
            "results": [
                {"title": "GST Portal", "url": "https://www.gst.gov.in/rates", "content": "India GST slabs"},
                {"title": "GST Council", "url": "https://gstcouncil.gov.in/decisions", "content": "GST council tax slabs"},
            ],
            "usage": {"credits": 2},
        }
        res_gst = verifier.search_tavily_for_claim(
            query="India GST slabs",
            category="regulation",
            client=gst_client,
            use_cache=False,
        )
        self.assertEqual(gst_client.search.call_count, 2)
        gst_first = gst_client.search.call_args_list[0].kwargs
        self.assertEqual(gst_first["search_depth"], "advanced")
        self.assertIn("gstcouncil.gov.in", gst_first["include_domains"])
        self.assertIn("gstcouncil.gov.in", res_gst["entity_domains"])

        # 3b. Unmapped regulation (high-risk) -> REGULATION_INCLUDE_DOMAINS + include_domains_mode="prefer" + advanced
        reg_client = MagicMock()
        reg_client.search.return_value = {
            "results": [{"title": "City Rule", "url": "https://www.city.gov/zoning", "content": "Zoning rule"}],
            "usage": {"credits": 2},
        }
        verifier.search_tavily_for_claim(
            query="California municipal zoning rule",
            category="regulation",
            client=reg_client,
            use_cache=False,
        )
        reg_kwargs = reg_client.search.call_args.kwargs
        self.assertEqual(reg_kwargs["search_depth"], "advanced")
        self.assertEqual(reg_kwargs["include_domains"], verifier.REGULATION_INCLUDE_DOMAINS)
        self.assertEqual(reg_kwargs["include_domains_mode"], "prefer")

        # 4. statistic (moderate) -> single general basic search
        stat_client = MagicMock()
        stat_client.search.return_value = {"results": [], "usage": {"credits": 1}}
        verifier.search_tavily_for_claim(
            query="global EV market share",
            category="statistic",
            client=stat_client,
            use_cache=False,
        )
        stat_kwargs = stat_client.search.call_args.kwargs
        self.assertEqual(stat_kwargs["search_depth"], "basic")
        self.assertEqual(stat_kwargs["topic"], "general")
        self.assertNotIn("include_domains", stat_kwargs)

    def test_tavily_extract_targets_single_best_url_and_handles_extract_failure(self):
        # Chooses official docs URL over higher-ranked forum post and passes extracted content to LLM.
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {"title": "Reddit Thread", "url": "https://www.reddit.com/r/tax/comments/1", "content": "Someone guessed 18%"},
                {"title": "Official GST Portal", "url": "https://www.gst.gov.in/rules", "content": "GST threshold raised to Rs 40 lakh for goods."},
                {"title": "Personal Blog", "url": "https://medium.com/@author/gst", "content": "Old GST guide"},
            ],
            "usage": {"credits": 2},
        }
        client.extract.return_value = {
            "results": [
                {
                    "url": "https://www.gst.gov.in/rules",
                    "raw_content": "Exact official rule: threshold raised to Rs 40 lakh for goods.",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 0.2},
        }

        captured_prompts = []

        def fake_llm(system_prompt, user_prompt, temperature=0.1):
            captured_prompts.append(user_prompt)
            return (
                '{"status": "outdated", "score": 15, "agreeing_sources": 2, '
                '"evidence_stance": "contradicts", '
                '"supporting_quote": "threshold raised to Rs 40 lakh for goods.", '
                '"current_info": "Threshold is Rs 40 lakh for goods.", "reason": "Updated on gst.gov.in."}',
                "gpt-4o-mini",
            )

        with patch("verifier.call_llm", side_effect=fake_llm):
            res = verifier.verify_claim(
                {"claim": "GST threshold is 20 lakh.", "category": "regulation", "search_query": "India GST threshold"},
                tavily_client=client,
                use_cache=False,
            )

        client.extract.assert_called_once()
        extract_kwargs = client.extract.call_args.kwargs
        self.assertEqual(extract_kwargs["urls"], ["https://www.gst.gov.in/rules"])
        self.assertIn("threshold raised to Rs 40 lakh", captured_prompts[0])
        self.assertEqual(res["extracted_url"], "https://www.gst.gov.in/rules")
        # Entity-mapped GST query runs 2 advanced searches (2 + 2) + 1 extract (0.2) = 4.2 credits.
        self.assertAlmostEqual(res["tavily_credits_used"], 4.2)

        # If Extract raises Timeout/429, verification still succeeds using search snippets (0 extract credits).
        client_fail_extract = MagicMock()
        client_fail_extract.search.return_value = client.search.return_value
        client_fail_extract.extract.side_effect = TavilyTimeoutError(30)
        with patch("verifier.time.sleep"):
            with patch("verifier.call_llm", side_effect=fake_llm):
                res_fallback = verifier.verify_claim(
                    {"claim": "GST threshold is 20 lakh.", "category": "regulation", "search_query": "India GST threshold"},
                    tavily_client=client_fail_extract,
                    use_cache=False,
                )
        self.assertEqual(res_fallback["status"], "outdated")
        self.assertAlmostEqual(res_fallback["tavily_credits_used"], 4.0)

    def test_on_disk_cache_hits_and_6_hour_ttl_expiry(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_db = os.path.join(tmp_dir, "tavily_test_cache.db")
            client = MagicMock()
            client.search.return_value = {
                "results": [
                    {
                        "title": "Python 3.14",
                        "url": "https://docs.python.org/3/",
                        "published_date": date.today().isoformat(),
                        "content": "Python 3.14 is latest.",
                    },
                    {
                        "title": "Python Downloads",
                        "url": "https://www.python.org/downloads/",
                        "published_date": date.today().isoformat(),
                        "content": "Download Python 3.14 latest stable release.",
                    },
                ],
                "usage": {"credits": 1},
            }
            client.extract.return_value = {
                "results": [{"url": "https://docs.python.org/3/", "raw_content": "Python 3.14.7 stable."}],
                "failed_results": [],
                "usage": {"credits": 0.2},
            }

            llm_reply = (
                '{"status": "fresh", "score": 95, "agreeing_sources": 1, '
                '"current_info": "Python 3.14 is latest.", "reason": "Confirmed."}',
                "gpt-4o-mini",
            )
            claim = {"claim": "Python 3.14 is latest.", "category": "version", "search_query": "latest Python version"}

            # 1. Cold run: calls news + general (2 search calls) + 1 extract call.
            with patch("verifier.call_llm", return_value=llm_reply):
                first = verifier.verify_claim(claim, tavily_client=client, use_cache=True, cache_path=cache_db)

            self.assertFalse(first["cache_hit"])
            self.assertEqual(client.search.call_count, 2)
            self.assertEqual(client.extract.call_count, 1)
            self.assertGreater(first["tavily_credits_used"], 0)

            # 2. Repeated check within 6 hours: hits cache -> 0 new Tavily calls & 0 credits spent!
            with patch("verifier.call_llm", return_value=llm_reply):
                second = verifier.verify_claim(claim, tavily_client=client, use_cache=True, cache_path=cache_db)

            self.assertTrue(second["cache_hit"])
            self.assertEqual(client.search.call_count, 2)
            self.assertEqual(client.extract.call_count, 1)
            self.assertEqual(second["tavily_credits_used"], 0.0)

            # 3. Direct TTL expiry check (> 6 hours later): _cache_get returns None and removes expired entry.
            key = verifier._build_cache_key("search", {"q": "ttl-test"})
            verifier._cache_set(key, "search", {"results": [{"url": "https://example.com"}]}, cache_path=cache_db, now=1000.0)
            self.assertIsNotNone(verifier._cache_get(key, cache_path=cache_db, now=1000.0 + 3600))
            self.assertIsNone(verifier._cache_get(key, cache_path=cache_db, now=1000.0 + verifier.CACHE_TTL_SECONDS + 1))

    def test_rate_limit_retry_and_partial_search_degradation(self):
        # 1. Transient 429 UsageLimitExceededError retries once and recovers cleanly.
        retry_client = MagicMock()
        retry_client.search.side_effect = [
            UsageLimitExceededError("Rate limit 100 RPM"),
            {"results": [{"title": "Source", "url": "https://example.com", "content": "Data for some stat"}], "usage": {"credits": 1}},
        ]
        with patch("verifier.time.sleep") as mock_sleep:
            out = verifier.search_tavily_for_claim(
                query="some stat",
                category="statistic",
                client=retry_client,
                use_cache=False,
            )
        self.assertEqual(retry_client.search.call_count, 2)
        mock_sleep.assert_called_once()
        self.assertEqual(len(out["results"]), 1)
        self.assertEqual(out["credits_used"], 1.0)

        # 2. In dual-search categories (company), if news fails on 429 but general succeeds,
        # verification degrades gracefully using general results without throwing away the claim.
        partial_client = MagicMock()
        partial_client.search.side_effect = [
            UsageLimitExceededError("429 on news"),
            UsageLimitExceededError("429 on news retry"),
            {"results": [{"title": "Official Bio", "url": "https://acmecorp.example/about", "content": "Alex Rivera is CEO of Acme Corp."}], "usage": {"credits": 1}},
        ]
        partial_client.extract.return_value = {
            "results": [{"url": "https://acmecorp.example/about", "raw_content": "Alex Rivera is CEO of Acme Corp."}],
            "usage": {"credits": 0.2},
        }
        llm_reply = (
            '{"status": "fresh", "score": 95, "agreeing_sources": 1, '
            '"evidence_stance": "supports", "supporting_quote": "Alex Rivera is CEO of Acme Corp.", '
            '"current_info": "Alex Rivera is CEO.", "reason": "Confirmed."}',
            "gpt-4o-mini",
        )
        with patch("verifier.time.sleep"):
            with patch("verifier.call_llm", return_value=llm_reply):
                res = verifier.verify_claim(
                    {"claim": "Alex Rivera is CEO of Acme Corp.", "category": "company", "search_query": "Acme Corp CEO"},
                    tavily_client=partial_client,
                    use_cache=False,
                )
        self.assertEqual(res["status"], "fresh")
        self.assertIsNotNone(res["search_warning"])
        self.assertIn("429", res["search_warning"])
        self.assertAlmostEqual(res["tavily_credits_used"], 1.2)

    def test_gateway_telemetry_attached_on_verify_and_rewrite(self):
        client = MagicMock()
        client.search.return_value = {
            "results": [{"title": "Docs", "url": "https://docs.python.org", "content": "Python 3.14"}],
            "usage": {"credits": 1},
        }
        client.extract.return_value = {"results": [], "usage": {"credits": 0}}

        def fake_call_with_meta(system_prompt, user_prompt, temperature=0.1, simulate_outage=None):
            meta = {
                "primary_model": "openai/gpt-4o-mini",
                "requested_model": "openai/gpt-4o-mini",
                "model_used": "anthropic/claude-3-5-sonnet",
                "provider_used": "anthropic",
                "fallback_models": ["anthropic/claude-3-5-sonnet"],
                "fallback_used": True,
                "latency_ms": 215.4,
                "prompt_tokens": 90,
                "completion_tokens": 35,
                "total_tokens": 125,
                "simulated_outage": False,
                "timestamp": "12:00:00",
            }
            llm._record_gateway_metrics(meta)
            return (
                '{"status": "fresh", "score": 95, "agreeing_sources": 1, '
                '"current_info": "Python 3.14.", "reason": "Confirmed."}',
                "anthropic/claude-3-5-sonnet",
            )

        with patch("verifier.call_llm", side_effect=fake_call_with_meta):
            res = verifier.verify_claim(
                {"claim": "Python 3.14 is current.", "category": "version", "search_query": "Python version"},
                tavily_client=client,
                use_cache=False,
            )

        self.assertTrue(res["fallback_used"])
        self.assertEqual(res["primary_model"], "openai/gpt-4o-mini")
        self.assertEqual(res["model"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(res["gateway_metrics"]["latency_ms"], 215.4)
        self.assertEqual(res["gateway_metrics"]["total_tokens"], 125)

        def fake_rewrite(system_prompt, user_prompt, temperature=0.2, simulate_outage=None):
            llm._record_gateway_metrics(
                {
                    "primary_model": "openai/gpt-4o-mini",
                    "requested_model": "openai/gpt-4o-mini",
                    "model_used": "anthropic/claude-3-5-sonnet",
                    "fallback_used": True,
                    "latency_ms": 180.0,
                    "prompt_tokens": 110,
                    "completion_tokens": 60,
                    "total_tokens": 170,
                }
            )
            return "Updated text", "anthropic/claude-3-5-sonnet"

        with patch("verifier.call_llm", side_effect=fake_rewrite):
            text, model = verifier.generate_corrected_answer("Original", [res])
        self.assertEqual(text, "Updated text")
        self.assertEqual(model, "anthropic/claude-3-5-sonnet")
        rw_metrics = verifier.get_last_gateway_metrics()
        self.assertEqual(rw_metrics["operation"], "Rewrite answer")
        self.assertTrue(rw_metrics["fallback_used"])
        self.assertEqual(rw_metrics["total_tokens"], 170)


class TestEvidenceLayer(unittest.TestCase):
    """Tests for primary-first entity search, source labels, <=25w snippets, conflicts, and stale duration."""

    def test_primary_source_first_search_and_configurable_entity_map(self):
        # 1. Built-in entities (OpenAI, Apple, Python, GST) resolve to official primary domains.
        self.assertIn("openai.com", verifier.resolve_entity_official_domains("OpenAI API GPT-4 pricing"))
        self.assertIn("apple.com", verifier.resolve_entity_official_domains("Apple iPhone 15 Pro price"))
        self.assertIn("python.org", verifier.resolve_entity_official_domains("Latest Python version"))
        self.assertIn("gstcouncil.gov.in", verifier.resolve_entity_official_domains("India GST Council tax slabs"))

        # 2. Configurable via runtime dict OR ENTITY_DOMAIN_MAP environment JSON.
        custom_res = verifier.resolve_entity_official_domains(
            "Acme Widget SDK v3 pricing",
            entity_domains={"acme widget": ["acmewidget.dev", "docs.acmewidget.dev"]},
        )
        self.assertEqual(custom_res, ["acmewidget.dev", "docs.acmewidget.dev"])

        with patch.dict(os.environ, {"ENTITY_DOMAIN_MAP": '{"customcorp": ["customcorp.io", "docs.customcorp.io"]}'}, clear=False):
            env_res = verifier.resolve_entity_official_domains("CustomCorp Cloud tariff update")
            self.assertEqual(env_res, ["customcorp.io", "docs.customcorp.io"])

        # 3. Search runs Search #1 with include_domains=[official entity domains], then Search #2 general.
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "OpenAI API Pricing",
                        "url": "https://openai.com/api/pricing/",
                        "published_date": "2026-10-01",
                        "content": "GPT-5 input tokens cost $1.25 per 1M tokens, replacing legacy GPT-4 Turbo $10 rates.",
                    },
                    {
                        "title": "OpenAI Models Compare",
                        "url": "https://platform.openai.com/docs/models/gpt-4-turbo",
                        "published_date": "2026-10-01",
                        "content": "GPT-4 Turbo pricing updated in API schedule.",
                    },
                ],
                "usage": {"credits": 2},
            },
            {
                "results": [
                    {
                        "title": "Reddit discussion",
                        "url": "https://www.reddit.com/r/OpenAI/comments/99",
                        "published_date": "2026-10-02",
                        "content": "Developers discuss OpenAI API GPT-4 Turbo pricing update.",
                    },
                    {
                        "title": "Reuters Tech",
                        "url": "https://www.reuters.com/technology/openai-pricing-2026/",
                        "published_date": "2026-10-03",
                        "content": "OpenAI announced new API token rates replacing GPT-4 Turbo pricing.",
                    },
                ],
                "usage": {"credits": 2},
            },
        ]
        mock_tavily.extract.return_value = {
            "results": [{"url": "https://openai.com/api/pricing/", "raw_content": "Official OpenAI pricing table: GPT-4 Turbo superseded by $1.25 rates."}],
            "failed_results": [],
            "usage": {"credits": 0.2},
        }
        llm_reply = (
            '{"status": "outdated", "score": 15, "agreeing_sources": 3, '
            '"evidence_stance": "contradicts", '
            '"supporting_quote": "GPT-5 input tokens cost $1.25 per 1M tokens, replacing legacy GPT-4 Turbo $10 rates.", '
            '"current_info": "OpenAI updated API pricing.", '
            '"current_value": "$1.25 / 1M input tokens", '
            '"reason": "GPT-4 Turbo pricing was superseded."}'
        )
        with patch("verifier.call_llm", return_value=(llm_reply, "gpt-4o-mini")):
            result = verifier.verify_claim(
                {"claim": "OpenAI GPT-4 Turbo costs $10 per million input tokens.", "category": "price", "search_query": "OpenAI API pricing"},
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(mock_tavily.search.call_count, 2)
        call1_kw = mock_tavily.search.call_args_list[0].kwargs
        call2_kw = mock_tavily.search.call_args_list[1].kwargs
        self.assertIn("openai.com", call1_kw["include_domains"])
        self.assertEqual(call1_kw["topic"], "general")
        self.assertNotIn("include_domains", call2_kw)
        self.assertEqual(call2_kw["topic"], "general")
        self.assertIn("primary-first", result["search_strategy"])
        self.assertIn("openai.com", result["entity_domains"])
        # Primary source placed first and chosen for Extract.
        self.assertEqual(result["sources"][0]["source_type"], "Primary")
        self.assertEqual(result["extracted_url"], "https://openai.com/api/pricing/")

    def test_source_labels_published_dates_and_25_word_snippets(self):
        # Verify explicit Primary / News / Blog / Forum classification rules.
        self.assertEqual(verifier.classify_source_type("https://gstcouncil.gov.in/meetings", entity_domains=["gstcouncil.gov.in"]), "Primary")
        self.assertEqual(verifier.classify_source_type("https://reuters.com/world/india/gst"), "News")
        self.assertEqual(verifier.classify_source_type("https://medium.com/@taxwriter/gst-notes"), "Blog")
        self.assertEqual(verifier.classify_source_type("https://example.org/blog/2026-update"), "Blog")
        self.assertEqual(verifier.classify_source_type("https://www.reddit.com/r/IndiaTax/comments/123"), "Forum")
        self.assertEqual(verifier.classify_source_type("https://stackoverflow.com/questions/12345"), "Forum")

        # Snippets must be trimmed to <= 25 words even on lengthy source extracts, keeping published_date.
        long_text_45_words = "Python 3.14 " + " ".join([f"word{i}" for i in range(1, 46)])
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "Python Docs",
                        "url": "https://docs.python.org/3/",
                        "published_date": "2026-10-12",
                        "content": long_text_45_words,
                    }
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [
                    {
                        "title": "Reuters Tech",
                        "url": "https://www.reuters.com/technology/python-314/",
                        "published_date": "2026-10-14",
                        "content": "Python 3.14 stable release brings free-threaded improvements.",
                    },
                    {
                        "title": "Dev Blog",
                        "url": "https://medium.com/@dev/python-314",
                        "published_date": "",
                        "content": "Overview of Python 3.14 highlights in blog format.",
                    },
                ],
                "usage": {"credits": 1},
            },
            {"results": [], "usage": {"credits": 0}},
        ]
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        llm_reply = (
            '{"status": "fresh", "score": 92, "agreeing_sources": 3, '
            '"current_info": "Python 3.14 is latest.", "reason": "Confirmed by docs and news."}'
        )
        with patch("verifier.call_llm", return_value=(llm_reply, "gpt-4o-mini")):
            res = verifier.verify_claim(
                {"claim": "Python 3.14 is the latest stable Python release.", "category": "version", "search_query": "Python 3.14 latest version"},
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(len(res["sources"]), 3)
        self.assertEqual(res["sources"][0]["source_type"], "Primary")
        self.assertEqual(res["sources"][0]["published_date"], "2026-10-12")
        self.assertEqual(len(res["sources"][0]["snippet"].split()), 25)
        self.assertTrue(res["sources"][0]["snippet"].endswith("..."))
        self.assertEqual(res["sources"][1]["source_type"], "News")
        self.assertEqual(res["sources"][1]["published_date"], "2026-10-14")
        self.assertEqual(res["sources"][2]["source_type"], "Blog")
        self.assertIsNone(res["sources"][2]["published_date"])

    def test_source_agreement_summary_and_explicit_conflict_handling(self):
        # 1. Conflicting contemporaneous sources force status='uncertain', band 40-74, low confidence,
        #    conflict_detected=True, and explicit 'sources disagree' note — even if LLM guessed 'fresh'!
        conflict_tavily = MagicMock()
        conflict_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "Reuters Report",
                        "url": "https://www.reuters.com/business/acme-ceo-2026/",
                        "published_date": "2026-11-01",
                        "content": "Alex Rivera remains CEO of Acme Corp.",
                    },
                    {
                        "title": "Bloomberg Report",
                        "url": "https://www.bloomberg.com/news/acme-leadership",
                        "published_date": "2026-11-02",
                        "content": "Jordan Lee replaced Alex Rivera as Acme Corp CEO.",
                    },
                    {
                        "title": "Forum Thread",
                        "url": "https://www.reddit.com/r/tech/comments/acme",
                        "published_date": "2026-11-02",
                        "content": "Unclear who leads Acme Corp right now.",
                    },
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [],
                "usage": {"credits": 1},
            },
        ]
        conflict_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        # Mock LLM claims 'fresh' at 88, but 1 of 3 contemporaneous domains agree -> forced conflict override!
        llm_split = (
            '{"status": "fresh", "score": 88, "agreeing_sources": 1, "conflicting_sources": false, '
            '"current_info": "Alex Rivera is CEO.", "reason": "One report states Alex Rivera is CEO."}'
        )
        with patch("verifier.call_llm", return_value=(llm_split, "gpt-4o-mini")):
            res_conflict = verifier.verify_claim(
                {"claim": "Alex Rivera is CEO of Acme Corp.", "category": "company", "search_query": "Acme Corp CEO"},
                tavily_client=conflict_tavily,
                use_cache=False,
            )

        self.assertEqual(res_conflict["agreement_summary"], "1 of 3 independent sources agree")
        self.assertEqual(res_conflict["sources_agreeing"], 1)
        self.assertEqual(res_conflict["sources_total"], 3)
        self.assertTrue(res_conflict["conflict_detected"])
        self.assertIn("sources disagree", res_conflict["conflict_note"].lower())
        self.assertIn("disagree", res_conflict["reason"].lower())
        self.assertEqual(res_conflict["status"], "uncertain")
        self.assertGreaterEqual(res_conflict["score"], 40)
        self.assertLessEqual(res_conflict["score"], 74)
        self.assertEqual(res_conflict["confidence"], "low")

        # 2. Old historical pages (>180 days older) must NOT trigger conflict on an outdated claim.
        today = date(2026, 11, 11)
        dated_hist = [
            {"title": "New Release", "url": "https://www.python.org/downloads/", "published_date": "2026-10-01", "content": "Python 3.14 is current."},
            {"title": "Old 2022 Guide", "url": "https://blog.example1.com/py", "published_date": "2022-10-01", "content": "Python 3.11 released."},
            {"title": "Old 2023 Guide", "url": "https://blog.example2.com/py", "published_date": "2023-05-01", "content": "Using Python 3.11."},
        ]
        non_conflict = verifier.compute_freshness_score(
            status="outdated",
            llm_score=10,
            current_info="Python 3.14 is current.",
            search_results=dated_hist,
            llm_agreeing_count=1,
            today=today,
        )
        self.assertFalse(non_conflict["conflict_detected"])
        self.assertEqual(non_conflict["resolved_status"], "outdated")
        self.assertEqual(non_conflict["agreement_summary"], "1 of 3 independent sources agree")

    def test_stale_since_estimates_earliest_change_source_and_months(self):
        fixed_today = date(2026, 11, 11)
        # 2022 source discusses ONLY old Python 3.11; 2025-10-07 is earliest dated source about 3.14 change!
        sources_with_history = [
            {
                "title": "Python 3.11 Release Notes",
                "url": "https://docs.python.org/3.11/",
                "published_date": "2022-10-24",
                "content": "Python 3.11.0 is the new major release in 2022.",
            },
            {
                "title": "Python 3.14 Announcement",
                "url": "https://www.python.org/downloads/release/python-3140/",
                "published_date": "2025-10-07",
                "content": "Python 3.14.0 stable release announced as the latest Python version.",
            },
            {
                "title": "Python 3.14 Mid-Year Roundup",
                "url": "https://realpython.com/python-314-news/",
                "published_date": "2026-04-15",
                "content": "Python 3.14 adoption continues across frameworks.",
            },
        ]
        stale_since, months, duration_label = verifier.estimate_stale_since_from_sources(
            status="outdated",
            claim_text="Python 3.11 is the latest Python release.",
            search_results=sources_with_history,
            current_info="Python 3.14 is the latest stable Python release.",
            current_value="Python 3.14",
            change_summary="Python 3.12, 3.13, and 3.14 superseded 3.11",
            today=fixed_today,
        )
        self.assertEqual(stale_since, "2025-10-07")
        self.assertEqual(months, 13)
        self.assertEqual(duration_label, "stale for ~13 months")

        # Sub-month and ~1 month labels + fresh claims returning None.
        _, m0, lbl0 = verifier.estimate_stale_since_from_sources(
            status="outdated",
            claim_text="Old price $99.",
            search_results=[{"title": "Price update", "url": "https://example.com/p", "published_date": "2026-11-05", "content": "New price $149."}],
            current_info="New price $149.",
            today=fixed_today,
        )
        self.assertEqual(m0, 0)
        self.assertEqual(lbl0, "stale for <1 month")

        _, m1, lbl1 = verifier.estimate_stale_since_from_sources(
            status="outdated",
            claim_text="Old price $99.",
            search_results=[{"title": "Price update", "url": "https://example.com/p", "published_date": "2026-10-11", "content": "New price $149."}],
            current_info="New price $149.",
            today=fixed_today,
        )
        self.assertEqual(m1, 1)
        self.assertEqual(lbl1, "stale for ~1 month")

        self.assertEqual(
            verifier.estimate_stale_since_from_sources(
                status="fresh",
                claim_text="Fresh claim.",
                search_results=sources_with_history,
                today=fixed_today,
            ),
            (None, None, None),
        )

    def test_primary_first_preserves_6h_cache_and_credit_accounting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_db = os.path.join(tmp_dir, "evidence_cache.db")
            client = MagicMock()
            client.search.side_effect = [
                {
                    "results": [
                        {
                            "title": "Apple Press",
                            "url": "https://www.apple.com/newsroom/2025/09/iphone-17-pro/",
                            "published_date": "2025-09-09",
                            "content": "Apple introduced iPhone 17 Pro Max starting at $1,199 with A19 Pro and iOS 26.",
                        },
                        {
                            "title": "Apple Store Compare",
                            "url": "https://www.apple.com/iphone/",
                            "published_date": "2025-09-09",
                            "content": "Buy iPhone 17 Pro Max, Apple's current flagship lineup.",
                        },
                    ],
                    "usage": {"credits": 2},
                },
                {
                    "results": [
                        {
                            "title": "TechCrunch",
                            "url": "https://techcrunch.com/2025/09/09/iphone-17-pro/",
                            "published_date": "2025-09-09",
                            "content": "iPhone 17 Pro Max ships with iOS 26 and A19 Pro.",
                        }
                    ],
                    "usage": {"credits": 2},
                },
            ]
            client.extract.return_value = {
                "results": [
                    {
                        "url": "https://www.apple.com/newsroom/2025/09/iphone-17-pro/",
                        "raw_content": "iPhone 17 Pro Max starts at $1,199 with A19 Pro.",
                    }
                ],
                "failed_results": [],
                "usage": {"credits": 0.2},
            }
            llm_reply = (
                '{"status": "outdated", "score": 15, "agreeing_sources": 2, '
                '"evidence_stance": "contradicts", '
                '"supporting_quote": "Apple introduced iPhone 17 Pro Max starting at $1,199 with A19 Pro and iOS 26.", '
                '"current_info": "Apple sells iPhone 17 Pro Max starting at $1,199 with iOS 26.", '
                '"current_value": "iPhone 17 Pro Max ($1,199)", '
                '"change_summary": "iPhone 16 Pro Max and iPhone 17 Pro Max superseded iPhone 15 Pro Max", '
                '"reason": "Newer flagship generation launched."}',
                "gpt-4o-mini",
            )
            claim = {"claim": "Apple flagship is iPhone 15 Pro Max.", "category": "price", "search_query": "latest Apple iPhone Pro Max"}

            with patch("verifier.call_llm", return_value=llm_reply):
                cold = verifier.verify_claim(claim, tavily_client=client, use_cache=True, cache_path=cache_db)
            self.assertFalse(cold["cache_hit"])
            self.assertEqual(client.search.call_count, 2)
            self.assertEqual(client.extract.call_count, 1)
            self.assertAlmostEqual(cold["tavily_credits_used"], 4.2)
            self.assertEqual(cold["agreement_summary"], "2 of 2 independent sources agree")
            self.assertEqual(cold["sources"][0]["source_type"], "Primary")
            self.assertTrue(str(cold["stale_duration_label"]).startswith("stale for "))

            with patch("verifier.call_llm", return_value=llm_reply):
                warm = verifier.verify_claim(claim, tavily_client=client, use_cache=True, cache_path=cache_db)
            self.assertTrue(warm["cache_hit"])
            self.assertEqual(warm["tavily_credits_used"], 0.0)
            self.assertEqual(client.search.call_count, 2)
            self.assertEqual(client.extract.call_count, 1)


class TestBifrostDualJudgeAndRouting(unittest.TestCase):
    """Tests for task-based routing in verifier.py and Dual-judge consensus/disagreement."""

    def test_single_judge_and_rewrite_pass_task_names(self):
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Python Docs",
                    "url": "https://docs.python.org/3/",
                    "published_date": "2026-10-10",
                    "content": "Python 3.14 is current.",
                }
            ],
            "usage": {"credits": 1},
        }
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}

        observed_tasks: list[str | None] = []

        def fake_call(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            observed_tasks.append(task)
            llm._record_gateway_metrics(
                {
                    "primary_model": "openai/gpt-4o-mini",
                    "model_used": "openai/gpt-4o-mini",
                    "fallback_used": False,
                    "latency_ms": 110.0,
                    "prompt_tokens": 60,
                    "completion_tokens": 25,
                    "total_tokens": 85,
                    "task": task,
                }
            )
            if task == "rewrite":
                return "Updated Python text.", "openai/gpt-4o-mini"
            return (
                '{"status": "fresh", "score": 92, "agreeing_sources": 1, '
                '"current_info": "Python 3.14 is latest.", "reason": "Confirmed."}',
                "openai/gpt-4o-mini",
            )

        with patch("verifier.call_llm", side_effect=fake_call):
            res = verifier.verify_claim(
                {"claim": "Python 3.14 is latest.", "category": "version", "search_query": "latest Python"},
                tavily_client=mock_tavily,
                use_cache=False,
            )
            rewritten, rw_model = verifier.generate_corrected_answer("Python 3.14 is latest.", [res])

        self.assertEqual(observed_tasks, ["judge", "rewrite"])
        self.assertEqual(res["gateway_metrics"]["task"], "judge")
        self.assertFalse(res["dual_judge_used"])
        self.assertEqual(rewritten, "Updated Python text.")
        self.assertEqual(rw_model, "openai/gpt-4o-mini")
        self.assertEqual(verifier.get_last_gateway_metrics()["task"], "rewrite")

    def test_dual_judge_agreement_and_disagreement_handling(self):
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "OpenAI Pricing",
                        "url": "https://openai.com/api/pricing/",
                        "published_date": "2026-10-15",
                        "content": "Updated OpenAI API pricing schedule replaces legacy GPT-4 Turbo $10 rates.",
                    },
                    {
                        "title": "Reuters Tech",
                        "url": "https://www.reuters.com/technology/openai-pricing/",
                        "published_date": "2026-10-16",
                        "content": "OpenAI updated API pricing for GPT-4 Turbo.",
                    },
                ],
                "usage": {"credits": 2},
            },
            {"results": [], "usage": {"credits": 1}},
        ]
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}

        # Case 1: Both judges agree -> '2/2 models agree', single Tavily search pass.
        def fake_agree(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            m_name = model_override or "openai/gpt-4o-mini"
            sc = 15 if not model_override else 25
            llm._record_gateway_metrics(
                {
                    "primary_model": m_name,
                    "model_used": m_name,
                    "fallback_used": False,
                    "latency_ms": 140.0 if not model_override else 210.0,
                    "prompt_tokens": 80,
                    "completion_tokens": 30,
                    "total_tokens": 110,
                    "task": task,
                }
            )
            return (
                f'{{"status": "outdated", "score": {sc}, "agreeing_sources": 2, '
                f'"evidence_stance": "contradicts", '
                f'"supporting_quote": "Updated OpenAI API pricing schedule replaces legacy GPT-4 Turbo $10 rates.", '
                f'"current_info": "Updated pricing applies.", "reason": "Confirmed by {m_name}."}}',
                m_name,
            )

        with patch.dict(
            os.environ,
            {
                "LLM_MODEL_JUDGE": "openai/gpt-4o-mini",
                "LLM_MODEL_FALLBACKS": "anthropic/claude-3-5-sonnet-20241022",
            },
            clear=False,
        ):
            with patch("verifier.call_llm", side_effect=fake_agree):
                agree_res = verifier.verify_claim(
                    {"claim": "GPT-4 Turbo costs $10/M tokens.", "category": "price", "search_query": "OpenAI API pricing"},
                    tavily_client=mock_tavily,
                    use_cache=False,
                    dual_judge=True,
                )

        self.assertEqual(mock_tavily.search.call_count, 2)  # Only 1 search pass for both judges
        self.assertTrue(agree_res["dual_judge_used"])
        self.assertFalse(agree_res["model_disagreement"])
        self.assertEqual(agree_res["model_agreement_summary"], "2/2 models agree")
        self.assertEqual(
            agree_res["judge_models"],
            ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"],
        )
        self.assertEqual(len(agree_res["judge_verdicts"]), 2)
        self.assertEqual(len(agree_res["judge_telemetry"]), 2)
        self.assertEqual(agree_res["status"], "outdated")
        self.assertLessEqual(agree_res["score"], 39)
        self.assertIn("2/2 models agree", agree_res["why_this_score"])

        # Case 2: Judges disagree (fresh vs outdated) -> forces Uncertain (40-74), low confidence, 'Models disagree'.
        mock_tavily_disagree = MagicMock()
        mock_tavily_disagree.search.return_value = {
            "results": [
                {
                    "title": "OpenAI Docs",
                    "url": "https://openai.com/api/pricing/",
                    "published_date": "2026-10-15",
                    "content": "OpenAI API pricing overview.",
                },
                {
                    "title": "Reuters",
                    "url": "https://www.reuters.com/tech/openai/",
                    "published_date": "2026-10-16",
                    "content": "OpenAI pricing discussed.",
                },
            ],
            "usage": {"credits": 1},
        }
        mock_tavily_disagree.extract.return_value = {"results": [], "usage": {"credits": 0}}

        def fake_disagree(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            m_name = model_override or "openai/gpt-4o-mini"
            is_j2 = bool(model_override)
            llm._record_gateway_metrics(
                {
                    "primary_model": m_name,
                    "model_used": m_name,
                    "fallback_used": (not is_j2),  # Simulate Judge 1 hitting fallback while Judge 2 stays direct
                    "latency_ms": 150.0,
                    "prompt_tokens": 70,
                    "completion_tokens": 25,
                    "total_tokens": 95,
                    "task": task,
                }
            )
            if not is_j2:
                return (
                    '{"status": "fresh", "score": 92, "agreeing_sources": 2, '
                    '"evidence_stance": "supports", "supporting_quote": "OpenAI API pricing overview.", '
                    '"current_info": "Tier matches snippet.", "reason": "Looks current."}',
                    m_name,
                )
            return (
                '{"status": "outdated", "score": 15, "agreeing_sources": 2, '
                '"evidence_stance": "contradicts", "supporting_quote": "OpenAI pricing discussed.", '
                '"current_info": "New tier lowered rate.", "reason": "Legacy tier superseded."}',
                m_name,
            )

        with patch.dict(
            os.environ,
            {
                "LLM_MODEL_JUDGE": "openai/gpt-4o-mini",
                "LLM_MODEL_FALLBACKS": "anthropic/claude-3-5-sonnet-20241022",
            },
            clear=False,
        ):
            with patch("verifier.call_llm", side_effect=fake_disagree):
                dis_res = verifier.verify_claim(
                    {"claim": "API costs $10/M tokens.", "category": "price", "search_query": "OpenAI pricing"},
                    tavily_client=mock_tavily_disagree,
                    use_cache=False,
                    dual_judge=True,
                )

        self.assertTrue(dis_res["dual_judge_used"])
        self.assertTrue(dis_res["model_disagreement"])
        self.assertEqual(dis_res["model_agreement_summary"], "Models disagree")
        self.assertEqual(dis_res["status"], "uncertain")
        self.assertGreaterEqual(dis_res["score"], 40)
        self.assertLessEqual(dis_res["score"], 74)
        self.assertEqual(dis_res["confidence"], "low")
        self.assertTrue(dis_res["fallback_used"])
        self.assertIn("Models disagree", dis_res["why_this_score"])
        self.assertEqual(dis_res["judge_verdicts"][0]["status"], "fresh")
        self.assertEqual(dis_res["judge_verdicts"][1]["status"], "outdated")

        # Case 3: Partial judge failure -> Uncertain + low confidence + Models disagree + error note.
        def fake_partial_fail(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            if model_override:
                raise RuntimeError("Second judge upstream timeout")
            return (
                '{"status": "fresh", "score": 90, "agreeing_sources": 2, '
                '"current_info": "Looks current.", "reason": "Confirmed."}',
                "openai/gpt-4o-mini",
            )

        with patch("verifier.call_llm", side_effect=fake_partial_fail):
            part_res = verifier.verify_claim(
                {"claim": "API costs $10/M tokens.", "category": "price", "search_query": "OpenAI pricing"},
                tavily_client=mock_tavily_disagree,
                use_cache=False,
                dual_judge=True,
                secondary_judge_model="anthropic/claude-3-5-sonnet-20241022",
            )

        self.assertTrue(part_res["model_disagreement"])
        self.assertEqual(part_res["model_agreement_summary"], "Models disagree")
        self.assertEqual(part_res["status"], "uncertain")
        self.assertGreaterEqual(part_res["score"], 40)
        self.assertLessEqual(part_res["score"], 74)
        self.assertEqual(part_res["confidence"], "low")
        self.assertIn("Second judge", part_res["error"])
        self.assertEqual(part_res["judge_verdicts"][1]["status"], "error")


class TestParallelAndURLExtraction(unittest.TestCase):
    """Tests for 3-worker parallel claim verification, thread-safe cache, and URL extraction."""

    def test_parallel_verify_claims_concurrency_ordering_outage_and_span(self):
        import threading
        import time

        claims = [
            {"claim": f"Claim #{i}", "span": f"span #{i}", "category": "version", "search_query": f"q{i}"}
            for i in range(1, 6)
        ]
        counter_lock = threading.Lock()
        active_workers = 0
        peak_workers = 0
        outage_seen_in_workers: list[bool] = []
        progress_events: list[tuple[int, int, str]] = []
        caller_thread_id = threading.get_ident()
        callback_threads: list[int] = []

        def fake_verify_single(
            claim_item,
            tavily_client=None,
            use_cache=True,
            dual_judge=None,
            cache_path=None,
            entity_domains=None,
            **kwargs,
        ):
            nonlocal active_workers, peak_workers
            idx_num = int(claim_item["claim"].split("#")[1])
            with counter_lock:
                active_workers += 1
                peak_workers = max(peak_workers, active_workers)
                outage_seen_in_workers.append(llm.is_simulate_outage())
            try:
                # Stagger completion order so faster items finish first while output keeps input order.
                time.sleep(0.14 if idx_num % 2 == 1 else 0.05)
                return {
                    "claim": claim_item["claim"],
                    "span": claim_item.get("span"),
                    "category": claim_item["category"],
                    "status": "fresh",
                    "score": 80 + idx_num,
                    "current_info": f"Info {idx_num}",
                    "reason": "Ok",
                    "sources": [],
                    "tavily_credits_used": 1.0,
                }
            finally:
                with counter_lock:
                    active_workers -= 1

        def on_prog(done: int, total: int, res: dict):
            callback_threads.append(threading.get_ident())
            progress_events.append((done, total, res["claim"]))

        mock_client = MagicMock()
        try:
            llm.set_simulate_outage(True)
            with patch("verifier.get_tavily_client", return_value=mock_client):
                with patch("verifier.verify_claim", side_effect=fake_verify_single):
                    results = verifier.verify_claims(
                        claims,
                        progress_callback=on_prog,
                        max_workers=3,
                    )
        finally:
            llm.set_simulate_outage(False)

        self.assertGreaterEqual(peak_workers, 2)
        self.assertLessEqual(peak_workers, 3)
        self.assertTrue(outage_seen_in_workers and all(outage_seen_in_workers))
        self.assertEqual([r["claim"] for r in results], [f"Claim #{i}" for i in range(1, 6)])
        self.assertEqual([r["span"] for r in results], [f"span #{i}" for i in range(1, 6)])
        self.assertEqual(len(progress_events), 5)
        self.assertEqual([ev[0] for ev in progress_events], [1, 2, 3, 4, 5])
        self.assertTrue(all(ev[1] == 5 for ev in progress_events))
        self.assertTrue(all(tid == caller_thread_id for tid in callback_threads))

    def test_thread_safe_sqlite_cache_and_credit_accuracy_under_parallel_verify(self):
        claims = [
            {"claim": f"Product X {i}", "span": f"Product X {i}", "category": "statistic", "search_query": f"product x {i}"}
            for i in range(1, 7)
        ]

        mock_client = MagicMock()

        def fake_search(**kwargs):
            q = kwargs.get("query", "")
            return {
                "results": [
                    {
                        "title": "Reuters",
                        "url": f"https://www.reuters.com/item/{abs(hash(q)) % 1000}",
                        "published_date": "2026-10-15",
                        "content": f"Current stats for {q}.",
                    }
                ],
                "usage": {"credits": 1},
            }

        mock_client.search.side_effect = fake_search

        def fake_extract(**kwargs):
            urls = kwargs.get("urls") or []
            return {
                "results": [
                    {
                        "url": urls[0] if urls else "https://www.reuters.com/item/1",
                        "raw_content": "Confirmed figure from live page.",
                    }
                ],
                "failed_results": [],
                "usage": {"credits": 0.2},
            }

        mock_client.extract.side_effect = fake_extract

        def fake_judge(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            return (
                '{"status": "fresh", "score": 88, "agreeing_sources": 1, '
                '"current_info": "Matches latest data.", "reason": "Confirmed."}',
                "openai/gpt-4o-mini",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = os.path.join(tmpdir, "parallel_cache.db")
            with patch("verifier.get_tavily_client", return_value=mock_client):
                with patch("verifier.call_llm", side_effect=fake_judge):
                    first_pass = verifier.verify_claims(
                        claims,
                        cache_path=tmp_db,
                        max_workers=3,
                    )
                    second_pass = verifier.verify_claims(
                        claims,
                        cache_path=tmp_db,
                        max_workers=3,
                    )

        self.assertEqual(len(first_pass), 6)
        self.assertEqual(mock_client.search.call_count, 6)
        self.assertEqual(mock_client.extract.call_count, 6)
        self.assertAlmostEqual(sum(r["tavily_credits_used"] for r in first_pass), 7.2)
        self.assertAlmostEqual(sum(r["tavily_credits_used"] for r in second_pass), 0.0)
        self.assertTrue(all(r["cache_hit"] for r in second_pass))

    def test_extract_article_from_url_validation_cache_truncation_and_errors(self):
        # 1. Invalid URLs raise ValueError before calling Tavily
        for bad_url in ["", "   ", "ftp://example.com/file", "example.com/no-scheme"]:
            with self.subTest(bad_url=bad_url):
                with self.assertRaises(ValueError):
                    verifier.extract_article_from_url(bad_url)

        # 2. Successful extract, 6h cache hit, and truncation
        long_text = "First line.\r\n\r\n" + ("Python 3.13 release details and benchmarks. " * 400)
        mock_client = MagicMock()
        mock_client.extract.return_value = {
            "results": [
                {
                    "url": "https://docs.python.org/3/whatsnew/3.13.html",
                    "title": "What's New In Python 3.13",
                    "raw_content": long_text,
                }
            ],
            "failed_results": [],
            "usage": {"credits": 1},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_db = os.path.join(tmpdir, "url_extract_cache.db")
            first = verifier.extract_article_from_url(
                "https://docs.python.org/3/whatsnew/3.13.html",
                client=mock_client,
                use_cache=True,
                cache_path=tmp_db,
                max_chars=500,
            )
            self.assertEqual(first["url"], "https://docs.python.org/3/whatsnew/3.13.html")
            self.assertEqual(first["title"], "What's New In Python 3.13")
            self.assertLessEqual(len(first["content"]), 500)
            self.assertGreaterEqual(len(first["content"]), 490)
            self.assertTrue(first["truncated"])
            self.assertFalse(first["cache_hit"])
            self.assertEqual(first["credits_used"], 1.0)
            self.assertNotIn("\r", first["content"])

            cached = verifier.extract_article_from_url(
                "https://docs.python.org/3/whatsnew/3.13.html",
                client=mock_client,
                use_cache=True,
                cache_path=tmp_db,
                max_chars=500,
            )
            self.assertTrue(cached["cache_hit"])
            self.assertEqual(cached["credits_used"], 0.0)
            self.assertEqual(mock_client.extract.call_count, 1)

        # 3. Failed URL extraction raises descriptive RuntimeError
        failing_client = MagicMock()
        failing_client.extract.return_value = {
            "results": [],
            "failed_results": [{"url": "https://example.com/blocked", "error": "403 Forbidden"}],
            "usage": {"credits": 0},
        }
        with self.assertRaises(RuntimeError) as ctx:
            verifier.extract_article_from_url(
                "https://example.com/blocked",
                client=failing_client,
                use_cache=False,
            )
        self.assertIn("403 Forbidden", str(ctx.exception))

    def test_worker_tavily_client_isolates_requests_session(self):
        from tavily import TavilyClient

        base_client = TavilyClient(api_key="tvly-test-key-000")
        worker_client, owns_client = verifier._make_worker_tavily_client(base_client)
        try:
            self.assertTrue(owns_client)
            self.assertIsNot(worker_client, base_client)
            self.assertIsNot(worker_client.session, base_client.session)
            self.assertEqual(worker_client.api_key, base_client.api_key)
        finally:
            if owns_client and hasattr(worker_client, "session"):
                worker_client.session.close()
            if hasattr(base_client, "session"):
                base_client.session.close()

    def test_cloud_temp_cache_path_and_unwritable_cache_no_crash(self):
        import sys
        import types

        # 1. Default cache path resolves inside OS temp directory (writable on Streamlit Cloud).
        tmp_root = os.path.normcase(os.path.abspath(tempfile.gettempdir()))
        with patch.dict(os.environ, {"TAVILY_CACHE_PATH": ""}, clear=False):
            default_path = verifier.get_cache_path()
        self.assertTrue(os.path.normcase(os.path.abspath(default_path)).startswith(tmp_root))

        # 2. TAVILY_CACHE_PATH falls back to st.secrets when env is empty.
        fake_st = types.SimpleNamespace(secrets={"TAVILY_CACHE_PATH": "/tmp/custom_st_cache.db"})
        with patch.dict(os.environ, {"TAVILY_CACHE_PATH": ""}, clear=False):
            with patch.dict(sys.modules, {"streamlit": fake_st}):
                self.assertEqual(verifier.get_cache_path(), "/tmp/custom_st_cache.db")

        # 3. Unwritable cache directory / sqlite errors never crash _cache_get, _cache_set, or clear_tavily_cache.
        with patch("verifier.os.makedirs", side_effect=PermissionError("read-only fs")):
            verifier._cache_set("k1", "search", {"results": []}, cache_path="/readonly/cache.db")
        with patch("verifier.os.path.exists", return_value=True), patch(
            "verifier.sqlite3.connect", side_effect=PermissionError("read-only fs")
        ):
            self.assertIsNone(verifier._cache_get("k1", cache_path="/readonly/cache.db"))
            self.assertEqual(verifier.clear_tavily_cache(cache_path="/readonly/cache.db"), 0)
        with patch("verifier.os.path.exists", side_effect=OSError("bad path")):
            self.assertIsNone(verifier._cache_get("k1", cache_path="/readonly/cache.db"))
            self.assertEqual(verifier.clear_tavily_cache(cache_path="/readonly/cache.db"), 0)

    def test_verify_claims_propagates_streamlit_secrets_to_worker_threads(self):
        import sys
        import types

        fake_st = types.SimpleNamespace(
            secrets={
                "USE_BIFROST": False,
                "GEMINI_API_KEY": "AIza-thread-secret",
                "TAVILY_API_KEY": "tvly-thread-secret",
                "ENTITY_DOMAIN_MAP": {"cloudco": ["cloudco.dev"]},
            }
        )
        observed_in_workers: list[tuple[str, str, bool]] = []

        def worker_check(claim_item, **kwargs):
            # Even with os.environ cleared, worker reads snapshotted secrets cleanly.
            observed_in_workers.append(
                (
                    verifier._get_config_value("GEMINI_API_KEY"),
                    verifier._get_config_value("TAVILY_API_KEY"),
                    llm.is_use_bifrost(),
                )
            )
            return {
                "claim": claim_item["claim"],
                "category": claim_item.get("category", "other"),
                "status": "fresh",
                "score": 90,
                "current_info": "Ok",
                "reason": "Ok",
                "sources": [],
                "tavily_credits_used": 0.0,
            }

        mock_tavily = MagicMock()
        claims = [
            {"claim": "Claim 1", "category": "version", "search_query": "q1"},
            {"claim": "Claim 2", "category": "price", "search_query": "q2"},
        ]
        cleared_env = {
            "USE_BIFROST": "",
            "LLM_API_KEY": "",
            "GEMINI_API_KEY": "",
            "TAVILY_API_KEY": "",
            "ENTITY_DOMAIN_MAP": "",
        }
        with patch("llm.load_dotenv"), patch("verifier.load_dotenv"), patch.dict(
            os.environ, cleared_env, clear=False
        ):
            with patch.dict(sys.modules, {"streamlit": fake_st}):
                with patch("verifier.TavilyClient", return_value=mock_tavily) as mock_tv_cls:
                    verifier.get_tavily_client()
                mock_tv_cls.assert_called_once_with(api_key="tvly-thread-secret")

                with patch("verifier.get_tavily_client", return_value=mock_tavily):
                    with patch("verifier.verify_claim", side_effect=worker_check):
                        res = verifier.verify_claims(claims, max_workers=2)

        self.assertEqual(len(res), 2)
        self.assertEqual(
            observed_in_workers,
            [
                ("AIza-thread-secret", "tvly-thread-secret", False),
                ("AIza-thread-secret", "tvly-thread-secret", False),
            ],
        )

    def test_no_false_conflict_on_negated_or_claim_contradiction_text(self):
        today = date(2026, 11, 11)
        sources = [
            {
                "title": "GST Council Notification",
                "url": "https://gstcouncil.gov.in/rates",
                "published_date": "2026-10-15",
                "content": "Standard GST rate updated to 5% and 18% slabs with 40% demerit slab.",
            },
            {
                "title": "CBIC Circular",
                "url": "https://www.cbic.gov.in/circular",
                "published_date": "2026-10-16",
                "content": "Official GST rates are 5% and 18% slabs with 40% demerit rate.",
            },
        ]
        # Phrases like "no conflict" or "official sources contradict the outdated claim"
        # must keep status='outdated' and conflict_detected=False.
        scoring = verifier.compute_freshness_score(
            status="outdated",
            llm_score=10,
            current_info="Current GST rates are 5% and 18% slabs with 40% demerit slab.",
            search_results=sources,
            llm_agreeing_count=2,
            reason="No conflict among official notices: current publications directly contradict the outdated 4-tier claim.",
            entity_domains=["gstcouncil.gov.in", "cbic.gov.in"],
            today=today,
        )
        self.assertFalse(scoring["conflict_detected"])
        self.assertEqual(scoring["resolved_status"], "outdated")
        self.assertLessEqual(scoring["score"], 39)
        self.assertIn("confirm the claim is outdated", scoring["why_this_score"])

    def test_canonical_url_and_long_mirror_snippet_deduplication(self):
        r1 = [
            {
                "title": "OpenAI Pricing (tracking link)",
                "url": "https://www.openai.com/api/pricing/?utm_source=news&utm_medium=referral#models",
                "content": "OpenAI API GPT-4o standard pricing is $2.50 per 1M input tokens and $10.00 per 1M output tokens across tiers.",
            }
        ]
        r2 = [
            {
                "title": "OpenAI Pricing (clean http)",
                "url": "http://openai.com/api/pricing",
                "content": "OpenAI API GPT-4o standard pricing is $2.50 per 1M input tokens and $10.00 per 1M output tokens across tiers.",
            },
            {
                "title": "Mirror Syndication",
                "url": "https://syndicated-mirror.example/repost",
                "content": "OpenAI API GPT-4o standard pricing is $2.50 per 1M input tokens and $10.00 per 1M output tokens across tiers.",
            },
            {
                "title": "Independent Tech News",
                "url": "https://www.reuters.com/tech/openai-api-rates",
                "content": "Reuters confirms OpenAI maintains its $2.50 input and $10 output API rates for GPT-4o.",
            },
        ]
        merged = verifier._merge_search_results(r1, r2, limit=5, prioritize_primary=True)
        urls = [m["url"] for m in merged]
        self.assertEqual(len(merged), 2)
        self.assertTrue(urls[0].startswith("https://www.openai.com/api/pricing"))
        self.assertEqual(urls[1], "https://www.reuters.com/tech/openai-api-rates")

        top = verifier._extract_top_sources(
            merged,
            primary_urls={"http://openai.com/api/pricing/"},
            entity_domains=["openai.com"],
        )
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0]["source_type"], "Primary")

    def test_generate_corrected_answer_sanitizes_and_rejects_malformed_outputs(self):
        long_original = (
            "Apple sells the iPhone 15 Pro Max as its flagship phone for $1,199. "
            "OpenAI GPT-4 Turbo costs $10 per million input tokens. "
            "Python 3.11 is currently the latest stable release series."
        )
        findings = [
            {
                "claim": "Apple sells iPhone 15 Pro Max",
                "status": "outdated",
                "score": 15,
                "current_info": "Apple now sells iPhone 18 Pro Max.",
                "reason": "Newer flagship launched.",
            }
        ]

        # 1. Strips accidental outer code fence + preamble line cleanly
        wrapped = (
            "```markdown\n"
            "Here is the updated answer:\n\n"
            "Apple now sells the iPhone 18 Pro Max. OpenAI GPT-4 Turbo costs $10 per million input tokens. "
            "Python 3.15 is the latest stable release series.\n"
            "```"
        )
        with patch("verifier.call_llm", return_value=(wrapped, "openai/gpt-4o-mini")):
            cleaned, model = verifier.generate_corrected_answer(long_original, findings)
        self.assertEqual(model, "openai/gpt-4o-mini")
        self.assertFalse(cleaned.startswith("Here is"))
        self.assertNotIn("```", cleaned)
        self.assertIn("iPhone 18 Pro Max", cleaned)

        # 2. Rejects empty, JSON, prompt leakage, refusal, and severely truncated outputs
        for bad_output in (
            "   ",
            '{"corrected": "oops"}',
            "Original answer:\nfoo\n\nVerified claim findings:\nbar",
            "I'm sorry, but I cannot rewrite this content.",
            "Too short.",
        ):
            with patch("verifier.call_llm", return_value=(bad_output, "openai/gpt-4o-mini")):
                with self.assertRaises(ValueError):
                    verifier.generate_corrected_answer(long_original, findings)


class TestCorrectnessAndRelevanceGate(unittest.TestCase):
    """Regression tests for evidence-only judging, relevance filtering, validator caps, and cache invalidation."""

    def test_reg9a_iphone_18_pro_with_only_podcasts_returns_unverifiable_never_outdated(self):
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Apple Tech Podcast Ep 42",
                    "url": "https://podcasts.apple.com/us/podcast/iphone-18-pro-rumors/id12345",
                    "published_date": "2026-09-10",
                    "score": 0.62,
                    "content": "Podcast hosts discuss iPhone 18 Pro and Apple iPhone 17.",
                }
            ],
            "usage": {"credits": 1},
        }
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}

        with patch("verifier.call_llm") as mock_llm:
            res = verifier.verify_claim(
                {
                    "claim": "iPhone 18 Pro is available.",
                    "category": "version",
                    "search_query": "iPhone 18 Pro available",
                },
                tavily_client=mock_tavily,
                use_cache=False,
            )

        mock_llm.assert_not_called()
        self.assertEqual(res["status"], "unverifiable")
        self.assertNotEqual(res["status"], "outdated")
        self.assertIsNone(res["score"])
        self.assertEqual(res["confidence"], "low")
        self.assertEqual(res["sources"], [])
        # Primary had 0 relevant hits -> automatically ran general + topic="news" as well (3 calls).
        self.assertEqual(mock_tavily.search.call_count, 3)

    def test_reg9b_iphone_18_pro_with_apple_newsroom_returns_fresh(self):
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "Apple announces iPhone 18 Pro",
                        "url": "https://www.apple.com/newsroom/2026/09/iphone-18-pro/",
                        "published_date": "2026-09-09",
                        "score": 0.94,
                        "content": "Apple announced iPhone 18 Pro on September 9, 2026, with retail availability starting September 18, 2026.",
                    }
                ],
                "usage": {"credits": 1},
            },
            {"results": [], "usage": {"credits": 1}},
            {"results": [], "usage": {"credits": 1}},
        ]
        mock_tavily.extract.return_value = {
            "results": [
                {
                    "url": "https://www.apple.com/newsroom/2026/09/iphone-18-pro/",
                    "raw_content": "iPhone 18 Pro went on sale September 18, 2026 across Apple Store locations.",
                }
            ],
            "failed_results": [],
            "usage": {"credits": 0.2},
        }
        llm_reply = (
            '{"status": "fresh", "score": 96, "agreeing_sources": 1, '
            '"evidence_stance": "supports", '
            '"supporting_quote": "Apple announced iPhone 18 Pro on September 9, 2026, with retail availability starting September 18, 2026.", '
            '"current_info": "iPhone 18 Pro went on sale on September 18, 2026.", '
            '"current_value": "Available since Sep 18, 2026", '
            '"effective_date": "2026-09-18", '
            '"reason": "Apple Newsroom confirms iPhone 18 Pro availability."}',
            "gemini-2.5-flash",
        )

        with patch("verifier.call_llm", return_value=llm_reply):
            res = verifier.verify_claim(
                {
                    "claim": "iPhone 18 Pro is available.",
                    "category": "version",
                    "search_query": "iPhone 18 Pro available",
                },
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(res["status"], "fresh")
        self.assertEqual(res["evidence_stance"], "supports")
        self.assertIn("September 18, 2026", res["supporting_quote"])
        # Single independent relevant domain (apple.com) yields low confidence and score capped to 40..74.
        self.assertEqual(res["confidence"], "low")
        self.assertGreaterEqual(res["score"], 40)
        self.assertLessEqual(res["score"], 74)
        self.assertEqual(res["sources"][0]["source_type"], "Primary")

    def test_reg9c_conflicting_sources_returns_uncertain_with_disagreement_note(self):
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "Reuters Test Report",
                        "url": "https://www.reuters.com/technology/iphone-18-pro-battery-2026/",
                        "published_date": "2026-10-20",
                        "score": 0.89,
                        "content": "Independent tests show the iPhone 18 Pro battery outlasted every competing flagship phone.",
                    },
                    {
                        "title": "The Verge Lab Review",
                        "url": "https://www.theverge.com/reviews/iphone-18-pro-battery",
                        "published_date": "2026-10-22",
                        "score": 0.87,
                        "content": "Counter-tests show iPhone 18 Pro fell behind rival flagship Android batteries in 5G streaming.",
                    },
                ],
                "usage": {"credits": 1},
            },
            {"results": [], "usage": {"credits": 1}},
        ]
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        llm_reply = (
            '{"status": "uncertain", "score": 52, "agreeing_sources": 1, "conflicting_sources": true, '
            '"evidence_stance": "supports", '
            '"supporting_quote": "Independent tests show the iPhone 18 Pro battery outlasted every competing flagship phone.", '
            '"current_info": "Sources disagree on whether iPhone 18 Pro beats all competitors in battery tests.", '
            '"reason": "Sources disagree: Reuters reports top battery life while The Verge reports rivals lasted longer."}',
            "gpt-4o-mini",
        )

        with patch("verifier.call_llm", return_value=llm_reply):
            res = verifier.verify_claim(
                {
                    "claim": "The iPhone 18 Pro battery lasts longer than every competing phone.",
                    "category": "statistic",
                    "search_query": "iPhone 18 Pro battery test comparison",
                },
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(res["status"], "uncertain")
        self.assertTrue(res["conflict_detected"])
        self.assertIn("disagree", res["conflict_note"].lower())
        self.assertIn("disagree", res["reason"].lower())
        self.assertEqual(res["confidence"], "low")
        self.assertGreaterEqual(res["score"], 40)
        self.assertLessEqual(res["score"], 74)

    def test_reg9d_no_invented_dates_when_no_dated_source_or_snippet_exists(self):
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {
                        "title": "MacBook Air Specs",
                        "url": "https://www.apple.com/macbook-air/",
                        "published_date": None,
                        "score": 0.91,
                        "content": "MacBook Air now comes standard with 16 GB of unified memory in the base model.",
                    },
                    {
                        "title": "MacBook Air Compare",
                        "url": "https://www.apple.com/macbook-air/compare/",
                        "published_date": "",
                        "score": 0.88,
                        "content": "Every MacBook Air configuration includes 16 GB base memory.",
                    },
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [
                    {
                        "title": "MacRumors Buyer Guide",
                        "url": "https://www.macrumors.com/guide/macbook-air-memory/",
                        "published_date": None,
                        "score": 0.85,
                        "content": "Apple upgraded entry MacBook Air base RAM from 8 GB to 16 GB.",
                    }
                ],
                "usage": {"credits": 1},
            },
        ]
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        # Mock LLM attempts to hallucinate stale_since & effective_date not present anywhere in snippets.
        llm_hallucinated_dates = (
            '{"status": "outdated", "score": 15, "agreeing_sources": 2, '
            '"evidence_stance": "contradicts", '
            '"supporting_quote": "MacBook Air now comes standard with 16 GB of unified memory in the base model.", '
            '"current_info": "Base MacBook Air ships with 16 GB unified memory.", '
            '"current_value": "16 GB", '
            '"effective_date": "2025-03-01", '
            '"stale_since": "March 2025", '
            '"reason": "Base RAM increased to 16 GB."}',
            "gpt-4o-mini",
        )

        with patch("verifier.call_llm", return_value=llm_hallucinated_dates):
            res = verifier.verify_claim(
                {
                    "claim": "MacBook Air ships with 8 GB base memory.",
                    "category": "version",
                    "search_query": "MacBook Air base memory RAM",
                },
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(res["status"], "outdated")
        self.assertIsNone(res["effective_date"])
        self.assertIsNone(res["stale_since"])
        self.assertIsNone(res["stale_duration_label"])
        self.assertIsNone(res["stale_for_months"])

    def test_relevance_gate_drops_irrelevant_and_triggers_news_fallback(self):
        # 1. Drops podcasts/support/careers/login, low score (<0.35), and wrong product model ("iPhone 17"),
        #    keeping only relevant "iPhone 18 Pro" article on apple.com.
        raw = [
            {"title": "Podcasts", "url": "https://podcasts.apple.com/show/iphone-18", "score": 0.9, "content": "iPhone 18 Pro chat"},
            {"title": "Support", "url": "https://support.apple.com/iphone-18", "score": 0.9, "content": "iPhone 18 Pro help"},
            {"title": "Careers", "url": "https://careers.apple.com/iphone-18", "score": 0.9, "content": "iPhone 18 Pro engineering"},
            {"title": "Sign In", "url": "https://apple.com/account/login", "score": 0.9, "content": "iPhone 18 Pro sign in"},
            {"title": "Low Score", "url": "https://example.com/iphone-18", "score": 0.21, "content": "iPhone 18 Pro brief mention"},
            {"title": "Wrong Model", "url": "https://www.apple.com/iphone-17-pro/", "score": 0.88, "content": "iPhone 17 Pro specs and release."},
            {"title": "Right Model", "url": "https://www.apple.com/newsroom/iphone-18-pro/", "score": 0.82, "content": "Apple unveils iPhone 18 Pro."},
        ]
        filtered = verifier._filter_relevant_results(raw, claim="iPhone 18 Pro is available.", query="iPhone 18 Pro")
        self.assertEqual([r["title"] for r in filtered], ["Right Model"])

        # 2. Fewer than 2 relevant primary results automatically triggers general + topic="news" (time_range=month) + URL dedupe.
        client = MagicMock()
        client.search.side_effect = [
            {
                "results": [
                    {"title": "Podcasts", "url": "https://podcasts.apple.com/ep1", "score": 0.8, "content": "iPhone 18 Pro"},
                    {"title": "Newsroom", "url": "https://www.apple.com/newsroom/iphone-18-pro/?utm_source=x", "score": 0.9, "content": "iPhone 18 Pro announced"},
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [
                    {"title": "Reuters", "url": "https://www.reuters.com/tech/iphone-18-pro", "score": 0.88, "content": "iPhone 18 Pro hits stores"},
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [
                    {"title": "Newsroom Dup", "url": "https://www.apple.com/newsroom/iphone-18-pro/", "score": 0.87, "content": "iPhone 18 Pro on sale"},
                    {"title": "9to5Mac News", "url": "https://9to5mac.com/2026/09/18/iphone-18-pro-sale/", "score": 0.86, "content": "iPhone 18 Pro available today"},
                ],
                "usage": {"credits": 1},
            },
        ]
        search_out = verifier.search_tavily_for_claim(
            query="iPhone 18 Pro available",
            category="version",
            claim="iPhone 18 Pro is available.",
            client=client,
            use_cache=False,
        )
        self.assertEqual(client.search.call_count, 3)
        self.assertEqual(client.search.call_args_list[2].kwargs["topic"], "news")
        self.assertEqual(client.search.call_args_list[2].kwargs["time_range"], verifier.RECENT_NEWS_TIME_RANGE)
        urls = [r["url"] for r in search_out["results"]]
        self.assertEqual(len(urls), 3)  # Canonical URL deduped the duplicate newsroom hit
        self.assertIn("news", search_out["search_strategy"] if "search_strategy" in search_out else search_out["strategy"])

    def test_validator_rules_independent_domain_counting_query_enrichment_and_cache_version(self):
        # 1. no_evidence always forces unverifiable even if LLM guessed fresh/outdated.
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {"title": "Apple Newsroom", "url": "https://www.apple.com/newsroom/iphone-18-pro/", "content": "iPhone 18 Pro event overview.", "score": 0.85},
                    {"title": "Apple Store", "url": "https://store.apple.com/iphone-18-pro/", "content": "iPhone 18 Pro gallery preview.", "score": 0.82},
                ],
                "usage": {"credits": 1},
            },
            {
                "results": [
                    {"title": "MacRumors", "url": "https://www.macrumors.com/iphone-18-pro/", "content": "iPhone 18 Pro specs sheet.", "score": 0.8},
                ],
                "usage": {"credits": 1},
            },
        ]
        mock_tavily.extract.return_value = {"results": [], "usage": {"credits": 0}}
        with patch(
            "verifier.call_llm",
            return_value=(
                '{"status": "outdated", "score": 15, "evidence_stance": "no_evidence", "supporting_quote": null, "current_info": "Unsure", "reason": "No clear proof"}',
                "gpt-4o-mini",
            ),
        ):
            res_no_ev = verifier.verify_claim(
                {"claim": "iPhone 18 Pro is available.", "category": "version", "search_query": "iPhone 18 Pro"},
                tavily_client=mock_tavily,
                use_cache=False,
            )
        self.assertEqual(res_no_ev["status"], "unverifiable")
        self.assertIsNone(res_no_ev["score"])

        # 2. LLM returns status="outdated" WITHOUT a grounded contradicting quote -> downgraded to "uncertain" + capped 40..74 if low confidence.
        mock_tavily.search.side_effect = [
            {
                "results": [
                    {"title": "Apple Newsroom", "url": "https://www.apple.com/newsroom/iphone-18-pro/", "content": "iPhone 18 Pro announced.", "score": 0.85},
                    {"title": "Apple Store", "url": "https://store.apple.com/iphone-18-pro/", "content": "iPhone 18 Pro hardware overview.", "score": 0.82},
                ],
                "usage": {"credits": 1},
            },
            {"results": [], "usage": {"credits": 1}},
        ]
        with patch(
            "verifier.call_llm",
            return_value=(
                '{"status": "outdated", "score": 12, "agreeing_sources": 2, "evidence_stance": "contradicts", '
                '"supporting_quote": "This fabricated quote appears in zero search snippets anywhere.", '
                '"current_info": "iPhone 18 Pro announced.", "reason": "Memory claim"}',
                "gpt-4o-mini",
            ),
        ):
            res_downgraded = verifier.verify_claim(
                {"claim": "iPhone 18 Pro is available.", "category": "version", "search_query": "iPhone 18 Pro"},
                tavily_client=mock_tavily,
                use_cache=False,
            )
        self.assertEqual(res_downgraded["status"], "uncertain")
        # Both apple.com & store.apple.com collapse to 1 registered domain (apple.com) -> low confidence -> 40..74 band!
        self.assertEqual(res_downgraded["confidence"], "low")
        self.assertEqual(res_downgraded["agreement_summary"], "1 of 1 independent sources agree")
        self.assertGreaterEqual(res_downgraded["score"], 40)
        self.assertLessEqual(res_downgraded["score"], 74)

        # 3. Version/product query enrichment includes current Month Year + announced/released/official.
        enriched = verifier._enrich_search_query(
            "iPhone 18 Pro",
            claim="iPhone 18 Pro is available.",
            category="version",
            today=date(2026, 11, 12),
        )
        self.assertIn("November 2026", enriched)
        for kw in ("announced", "released", "official"):
            self.assertIn(kw, enriched.lower())

        # 4. Cache key embeds CACHE_VERSION and clear_tavily_cache clears rows cleanly.
        k_curr = verifier._build_cache_key("search", {"query": "iphone 18 pro"})
        with patch.object(verifier, "CACHE_VERSION", verifier.CACHE_VERSION + 1):
            k_next = verifier._build_cache_key("search", {"query": "iphone 18 pro"})
        self.assertNotEqual(k_curr, k_next)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "clear_test.db")
            verifier._cache_set(k_curr, "search", {"results": [{"url": "https://apple.com"}]}, cache_path=db_path)
            self.assertIsNotNone(verifier._cache_get(k_curr, cache_path=db_path))
            removed = verifier.clear_tavily_cache(cache_path=db_path)
            self.assertEqual(removed, 1)
            self.assertIsNone(verifier._cache_get(k_curr, cache_path=db_path))


class TestVerdictModelAndSixStatuses(unittest.TestCase):
    """Unit tests for the 6-status verdict model, score bands, was_ever_true rules, and date grounding."""

    def test_incorrect_vs_outdated_was_ever_true_normalization(self):
        """When was_ever_true is 'no', status must be forced to 'incorrect', not 'outdated'."""
        # LLM erroneously says 'outdated' on a claim that was never true
        raw_never_true = {
            "status": "outdated",
            "was_ever_true": "no",
            "score": 30,
            "current_value": "Narendra Modi",
            "correction": "Narendra Modi is the Prime Minister of India; Rahul Gandhi has never held this post.",
            "reason": "Rahul Gandhi was never PM.",
        }
        normalized = verifier._normalize_judgement(raw_never_true)
        self.assertEqual(normalized["status"], "incorrect")
        self.assertEqual(normalized["was_ever_true"], "no")
        self.assertLessEqual(normalized["score"], 15)
        self.assertIsNone(normalized.get("change_date"))
        self.assertIsNone(normalized.get("stale_since"))

        # When was_ever_true is 'yes', 'outdated' is preserved
        raw_was_true = {
            "status": "outdated",
            "was_ever_true": "yes",
            "change_date": "2023-04-11",
            "score": 25,
            "current_value": "X Corp.",
            "correction": "Twitter, Inc. was merged into X Corp.",
            "reason": "Twitter renamed and merged in 2023.",
        }
        normalized_true = verifier._normalize_judgement(raw_was_true)
        self.assertEqual(normalized_true["status"], "outdated")
        self.assertEqual(normalized_true["was_ever_true"], "yes")
        self.assertEqual(normalized_true["change_date"], "2023-04-11")

    def test_mocked_verify_claim_incorrect_claim(self):
        """Full verify_claim flow with mocked Tavily and LLM for never-true claim."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "Prime Minister of India - Official Portal",
                    "url": "https://pmindia.gov.in/en/",
                    "content": "Narendra Modi assumed office as the Prime Minister of India on 26 May 2014.",
                    "score": 0.98,
                }
            ]
        }

        mock_llm_resp = json.dumps({
            "status": "incorrect",
            "was_ever_true": "no",
            "change_date": None,
            "current_value": "Narendra Modi",
            "correction": "Narendra Modi is the Prime Minister of India; Rahul Gandhi has never held the office of Prime Minister.",
            "reason": "Authoritative government records confirm Narendra Modi is the Prime Minister. Rahul Gandhi has never served as Prime Minister.",
            "score": 5,
            "confidence": "high",
        })

        claim_input = {
            "claim": "The Prime Minister of India is Rahul Gandhi.",
            "category": "company",
        }

        with patch("verifier.call_llm", return_value=(mock_llm_resp, "gemini-2.5-flash")):
            res = verifier.verify_claim(
                claim_input,
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(res["status"], "incorrect")
        self.assertEqual(res["was_ever_true"], "no")
        self.assertIsNone(res["change_date"])
        self.assertIsNone(res["stale_since"])
        self.assertIsNotNone(res["score"])
        self.assertGreaterEqual(res["score"], 0)
        self.assertLessEqual(res["score"], 15)
        self.assertIn("Narendra Modi", res["correction"])

    def test_mocked_verify_claim_partially_correct_clamping(self):
        """Partially correct claims clamp within the 45-74 score band."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "GST Council Official",
                    "url": "https://gstcouncil.gov.in",
                    "content": "GST slabs include 5%, 12%, 18%, 28%, but threshold rules vary by state.",
                    "score": 0.95,
                }
            ]
        }

        # LLM suggests score 85, which must be clamped into 45-74 for partially_correct
        mock_llm_resp = json.dumps({
            "status": "partially_correct",
            "was_ever_true": "unknown",
            "change_date": None,
            "current_value": "Partially accurate GST details",
            "correction": "The 4 tax slabs are accurate, but threshold requirements differ between goods and services.",
            "reason": "Slabs are correct, but threshold rules are partially inaccurate.",
            "score": 85,
            "confidence": "medium",
        })

        claim_input = {
            "claim": "Under GST, rates are 5%, 12%, 18%, 28% and threshold is always 20 lakh.",
            "category": "regulation",
        }

        with patch("verifier.call_llm", return_value=(mock_llm_resp, "gemini-2.5-flash")):
            res = verifier.verify_claim(
                claim_input,
                tavily_client=mock_tavily,
                use_cache=False,
            )

        self.assertEqual(res["status"], "partially_correct")
        self.assertGreaterEqual(res["score"], 45)
        self.assertLessEqual(res["score"], 74)
        self.assertEqual(res["score_breakdown"]["status_band"], "45-74")

    def test_all_six_score_bands_enforced(self):
        """All 6 status bands are rigorously enforced by compute_freshness_score."""
        today = date(2026, 11, 13)
        dummy_sources = [{"url": "https://example.com/official", "score": 0.9}]

        # 1. Fresh: 75-100
        sc_fresh = verifier.compute_freshness_score("fresh", 50, "Verified current info", dummy_sources, today=today)
        self.assertGreaterEqual(sc_fresh["score"], 75)
        self.assertLessEqual(sc_fresh["score"], 100)
        self.assertEqual(sc_fresh["score_breakdown"]["status_band"], "75-100")

        # 2. Partially correct: 45-74
        sc_part = verifier.compute_freshness_score("partially_correct", 95, "Partially correct info", dummy_sources, today=today)
        self.assertGreaterEqual(sc_part["score"], 45)
        self.assertLessEqual(sc_part["score"], 74)
        self.assertEqual(sc_part["score_breakdown"]["status_band"], "45-74")

        # 3. Uncertain: 40-74
        sc_unc = verifier.compute_freshness_score("uncertain", 10, "Uncertain info", dummy_sources, today=today)
        self.assertGreaterEqual(sc_unc["score"], 40)
        self.assertLessEqual(sc_unc["score"], 74)
        self.assertEqual(sc_unc["score_breakdown"]["status_band"], "40-74")

        # 4. Outdated: 10-39
        sc_out = verifier.compute_freshness_score("outdated", 90, "Outdated info", dummy_sources, today=today)
        self.assertGreaterEqual(sc_out["score"], 10)
        self.assertLessEqual(sc_out["score"], 39)
        self.assertEqual(sc_out["score_breakdown"]["status_band"], "10-39")

        # 5. Incorrect: 0-15
        sc_inc = verifier.compute_freshness_score("incorrect", 50, "Incorrect info", dummy_sources, today=today)
        self.assertGreaterEqual(sc_inc["score"], 0)
        self.assertLessEqual(sc_inc["score"], 15)
        self.assertEqual(sc_inc["score_breakdown"]["status_band"], "0-15")

        # 6. Unverifiable: N/A (None)
        sc_unv = verifier.compute_freshness_score("unverifiable", 80, "No evidence", [], today=today)
        self.assertIsNone(sc_unv["score"])
        self.assertEqual(sc_unv["score_breakdown"]["status_band"], "N/A")

    def test_never_guess_a_date_rule(self):
        """Rule: Never guess a date. If was_ever_true is no, change_date and stale_since must be None.
        If was_ever_true is yes but no date found, keep outdated but set stale_since to None."""
        # Case A: was_ever_true is 'no', LLM hallucinated a date string
        judgement_hallucinated_date = {
            "status": "incorrect",
            "was_ever_true": "no",
            "change_date": "2024-01-01",
            "stale_since": "2024-01-01",
            "reason": "Never true.",
        }
        norm_a = verifier._normalize_judgement(judgement_hallucinated_date)
        self.assertIsNone(norm_a["change_date"], "change_date must be None when was_ever_true is 'no'")
        self.assertIsNone(norm_a["stale_since"], "stale_since must be None when was_ever_true is 'no'")

        # Case B: was_ever_true is 'yes', but no date found in sources
        judgement_no_date = {
            "status": "outdated",
            "was_ever_true": "yes",
            "change_date": None,
            "stale_since": None,
            "reason": "MacBook Air memory was increased to 16 GB, date unknown.",
        }
        norm_b = verifier._normalize_judgement(judgement_no_date)
        self.assertEqual(norm_b["status"], "outdated")
        self.assertEqual(norm_b["was_ever_true"], "yes")
        self.assertIsNone(norm_b["change_date"])
        self.assertIsNone(norm_b["stale_since"])

    def test_dual_judge_outdated_plus_incorrect_resolves_to_incorrect(self):
        """In dual-judge mode, if one judge says outdated and the other says incorrect,
        keep the more severe status (incorrect) with score clamped to 0-15."""
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {
                    "title": "PM India",
                    "url": "https://pmindia.gov.in",
                    "content": "Official portal of PM Narendra Modi.",
                    "score": 0.95,
                }
            ],
            "usage": {"credits": 1},
        }

        # Judge 1 says outdated, Judge 2 says incorrect
        pass_count = [0]
        def fake_two_judges(system_prompt, user_prompt, temperature=0.1, task=None, model_override=None, simulate_outage=None):
            pass_count[0] += 1
            m_name = model_override or "primary-judge"
            if pass_count[0] == 1:
                return (
                    json.dumps({
                        "status": "outdated",
                        "score": 30,
                        "current_value": "Narendra Modi",
                        "reason": "Was changed recently.",
                        "was_ever_true": "yes",
                    }),
                    m_name,
                )
            else:
                return (
                    json.dumps({
                        "status": "incorrect",
                        "score": 5,
                        "current_value": "Narendra Modi",
                        "correction": "Narendra Modi is the PM; Rahul Gandhi was never PM.",
                        "reason": "Never historically true.",
                        "was_ever_true": "no",
                    }),
                    m_name,
                )

        claim_item = {
            "claim": "The Prime Minister of India is Rahul Gandhi.",
            "category": "company",
            "search_query": "Prime Minister of India",
        }

        with patch.dict(
            os.environ,
            {
                "LLM_MODEL_JUDGE": "judge-1",
                "LLM_MODEL_FALLBACKS": "judge-2",
            },
            clear=False,
        ):
            with patch("verifier.call_llm", side_effect=fake_two_judges):
                res = verifier.verify_claim(
                    claim_item,
                    tavily_client=mock_tavily,
                    use_cache=False,
                    dual_judge=True,
                )

        self.assertEqual(res["status"], "incorrect")
        self.assertGreaterEqual(res["score"], 0)
        self.assertLessEqual(res["score"], 15)
        self.assertTrue(res["model_disagreement"])
        self.assertIn("resolved to incorrect", res["model_agreement_summary"].lower())


if __name__ == "__main__":
    unittest.main()
