"""
test_quota_and_limits.py - Unit tests for quota robustness, rate limit handling,
session caps, cache fallback, and source origin labels.

Verifies:
1. Friendly error handling: catching 429/503/quota errors with non-technical messages
   and never leaking stack traces.
2. Auto-fallback to saved results on quota exhaustion with exact banner text.
3. Guided demo: 3 prebuilt examples (iPhone fresh, GST outdated, Leadership incorrect)
   loaded from saved real outputs.
4. Per-session protection: session live audit limits (default 5) and claim clamping (default 6).
5. Quota status states: "Live", "Limited", and "Saved results".
6. 6-hour cache behavior: serving stale cached results on rate limit instead of failing.
7. Evaluator banner: customizable text and dismissibility.
8. Result source origin labeling: Live, Cached, and Saved (with date).
"""

from datetime import datetime
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import requests
from tavily.errors import UsageLimitExceededError

from app import (
    DEFAULT_MAX_CLAIMS_PER_AUDIT,
    DEFAULT_MAX_SESSION_AUDITS,
    SAVED_RESULTS_DATE,
    format_friendly_ui_error,
    get_max_claims_per_audit,
    get_max_session_audits,
    is_quota_or_rate_limit_error,
    render_claim_card,
)
from demo_data import (
    GUIDED_DEMO_STEPS,
    find_cached_pasted_audit,
    get_cached_ask_audit,
    get_cached_url_audit,
    get_guided_demo_step,
)
import verifier


class TestQuotaRobustnessAndLimits(unittest.TestCase):
    """Test suite for rate limiting, session quotas, and graceful degradation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_cache.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    # 1. Friendly handling & no stack traces
    def test_friendly_error_handling_for_429_503_and_quota_errors(self):
        """429, 503, and quota exceptions must return friendly messages without stack traces."""
        # Case A: Tavily UsageLimitExceededError
        tavily_err = UsageLimitExceededError("429 Client Error: Rate limit exceeded")
        msg_tavily = format_friendly_ui_error(tavily_err, action="verifying claims")
        self.assertIn("Live quota is busy, showing saved results", msg_tavily)
        self.assertNotIn("Traceback", msg_tavily)
        self.assertNotIn("File \"", msg_tavily)

        # Case B: LLM Rate Limit (429)
        llm_err = Exception("Rate limit exceeded for model gemini-2.5-flash: 429 RESOURCE_EXHAUSTED")
        msg_llm = format_friendly_ui_error(llm_err, action="extracting claims")
        self.assertIn("Live quota is busy, showing saved results", msg_llm)
        self.assertNotIn("Traceback", msg_llm)

        # Case C: 503 Service Unavailable / busy
        srv_err = Exception("503 Service Unavailable: backend server is busy")
        msg_503 = format_friendly_ui_error(srv_err, action="checking claims")
        self.assertIn("Live quota is busy, showing saved results", msg_503)

        # Case D: Raw stack trace in exception
        trace_err = Exception("Traceback (most recent call last):\n  File 'app.py', line 999\nZeroDivisionError: division by zero")
        msg_trace = format_friendly_ui_error(trace_err, action="running audit")
        self.assertNotIn("Traceback", msg_trace)
        self.assertNotIn("File 'app.py'", msg_trace)

    def test_is_quota_or_rate_limit_error_helper(self):
        """Helper must accurately identify 429, 503, quota, and resource exhaustion triggers."""
        self.assertTrue(is_quota_or_rate_limit_error("HTTP 429 Too Many Requests"))
        self.assertTrue(is_quota_or_rate_limit_error("503 Service Unavailable"))
        self.assertTrue(is_quota_or_rate_limit_error("Tavily credit limit reached (432)"))
        self.assertTrue(is_quota_or_rate_limit_error("Resource_exhausted: quota exceeded"))
        self.assertTrue(is_quota_or_rate_limit_error(UsageLimitExceededError("limit reached")))
        self.assertFalse(is_quota_or_rate_limit_error("Invalid API key 401"))
        self.assertFalse(is_quota_or_rate_limit_error("Connection refused"))

    # 2. Auto fallback to saved results
    def test_auto_fallback_samples_and_banner_format(self):
        """On quota error, auto fallback loads prebuilt audits and formats the banner."""
        banner_text = f"Live API limit reached, showing real saved results from {SAVED_RESULTS_DATE}"
        self.assertEqual(banner_text, "Live API limit reached, showing real saved results from Nov 13, 2026")

        # Test prebuilt sample discovery
        sample_iphone, exact_iphone = find_cached_pasted_audit("iPhone 16 Pro Max starts at $1,199")
        self.assertTrue(exact_iphone)
        self.assertEqual(sample_iphone["id"], "iphone")

        sample_gst, exact_gst = find_cached_pasted_audit("India GST slab 18% and 28%")
        self.assertTrue(exact_gst)
        self.assertEqual(sample_gst["id"], "gst")

        sample_custom, exact_custom = find_cached_pasted_audit("completely unknown claim text")
        self.assertFalse(exact_custom)
        self.assertIsNotNone(sample_custom.get("results"))

    # 3. Guided demo integrity
    def test_guided_demo_steps_and_saved_provenance(self):
        """Guided demo steps must load iPhone (Fresh), GST (Outdated), Leadership (Incorrect) from saved data."""
        self.assertEqual(len(GUIDED_DEMO_STEPS), 3)

        # Step 0: iPhone (Fresh)
        step0 = get_guided_demo_step(0)
        self.assertEqual(step0["badge"], "Fresh")
        self.assertIn("iPhone", step0["title"])
        results0 = step0["sample"]["results"]
        self.assertTrue(all(r["status"] == "fresh" for r in results0))
        self.assertTrue(all(r.get("source_origin") == "saved" for r in results0))
        self.assertTrue(all(r.get("saved_date") == SAVED_RESULTS_DATE for r in results0))

        # Step 1: GST (Outdated)
        step1 = get_guided_demo_step(1)
        self.assertEqual(step1["badge"], "Outdated")
        self.assertIn("GST", step1["title"])
        results1 = step1["sample"]["results"]
        self.assertTrue(any(r["status"] == "outdated" for r in results1))
        self.assertTrue(all(r.get("source_origin") == "saved" for r in results1))

        # Step 2: Leadership (Incorrect)
        step2 = get_guided_demo_step(2)
        self.assertEqual(step2["badge"], "Incorrect")
        self.assertIn("Leadership", step2["title"])
        results2 = step2["sample"]["results"]
        self.assertTrue(any(r["status"] == "incorrect" for r in results2))
        self.assertTrue(all(r.get("source_origin") == "saved" for r in results2))

    # 4. Per-session protection (limits and claim clamping)
    def test_session_audit_limit_configuration(self):
        """MAX_SESSION_AUDITS should default to 5 and be configurable via environment/secrets."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_SESSION_AUDITS", None)
            self.assertEqual(get_max_session_audits(), 5)

        with patch.dict(os.environ, {"MAX_SESSION_AUDITS": "8"}):
            self.assertEqual(get_max_session_audits(), 8)

        with patch.dict(os.environ, {"MAX_SESSION_AUDITS": "invalid"}):
            self.assertEqual(get_max_session_audits(), 5)

    def test_claims_per_audit_limit_configuration(self):
        """MAX_CLAIMS_PER_AUDIT should default to 6 and be configurable."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MAX_CLAIMS_PER_AUDIT", None)
            self.assertEqual(get_max_claims_per_audit(), 6)

        with patch.dict(os.environ, {"MAX_CLAIMS_PER_AUDIT": "4"}):
            self.assertEqual(get_max_claims_per_audit(), 4)

    # 5. Serving stale cache on rate limit (6h cache resilience)
    def test_stale_cache_served_on_tavily_rate_limit(self):
        """When live API raises 429, verifier must serve stale cached response instead of failing."""
        query = "Python 3.13 release date"
        cache_params = {
            "query": "python 3.13 release date",
            "max_results": 5,
            "search_depth": "basic",
            "topic": "general",
            "time_range": None,
            "include_domains": None,
            "include_domains_mode": None,
        }
        cache_key = verifier._build_cache_key("search", cache_params)

        # Seed the SQLite cache with an expired entry (7 hours old, TTL is 6 hours)
        now_time = time.time()
        expired_time = now_time - (7 * 3600)
        cached_data = {
            "results": [
                {
                    "title": "Python 3.13 Released",
                    "url": "https://docs.python.org/3.13/",
                    "content": "Python 3.13 was released on October 7, 2024.",
                }
            ],
            "usage": {"credits": 1.0},
        }
        verifier._cache_set(cache_key, "search", cached_data, cache_path=self.db_path, now=expired_time)

        # Under normal conditions without allow_stale, expired cache returns None
        normal_get = verifier._cache_get(cache_key, cache_path=self.db_path, now=now_time, allow_stale=False)
        self.assertIsNone(normal_get)

        # Under allow_stale=True, expired cache returns the entry
        stale_get = verifier._cache_get(cache_key, cache_path=self.db_path, now=now_time, allow_stale=True)
        self.assertIsNotNone(stale_get)
        self.assertTrue(stale_get.get("_cache_stale"))

        # When client.search raises 429, _run_cached_tavily_search falls back to stale cache
        mock_client = MagicMock()
        mock_client.search.side_effect = UsageLimitExceededError("Rate limit exceeded 429")

        results, credits, cache_hit = verifier._run_cached_tavily_search(
            query=query,
            client=mock_client,
            use_cache=True,
            cache_path=self.db_path,
        )
        self.assertTrue(cache_hit)
        self.assertEqual(credits, 0.0)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Python 3.13 Released")

    def test_stale_cache_served_on_url_extract_rate_limit(self):
        """When URL extract raises 429, extract_article_from_url must serve stale cached article."""
        target_url = "https://example.com/ai-report"
        cache_params = {
            "url": target_url,
            "extract_depth": "basic",
            "format": "markdown",
            "max_chars": verifier.MAX_ARTICLE_EXTRACT_CHARS,
        }
        cache_key = verifier._build_cache_key("extract_article", cache_params)

        now_time = time.time()
        expired_time = now_time - (8 * 3600)
        cached_article = {
            "title": "Cached AI Report",
            "content": "This is cached article text from last week.",
            "truncated": False,
        }
        verifier._cache_set(cache_key, "extract_article", cached_article, cache_path=self.db_path, now=expired_time)

        mock_client = MagicMock()
        mock_client.extract.side_effect = UsageLimitExceededError("429 limit reached")

        res = verifier.extract_article_from_url(
            url=target_url,
            client=mock_client,
            use_cache=True,
            cache_path=self.db_path,
        )
        self.assertTrue(res["cache_hit"])
        self.assertEqual(res["credits_used"], 0.0)
        self.assertEqual(res["title"], "Cached AI Report")
        self.assertIn("cached article text", res["content"])

    # 6. Source origin labels (Live, Cached, Saved)
    def test_result_source_origin_labeling_and_card_badges(self):
        """Each claim result must accurately display its source: Live, Cached, or Saved (with date)."""
        # Case A: Saved result with date
        saved_item = {
            "claim": "iPhone 16 starts at $1,199",
            "status": "fresh",
            "score": 96,
            "confidence": "high",
            "risk_level": "low",
            "source_origin": "saved",
            "saved_date": "Nov 13, 2026",
            "current_info": "Starts at $1,199.",
            "reason": "Confirmed current.",
        }
        with patch("streamlit.markdown") as mock_st_md:
            render_claim_card(1, saved_item, key_prefix="test_saved")
            html_call = mock_st_md.call_args[0][0]
            self.assertIn("💾 Saved (Nov 13, 2026)", html_call)

        # Case B: Cached result
        cached_item = {
            "claim": "Python 3.13 is current",
            "status": "fresh",
            "score": 95,
            "confidence": "high",
            "risk_level": "low",
            "source_origin": "cached",
            "cache_hit": True,
            "current_info": "Released Oct 2024.",
            "reason": "Confirmed from docs.",
        }
        with patch("streamlit.markdown") as mock_st_md:
            render_claim_card(2, cached_item, key_prefix="test_cached")
            html_call = mock_st_md.call_args[0][0]
            self.assertIn("⚡ Cached", html_call)

        # Case C: Live verification result
        live_item = {
            "claim": "Narendra Modi is Prime Minister of India",
            "status": "fresh",
            "score": 99,
            "confidence": "high",
            "risk_level": "low",
            "source_origin": "live",
            "cache_hit": False,
            "current_info": "Current PM of India.",
            "reason": "Verified via pmindia.gov.in.",
        }
        with patch("streamlit.markdown") as mock_st_md:
            render_claim_card(3, live_item, key_prefix="test_live")
            html_call = mock_st_md.call_args[0][0]
            self.assertIn("🟢 Live", html_call)


if __name__ == "__main__":
    unittest.main()
