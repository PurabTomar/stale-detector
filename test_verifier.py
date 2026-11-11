"""
test_verifier.py - Unit tests for explainable freshness scoring in verifier.py.

Run with:
    python -m unittest test_verifier.py
"""

from datetime import date
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
        # 1. Tavily failure returns safe fallback breakdown + low confidence + risk_level.
        failing_tavily = MagicMock()
        failing_tavily.search.side_effect = RuntimeError("Network timeout")
        res_tavily_err = verifier.verify_claim(
            {"claim": "GST rate is 18%.", "category": "regulation", "search_query": "GST rate"},
            tavily_client=failing_tavily,
            use_cache=False,
        )
        self.assertEqual(res_tavily_err["status"], "uncertain")
        self.assertEqual(res_tavily_err["confidence"], "low")
        self.assertEqual(res_tavily_err["risk_level"], "critical")
        self.assertIn("score_breakdown", res_tavily_err)
        self.assertIn("why_this_score", res_tavily_err)
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
        self.assertEqual(res_llm_err["confidence"], "low")
        self.assertEqual(res_llm_err["risk_level"], "critical")
        self.assertEqual(len(res_llm_err["sources"]), 1)
        self.assertIn("score_breakdown", res_llm_err)
        self.assertIn("why_this_score", res_llm_err)
        self.assertNotIn("Traceback", res_llm_err["error"])

    def test_score_clamping_empty_sources_and_confidence_rules(self):
        today = date(2026, 11, 11)

        # Out-of-range LLM scores clamp cleanly to 0..100, and empty sources use 50/0/50 defaults.
        empty_high = verifier.compute_freshness_score(
            status="fresh",
            llm_score=140,
            current_info="Confirmed.",
            search_results=[],
            llm_agreeing_count=3,
            today=today,
        )
        # 50% of 100 + 20% of 50 + 20% of 0 + 10% of 50 = 50 + 10 + 0 + 5 = 65
        self.assertEqual(empty_high["score"], 65)
        self.assertEqual(empty_high["score_breakdown"]["llm_verdict"]["score"], 100)
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
        # 50% of 0 + 20% of 50 + 20% of 0 + 10% of 50 = 15
        self.assertEqual(empty_low["score"], 15)
        self.assertEqual(empty_low["score_breakdown"]["llm_verdict"]["score"], 0)

        # Uncertain status never earns 'high' confidence even with recent official unanimous sources
        recent_official = [
            {
                "title": f"Gov {i}",
                "url": f"https://www.sec.gov/rule-{i}",
                "published_date": "2026-11-10",
                "content": "Filing rule update",
            }
            for i in range(4)
        ]
        uncertain_cap = verifier.compute_freshness_score(
            status="uncertain",
            llm_score=70,
            current_info="Filing rule update",
            search_results=recent_official,
            llm_agreeing_count=4,
            today=today,
        )
        self.assertEqual(uncertain_cap["confidence"], "medium")

    def test_friendly_tavily_invalid_key_and_limit_errors(self):
        # 1. Invalid Tavily API key returns crisp beginner-friendly message without tracebacks
        bad_key_client = MagicMock()
        bad_key_client.search.side_effect = InvalidAPIKeyError("Invalid API key")
        res_bad_key = verifier.verify_claim(
            {"claim": "GPT-4 costs $10.", "category": "price", "search_query": "GPT-4 pricing"},
            tavily_client=bad_key_client,
            use_cache=False,
        )
        self.assertEqual(res_bad_key["status"], "uncertain")
        self.assertIn("TAVILY_API_KEY", res_bad_key["error"])
        self.assertNotIn("Traceback", res_bad_key["error"])

        # 2. Exhausted Tavily limit returns friendly quota/limit message
        limit_client = MagicMock()
        limit_client.search.side_effect = UsageLimitExceededError("Monthly credit limit exceeded")
        with patch("verifier.time.sleep"):
            res_limit = verifier.verify_claim(
                {"claim": "GPT-4 costs $10.", "category": "statistic", "search_query": "GPT-4 stats"},
                tavily_client=limit_client,
                use_cache=False,
            )
        self.assertIn("429", res_limit["error"])
        self.assertIn("limit", res_limit["error"].lower())
        self.assertNotIn("Traceback", res_limit["error"])


class TestTavilyUpgrade(unittest.TestCase):
    """Tests for category-specific search strategies, Tavily Extract, 6h cache, and rate limits."""

    def test_category_search_strategies_and_advanced_depth_gating(self):
        # 1. price (high-risk) -> news + general, BOTH search_depth="advanced"
        price_client = MagicMock()
        price_client.search.side_effect = [
            {"results": [{"title": "News Price", "url": "https://reuters.com/apple-price", "content": "Price update"}], "usage": {"credits": 2}},
            {"results": [{"title": "Apple Store", "url": "https://apple.com/iphone", "content": "iPhone pricing"}], "usage": {"credits": 2}},
        ]
        res_price = verifier.search_tavily_for_claim(
            query="iPhone 15 Pro price",
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

        # 2. version & company (moderate risk) -> news + general, search_depth="basic" (NOT advanced)
        for cat in ("version", "company"):
            mod_client = MagicMock()
            mod_client.search.return_value = {
                "results": [{"title": "Release", "url": "https://docs.python.org/3/", "content": "Latest info"}],
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

        # 3. regulation (high-risk) -> official/gov include_domains + include_domains_mode="prefer" + advanced
        reg_client = MagicMock()
        reg_client.search.return_value = {
            "results": [{"title": "GST Portal", "url": "https://www.gst.gov.in/rates", "content": "GST slabs"}],
            "usage": {"credits": 2},
        }
        verifier.search_tavily_for_claim(
            query="India GST slabs",
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
                {"title": "Official GST Portal", "url": "https://www.gst.gov.in/rules", "content": "GST rule summary"},
                {"title": "Personal Blog", "url": "https://medium.com/@author/gst", "content": "Old guide"},
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
        self.assertAlmostEqual(res["tavily_credits_used"], 2.2)

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
        self.assertAlmostEqual(res_fallback["tavily_credits_used"], 2.0)

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
                    }
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
            {"results": [{"title": "Source", "url": "https://example.com", "content": "Data"}], "usage": {"credits": 1}},
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
            {"results": [{"title": "Official Bio", "url": "https://openai.com/about", "content": "Sam Altman CEO"}], "usage": {"credits": 1}},
        ]
        partial_client.extract.return_value = {
            "results": [{"url": "https://openai.com/about", "raw_content": "Sam Altman is CEO of OpenAI."}],
            "usage": {"credits": 0.2},
        }
        llm_reply = (
            '{"status": "fresh", "score": 95, "agreeing_sources": 1, '
            '"current_info": "Sam Altman is CEO.", "reason": "Confirmed."}',
            "gpt-4o-mini",
        )
        with patch("verifier.time.sleep"):
            with patch("verifier.call_llm", return_value=llm_reply):
                res = verifier.verify_claim(
                    {"claim": "Sam Altman is CEO of OpenAI.", "category": "company", "search_query": "OpenAI CEO"},
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


if __name__ == "__main__":
    unittest.main()
