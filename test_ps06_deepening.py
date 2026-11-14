"""
test_ps06_deepening.py - Offline mocked unit tests for PS06 deepening requirements.

Covers:
1. Freshness Detection vs Live Verification partitioning by threshold.
2. Verification Packs (api_docs, pricing, regulation, leadership, company_announcement).
3. Source Stance & registered domain independence deduplication (quote <= 25 words).
4. Score breakdown 4-pillar computation (evidence agreement, authority, recency, model agreement).
5. Contribution Ledger (Tavily & Bifrost tracking, session accumulation, cost calculation).
6. "Why Tavily?" (LLM-only re-evaluation) and "Why Bifrost?" (outage failover simulation).
7. Relative-time handling & conflicting sources neutrality.
"""

from datetime import date
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from extractor import assess_claim_volatility, extract_claims
from ledger import compute_audit_ledger, create_empty_ledger, merge_ledgers
from verifier import (
    compute_freshness_score,
    create_skipped_claim_result,
    get_registered_domain,
    get_verification_pack_for_claim,
    load_verification_packs,
    partition_claims_by_threshold,
    run_llm_only_verification,
    run_why_bifrost_simulation,
    search_tavily_for_claim,
    truncate_snippet_words,
    verify_claim,
)


class TestPS06Deepening(unittest.TestCase):

    def test_partition_claims_by_threshold_and_skipped_result(self):
        """Req 1: Partition claims above/below sensitivity threshold and format skipped results."""
        claims = [
            {"claim": "Python 3.14 is the latest release", "category": "version", "time_sensitivity": 85},
            {"claim": "The boiling point of water is 100 degrees Celsius", "category": "fact", "time_sensitivity": 10},
            {"claim": "Anthropic Claude 3.5 Sonnet costs $3 per million input tokens", "category": "price", "time_sensitivity": 75},
            {"claim": "The speed of light is constant in a vacuum", "category": "fact", "time_sensitivity": 5},
        ]
        flagged, skipped = partition_claims_by_threshold(claims, threshold=40)
        self.assertEqual(len(flagged), 2)
        self.assertEqual(len(skipped), 2)
        self.assertEqual(flagged[0]["claim"], "Python 3.14 is the latest release")
        self.assertEqual(flagged[1]["claim"], "Anthropic Claude 3.5 Sonnet costs $3 per million input tokens")
        self.assertEqual(skipped[0]["claim"], "The boiling point of water is 100 degrees Celsius")
        self.assertEqual(skipped[1]["claim"], "The speed of light is constant in a vacuum")

        # Test skipped claim result structure
        skipped_res = create_skipped_claim_result(skipped[0])
        self.assertEqual(skipped_res["status"], "skipped")
        self.assertIsNone(skipped_res["score"])
        self.assertEqual(skipped_res["tavily_credits_used"], 0.0)
        self.assertIn("Stable", skipped_res["reason"])
        self.assertEqual(len(skipped_res["sources"]), 0)

    def test_extractor_attaches_volatility_metadata(self):
        """Req 1: Extractor attaches time_sensitivity (0-100), volatility_class, and reason."""
        ts, vol, reason = assess_claim_volatility("Apple iPhone 16 Pro starts at $999", "price")
        self.assertGreaterEqual(ts, 60)
        self.assertEqual(vol, "price")
        self.assertTrue(len(reason) > 5)

        ts_stable, vol_stable, _ = assess_claim_volatility("The Earth orbits the Sun", "other")
        self.assertLessEqual(ts_stable, 30)
        self.assertEqual(vol_stable, "stable_fact")

    def test_verification_packs_loading_and_matching(self):
        """Req 2: Verification packs loaded from JSON and correctly matched by category."""
        packs = load_verification_packs()
        for expected_pack in ["api_docs", "pricing", "regulation", "leadership", "company_announcement"]:
            self.assertIn(expected_pack, packs)
            p = packs[expected_pack]
            self.assertIn("topic", p)
            self.assertIn("time_range", p)
            self.assertIn("include_domains", p)
            self.assertIn("search_depth", p)
            self.assertIn("extract_targets", p)

        # Matching checks
        pack_name, pack_data = get_verification_pack_for_claim(
            "OpenAI API token rate is $2.50", category="price"
        )
        self.assertEqual(pack_name, "pricing")
        self.assertIn("openai.com", pack_data["include_domains"])

        pack_name, pack_data = get_verification_pack_for_claim(
            "Python 3.13 deprecates legacy pipes", category="version"
        )
        self.assertEqual(pack_name, "api_docs")
        self.assertIn("docs.python.org", pack_data["include_domains"])

        pack_name, pack_data = get_verification_pack_for_claim(
            "India GST Council meets today", category="regulation"
        )
        self.assertEqual(pack_name, "regulation")
        self.assertIn("gov.in", pack_data["include_domains"])

    def test_search_tavily_for_claim_applies_pack_parameters(self):
        """Req 2: search_tavily_for_claim executes using pack include_domains and depth."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Pricing Page", "url": "https://stripe.com/pricing", "content": "Stripe processing fees are 2.9% + 30c"}
            ],
            "usage": {"credits": 1},
        }

        res = search_tavily_for_claim(
            query="Stripe processing fees",
            category="price",
            client=mock_client,
            use_cache=False,
            claim="Stripe processing fees are 2.9% + 30c",
        )
        self.assertTrue(mock_client.search.called)
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(res["results"][0]["url"], "https://stripe.com/pricing")

    def test_source_stance_and_domain_independence(self):
        """Req 3: Stance support/contradict/neutral, quote <= 25 words, registered domain deduplication."""
        # Independence of registered domain
        dom1 = get_registered_domain("https://subdomain.docs.python.org/3/library/os.html")
        dom2 = get_registered_domain("https://peps.python.org/pep-0594/")
        dom3 = get_registered_domain("https://www.reuters.com/technology/apple-launch")
        dom4 = get_registered_domain("https://subdomain.gstcouncil.gov.in/rules")

        self.assertEqual(dom1, "python.org")
        self.assertEqual(dom2, "python.org")
        self.assertEqual(dom3, "reuters.com")
        self.assertEqual(dom4, "gstcouncil.gov.in")  # multi-part public suffix .gov.in (effective TLD+1)

        # Quote truncation (<= 25 words)
        long_quote = "word " * 40
        truncated = truncate_snippet_words(long_quote, max_words=25)
        self.assertLessEqual(len(truncated.replace("...", "").split()), 25)

    def test_full_verify_claim_with_source_evaluations_and_stances(self):
        """Req 3 & 4: verify_claim parses source stances, counts registered domains, outputs breakdown."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {"title": "Release Notes", "url": "https://docs.python.org/release/3.14", "content": "Python 3.14 was released officially."},
                {"title": "Python Blog", "url": "https://blog.python.org/2026/py314", "content": "Python 3.14 is live today."},
                {"title": "Tech News", "url": "https://techcrunch.com/python-314", "content": "Python 3.14 brings JIT improvements."},
            ],
            "usage": {"credits": 1},
        }
        mock_client.extract.return_value = {"results": [], "usage": {"credits": 0}}

        llm_reply_json = json.dumps({
            "status": "fresh",
            "score": 95,
            "evidence_stance": "supports",
            "supporting_quote": "Python 3.14 was released officially.",
            "source_evaluations": [
                {
                    "url": "https://docs.python.org/release/3.14",
                    "stance": "support",
                    "quote": "Python 3.14 was released officially.",
                    "publication_date": "2026-10-01",
                    "source_type": "primary"
                },
                {
                    "url": "https://blog.python.org/2026/py314",
                    "stance": "support",
                    "quote": "Python 3.14 is live today.",
                    "publication_date": "2026-10-02",
                    "source_type": "primary"
                },
                {
                    "url": "https://techcrunch.com/python-314",
                    "stance": "support",
                    "quote": "Python 3.14 brings JIT improvements.",
                    "publication_date": "2026-10-02",
                    "source_type": "news"
                }
            ],
            "was_ever_true": "yes",
            "agreeing_sources": 2,
            "conflicting_sources": False,
            "current_info": "Python 3.14 is the current release.",
            "current_value": "Python 3.14",
            "reason": "Confirmed across official docs and news.",
        })

        with patch("verifier.call_llm", return_value=(llm_reply_json, "gemini-2.5-flash")):
            res = verify_claim(
                {"claim": "Python 3.14 is available", "category": "version", "search_query": "Python 3.14"},
                tavily_client=mock_client,
                use_cache=False,
            )

        self.assertEqual(res["status"], "fresh")
        self.assertGreaterEqual(res["score"], 75)
        # Even though 2 python.org URLs were evaluated, they count as 1 registered domain (python.org) + 1 techcrunch.com = 2 domains
        self.assertEqual(res["support_domains"], 2)
        self.assertEqual(res["contradict_domains"], 0)
        self.assertIn("2 support", res["stance_summary"])

        # Check 4-pillar score breakdown
        bd = res["score_breakdown"]
        self.assertIn("evidence_agreement", bd)
        self.assertIn("authority", bd)
        self.assertIn("recency", bd)
        self.assertIn("model_agreement", bd)

    def test_contribution_ledger_audit_and_session_accumulation(self):
        """Req 5: Contribution ledger accurately tracks Tavily & Bifrost metrics and accumulates."""
        verified_results = [
            {
                "status": "fresh",
                "score": 90,
                "tavily_credits_used": 1.0,
                "search_strategy": "primary-first",
                "extracted_url": "https://docs.python.org/release/3.14",
                "sources": [
                    {"url": "https://docs.python.org/release/3.14", "source_type": "primary", "title": "Docs"},
                    {"url": "https://news.ycombinator.com/item?id=1", "source_type": "forum", "title": "Discussion"},
                ],
                "judge_telemetry": [
                    {
                        "model_used": "gemini-2.5-flash",
                        "prompt_tokens": 800,
                        "completion_tokens": 150,
                        "latency_ms": 350.0,
                        "fallback_used": False,
                    }
                ]
            },
            {
                "status": "outdated",
                "score": 25,
                "tavily_credits_used": 2.0,
                "search_strategy": "news + general",
                "sources": [
                    {"url": "https://reuters.com/tech", "source_type": "news", "title": "Reuters"},
                ],
                "judge_telemetry": [
                    {
                        "model_used": "gemini-3.1-flash-lite",
                        "prompt_tokens": 600,
                        "completion_tokens": 100,
                        "latency_ms": 280.0,
                        "fallback_used": True,
                    }
                ]
            }
        ]

        extract_meta = {
            "model_used": "gemini-2.5-flash",
            "prompt_tokens": 500,
            "completion_tokens": 120,
            "latency_ms": 220.0,
            "fallback_used": False,
        }

        pricing = {
            "gemini-2.5-flash": {"input_per_million": 0.15, "output_per_million": 0.60},
            "gemini-3.1-flash-lite": {"input_per_million": 0.075, "output_per_million": 0.30},
        }

        audit_ledger = compute_audit_ledger(
            claims_results=verified_results,
            extract_meta=extract_meta,
            pricing_table=pricing,
        )

        # Tavily checks
        self.assertEqual(audit_ledger["tavily"]["searches"], 2)
        self.assertEqual(audit_ledger["tavily"]["extracts"], 1)
        self.assertAlmostEqual(audit_ledger["tavily"]["credits"], 3.0)
        self.assertEqual(audit_ledger["tavily"]["primary_sources"], 1)
        self.assertEqual(len(audit_ledger["tavily"]["unique_domains"]), 3)  # python.org, ycombinator.com, reuters.com

        # Bifrost checks
        self.assertEqual(audit_ledger["bifrost"]["calls"], 3)  # extract + 2 judges
        self.assertEqual(audit_ledger["bifrost"]["fallbacks"], 1)
        self.assertEqual(audit_ledger["bifrost"]["tokens_in"], 1900)
        self.assertEqual(audit_ledger["bifrost"]["tokens_out"], 370)
        self.assertEqual(audit_ledger["bifrost"]["total_tokens"], 2270)
        self.assertTrue(audit_ledger["bifrost"]["cost_set"])
        self.assertGreater(audit_ledger["bifrost"]["cost"], 0.0)

        # Accumulation check
        session_ledger = create_empty_ledger()
        session_ledger = merge_ledgers(session_ledger, audit_ledger)
        session_ledger = merge_ledgers(session_ledger, audit_ledger)
        self.assertEqual(session_ledger["tavily"]["searches"], 4)
        self.assertAlmostEqual(session_ledger["tavily"]["credits"], 6.0)
        self.assertEqual(session_ledger["bifrost"]["calls"], 6)
        self.assertEqual(session_ledger["bifrost"]["fallbacks"], 2)

    def test_why_tavily_offline_llm_only_verification(self):
        """Req 6: 'Why Tavily?' re-evaluates claims using ONLY LLM parametric memory."""
        mock_reply = (
            '{"status": "fresh", "current_value": "$100", "confidence": "high", "reason": "Remembered from 2021 training."}',
            "mock-llm",
        )
        claims = [
            {"claim": "Product X costs $100", "category": "price"},
        ]
        with patch("verifier._invoke_call_llm", return_value=mock_reply):
            llm_results = run_llm_only_verification(claims)

        self.assertEqual(len(llm_results), 1)
        self.assertEqual(llm_results[0]["llm_status"], "fresh")
        self.assertEqual(llm_results[0]["llm_value"], "$100")
        self.assertEqual(llm_results[0]["llm_confidence"], "high")

    def test_why_bifrost_simulation_outage_failover_timeline(self):
        """Req 6: 'Why Bifrost?' forces primary model outage and produces failover timeline."""
        mock_res = {
            "status": "fresh",
            "score": 90,
            "model": "gemini-3.1-flash-lite",
            "fallback_used": True,
        }
        claim_item = {"claim": "FastAPI is asynchronous", "category": "version"}
        with patch("verifier.verify_claim", return_value=mock_res):
            sim = run_why_bifrost_simulation(claim_item, primary_model="gemini-2.5-flash", fallback_model="gemini-3.1-flash-lite")

        self.assertTrue(sim["success"])
        self.assertEqual(sim["primary_failed"], "gemini-2.5-flash")
        self.assertEqual(sim["answering_model"], "gemini-3.1-flash-lite")
        self.assertGreaterEqual(len(sim["timeline"]), 3)
        events = [t["event"] for t in sim["timeline"]]
        self.assertIn("Request Dispatched", events)
        self.assertIn("Primary Provider Outage / 503", events)
        self.assertIn("Bifrost Circuit-Breaker Failover", events)
        self.assertIn("Fallback Response Received", events)

    def test_relative_time_and_conflicting_neutrality(self):
        """Req 7 & 8: Relative-time handling & political/sensitive conflicting neutrality."""
        # Score computation bands
        fresh_res = compute_freshness_score(status="fresh", llm_score=90, current_info="Confirmed", search_results=[])
        self.assertGreaterEqual(fresh_res["score"], 75)
        self.assertLessEqual(fresh_res["score"], 100)

        outdated_res = compute_freshness_score(status="outdated", llm_score=80, current_info="Old info", search_results=[])
        self.assertLessEqual(outdated_res["score"], 39)
        self.assertGreaterEqual(outdated_res["score"], 10)

        incorrect_res = compute_freshness_score(status="incorrect", llm_score=50, current_info="Never true", search_results=[])
        self.assertLessEqual(incorrect_res["score"], 15)

        unverifiable_res = compute_freshness_score(status="unverifiable", llm_score=80, current_info="No info", search_results=[])
        self.assertIsNone(unverifiable_res["score"])

        # Conflicting sources neutrality
        conflict_res = compute_freshness_score(status="uncertain", llm_score=55, current_info="Sources disagree", search_results=[], llm_conflicting=True)
        self.assertTrue(conflict_res["conflict_detected"])
        self.assertEqual(conflict_res["resolved_status"], "uncertain")


if __name__ == "__main__":
    unittest.main()
