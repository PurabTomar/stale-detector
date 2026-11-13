"""
test_api.py - Unit tests for the FastAPI REST service in api.py.

Run with:
    python -m unittest test_api.py -v
"""

import os
import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    import api

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    TestClient = None  # type: ignore[assignment,misc]
    api = None  # type: ignore[assignment]


@unittest.skipIf(not HAS_FASTAPI, "FastAPI is not installed; skipping REST API tests")
class TestVerifyAPI(unittest.TestCase):
    def setUp(self):
        if not HAS_FASTAPI:
            self.skipTest("FastAPI is not installed; skipping REST API tests")
        self.client = TestClient(api.app)

    def test_root_info_serves_html_for_browser_and_json_for_api(self):
        # 1. Browser request with text/html -> returns HTML single-page app
        resp_browser = self.client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
        self.assertEqual(resp_browser.status_code, 200)
        self.assertIn("text/html", resp_browser.headers.get("content-type", ""))
        self.assertIn("<!DOCTYPE html>", resp_browser.text)
        self.assertIn("Stale-Answer Detector", resp_browser.text)
        self.assertIn("themeToggleBtn", resp_browser.text)
        self.assertIn("Light mode", resp_browser.text)
        self.assertIn("sadPulse", resp_browser.text)

        # 2. API / curl request with application/json -> returns JSON
        resp_api = self.client.get("/", headers={"accept": "application/json"})
        self.assertEqual(resp_api.status_code, 200)
        self.assertIn("application/json", resp_api.headers.get("content-type", ""))
        data = resp_api.json()
        self.assertEqual(data["service"], "Stale-Answer Detector API")

        # 3. GET /api explicit endpoint -> returns JSON
        resp_direct_api = self.client.get("/api")
        self.assertEqual(resp_direct_api.status_code, 200)
        self.assertEqual(resp_direct_api.json()["service"], "Stale-Answer Detector API")

    def test_health_endpoint_bifrost_and_direct_gemini(self):
        with patch(
            "api.check_gateway_health",
            return_value={"healthy": True, "mode": "bifrost", "status": "ok"},
        ), patch("api.is_use_bifrost", return_value=True):
            resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["routing_mode"], "bifrost")
        self.assertTrue(body["gateway"]["healthy"])

        with patch(
            "api.check_gateway_health",
            return_value={"healthy": True, "mode": "direct_gemini", "status": "direct_gemini"},
        ), patch("api.is_use_bifrost", return_value=False):
            resp_direct = self.client.get("/health")
        self.assertEqual(resp_direct.status_code, 200)
        self.assertEqual(resp_direct.json()["routing_mode"], "direct_gemini")

    def test_post_verify_returns_structured_claims_json(self):
        extracted = [
            {
                "claim": "Apple's latest iPhone is iPhone 15 Pro Max at $1,199.",
                "category": "price",
                "search_query": "latest iPhone model price",
            }
        ]
        extract_metrics = {
            "primary_model": "gpt-4o-mini",
            "model_used": "gpt-4o-mini",
            "fallback_used": False,
            "latency_ms": 85.0,
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
            "timestamp": "14:00:00",
        }
        verified = [
            {
                "claim": "Apple's latest iPhone is iPhone 15 Pro Max at $1,199.",
                "category": "price",
                "search_query": "latest iPhone model price",
                "status": "outdated",
                "score": 20,
                "confidence": "high",
                "risk_level": "critical",
                "current_info": "Newer flagship iPhone models have been released.",
                "reason": "iPhone 15 Pro Max is no longer Apple's latest flagship.",
                "why_this_score": "Score 20/100: LLM judged 'outdated'.",
                "score_breakdown": {"llm_verdict": {"score": 15, "weight": 0.5, "weighted_points": 7.5}},
                "sources": [{"title": "Apple", "url": "https://www.apple.com/iphone/"}],
                "extracted_url": "https://www.apple.com/iphone/",
                "search_strategy": "news (month) + general [advanced]",
                "cache_hit": False,
                "tavily_credits_used": 4.2,
                "model": "gpt-4o-mini",
                "primary_model": "gpt-4o-mini",
                "fallback_used": False,
                "gateway_metrics": {
                    "primary_model": "gpt-4o-mini",
                    "model_used": "gpt-4o-mini",
                    "fallback_used": False,
                    "latency_ms": 140.0,
                    "prompt_tokens": 120,
                    "completion_tokens": 45,
                    "total_tokens": 165,
                    "timestamp": "14:00:01",
                },
            }
        ]

        with patch("api.extract_claims", return_value=(extracted, "gpt-4o-mini")) as mock_ext:
            with patch("api.get_extract_metrics", return_value=extract_metrics):
                with patch("api.verify_claims", return_value=verified) as mock_ver:
                    with patch("api.is_use_bifrost", return_value=True):
                        resp = self.client.post(
                            "/verify",
                            json={"answer": "Apple's latest iPhone is the iPhone 15 Pro Max at $1,199."},
                        )

        self.assertEqual(resp.status_code, 200)
        mock_ext.assert_called_once()
        mock_ver.assert_called_once_with(extracted, use_cache=True)

        data = resp.json()
        self.assertEqual(data["app"], "Stale-Answer Detector")
        self.assertEqual(data["routing_mode"], "bifrost")
        self.assertEqual(data["summary"]["overall_score"], 20)
        self.assertEqual(data["summary"]["overall_tier"], "Stale")
        self.assertEqual(data["summary"]["claims_checked"], 1)
        self.assertEqual(data["summary"]["outdated"], 1)
        self.assertEqual(data["tokens_used"]["total_tokens"], 235)
        self.assertEqual(len(data["claims"]), 1)
        self.assertEqual(data["claims"][0]["status"], "outdated")
        self.assertEqual(data["claims"][0]["sources"][0]["url"], "https://www.apple.com/iphone/")

    def test_post_verify_no_claims_found(self):
        with patch("api.extract_claims", return_value=([], "gemini-2.5-flash")):
            with patch("api.get_extract_metrics", return_value=None):
                with patch("api.verify_claims") as mock_ver:
                    with patch("api.is_use_bifrost", return_value=False):
                        resp = self.client.post("/verify", json={"answer": "Hello! How can I help today?"})

        self.assertEqual(resp.status_code, 200)
        mock_ver.assert_not_called()
        data = resp.json()
        self.assertEqual(data["routing_mode"], "direct_gemini")
        self.assertEqual(data["claims"], [])
        self.assertEqual(data["summary"]["claims_checked"], 0)
        self.assertIsNone(data["summary"]["overall_score"])

    def test_post_verify_validation_and_friendly_errors(self):
        # 1. Empty/blank answer -> 422
        for bad_body in [{}, {"answer": ""}, {"answer": "   "}]:
            resp = self.client.post("/verify", json=bad_body)
            self.assertEqual(resp.status_code, 422)

        # 2. Invalid LLM JSON -> 502 without traceback
        with patch("api.extract_claims", side_effect=ValueError("LLM did not return valid JSON.")):
            resp_json = self.client.post("/verify", json={"answer": "Test answer."})
        self.assertEqual(resp_json.status_code, 502)
        self.assertIn("invalid JSON", resp_json.json()["detail"])

        # 3. Bifrost down -> 503
        with patch(
            "api.extract_claims",
            side_effect=RuntimeError("Bifrost gateway is down or unreachable at http://localhost:8080/v1."),
        ):
            resp_down = self.client.post("/verify", json={"answer": "Test answer."})
        self.assertEqual(resp_down.status_code, 503)
        self.assertIn("Bifrost", resp_down.json()["detail"])

        # 4. Unexpected multi-line traceback error -> 500 with clean line only
        with patch(
            "api.extract_claims",
            side_effect=RuntimeError('Traceback (most recent call last):\n  File "api.py", line 1\nUnexpected failure'),
        ):
            resp_err = self.client.post("/verify", json={"answer": "Test answer."})
        self.assertEqual(resp_err.status_code, 500)
        detail = resp_err.json()["detail"]
        self.assertEqual(detail, "Unexpected failure")
        self.assertNotIn("Traceback", detail)

    def test_post_verify_serializes_unverifiable_structured_fields_and_risk_weighted_score(self):
        extracted = [
            {"claim": "GST rate is 5%.", "category": "regulation", "search_query": "GST rate India"},
            {"claim": "Python 3.11 is latest.", "category": "version", "search_query": "latest Python version"},
            {"claim": "Private codename Zeta launched.", "category": "other", "search_query": "Private codename Zeta"},
        ]
        verified = [
            {
                "claim": "GST rate is 5%.",
                "category": "regulation",
                "search_query": "GST rate India",
                "status": "fresh",
                "score": 90,
                "confidence": "high",
                "risk_level": "critical",
                "current_info": "5% GST applies to essentials.",
                "current_value": "5%",
                "effective_date": "2025-10-01",
                "change_summary": None,
                "stale_since": None,
                "reason": "Confirmed by GST Council.",
                "sources": [{"title": "GST Council", "url": "https://gstcouncil.gov.in"}],
                "model": "gpt-4o-mini",
                "primary_model": "gpt-4o-mini",
                "fallback_used": False,
            },
            {
                "claim": "Python 3.11 is latest.",
                "category": "version",
                "search_query": "latest Python version",
                "status": "outdated",
                "score": 10,
                "confidence": "high",
                "risk_level": "moderate",
                "current_info": "Python 3.14 is latest.",
                "current_value": "Python 3.14",
                "effective_date": "2025-10-07",
                "change_summary": "Python 3.12, 3.13, and 3.14 superseded 3.11",
                "stale_since": "2023-10",
                "reason": "Newer major versions released.",
                "sources": [{"title": "Python.org", "url": "https://www.python.org/downloads/"}],
                "model": "gpt-4o-mini",
                "primary_model": "gpt-4o-mini",
                "fallback_used": False,
            },
            {
                "claim": "Private codename Zeta launched.",
                "category": "other",
                "search_query": "Private codename Zeta",
                "status": "unverifiable",
                "score": None,
                "confidence": "low",
                "risk_level": "low",
                "current_info": "Unverified (no live web evidence found).",
                "current_value": None,
                "effective_date": None,
                "change_summary": None,
                "stale_since": None,
                "reason": "No sources found.",
                "sources": [],
                "model": None,
                "primary_model": "gpt-4o-mini",
                "fallback_used": False,
            },
        ]

        with patch("api.extract_claims", return_value=(extracted, "gpt-4o-mini")):
            with patch("api.get_extract_metrics", return_value=None):
                with patch("api.verify_claims", return_value=verified):
                    with patch("api.is_use_bifrost", return_value=True):
                        resp = self.client.post("/verify", json={"answer": "Mixed claims answer."})

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Risk-weighted: (90*3.0 + 10*1.5) / 4.5 = 63, skipping unverifiable claim.
        self.assertEqual(data["summary"]["overall_score"], 63)
        self.assertEqual(data["summary"]["overall_tier"], "Mixed")
        self.assertEqual(data["summary"]["claims_checked"], 3)
        self.assertEqual(data["summary"]["fresh"], 1)
        self.assertEqual(data["summary"]["outdated"], 1)
        self.assertEqual(data["summary"]["unverifiable"], 1)

        c_outdated = data["claims"][1]
        self.assertEqual(c_outdated["current_value"], "Python 3.14")
        self.assertEqual(c_outdated["effective_date"], "2025-10-07")
        self.assertEqual(c_outdated["change_summary"], "Python 3.12, 3.13, and 3.14 superseded 3.11")
        self.assertEqual(c_outdated["stale_since"], "2023-10")

        c_unv = data["claims"][2]
        self.assertEqual(c_unv["status"], "unverifiable")
        self.assertIsNone(c_unv["score"])

    def test_post_verify_serializes_evidence_layer_fields(self):
        extracted = [
            {"claim": "Python 3.11 is latest.", "category": "version", "search_query": "latest Python version"},
            {"claim": "Acme CEO is Jordan Lee.", "category": "company", "search_query": "Acme CEO"},
        ]
        verified = [
            {
                "claim": "Python 3.11 is latest.",
                "category": "version",
                "search_query": "latest Python version",
                "status": "outdated",
                "score": 12,
                "confidence": "high",
                "risk_level": "moderate",
                "current_info": "Python 3.14 is latest.",
                "current_value": "Python 3.14",
                "stale_since": "2025-10-07",
                "stale_for_months": 13,
                "stale_duration_label": "stale for ~13 months",
                "entity_domains": ["python.org", "docs.python.org"],
                "sources_agreeing": 2,
                "sources_total": 3,
                "agreement_summary": "2 of 3 sources agree",
                "conflict_detected": False,
                "conflict_note": None,
                "reason": "Newer major versions released.",
                "sources": [
                    {
                        "title": "Python Downloads",
                        "url": "https://www.python.org/downloads/",
                        "source_type": "Primary",
                        "published_date": "2025-10-07",
                        "snippet": "Python 3.14.0 stable release announced.",
                    }
                ],
                "model": "gpt-4o-mini",
            },
            {
                "claim": "Acme CEO is Jordan Lee.",
                "category": "company",
                "search_query": "Acme CEO",
                "status": "uncertain",
                "score": 52,
                "confidence": "low",
                "risk_level": "moderate",
                "current_info": "Sources disagree on Acme CEO.",
                "sources_agreeing": 1,
                "sources_total": 2,
                "agreement_summary": "1 of 2 sources agree",
                "conflict_detected": True,
                "conflict_note": "Sources disagree on the current fact (1 of 2 sources agree).",
                "reason": "Sources disagree.",
                "sources": [
                    {
                        "title": "Reuters",
                        "url": "https://www.reuters.com/business/acme",
                        "source_type": "News",
                        "published_date": "2026-11-01",
                        "snippet": "One report states Jordan Lee is CEO.",
                    }
                ],
                "model": "gpt-4o-mini",
            },
        ]

        with patch("api.extract_claims", return_value=(extracted, "gpt-4o-mini")):
            with patch("api.get_extract_metrics", return_value=None):
                with patch("api.verify_claims", return_value=verified):
                    with patch("api.is_use_bifrost", return_value=True):
                        resp = self.client.post("/verify", json={"answer": "Test answer."})

        self.assertEqual(resp.status_code, 200)
        c0, c1 = resp.json()["claims"]
        self.assertEqual(c0["stale_for_months"], 13)
        self.assertEqual(c0["stale_duration_label"], "stale for ~13 months")
        self.assertEqual(c0["entity_domains"], ["python.org", "docs.python.org"])
        self.assertEqual(c0["agreement_summary"], "2 of 3 sources agree")
        self.assertEqual(c0["sources"][0]["source_type"], "Primary")
        self.assertEqual(c0["sources"][0]["published_date"], "2025-10-07")
        self.assertEqual(c0["sources"][0]["snippet"], "Python 3.14.0 stable release announced.")

        self.assertTrue(c1["conflict_detected"])
        self.assertIn("Sources disagree", c1["conflict_note"])
        self.assertEqual(c1["agreement_summary"], "1 of 2 sources agree")
        self.assertEqual(c1["sources"][0]["source_type"], "News")

    def test_post_verify_dual_judge_flag_and_gateway_summary_metrics(self):
        extracted = [
            {"claim": "Acme costs $50/mo.", "category": "price", "search_query": "Acme pricing"}
        ]
        extract_metrics = {
            "primary_model": "gemini/gemini-2.5-flash-lite",
            "model_used": "gemini/gemini-2.5-flash-lite",
            "fallback_used": False,
            "latency_ms": 60.0,
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "total_tokens": 40,
            "task": "extract",
        }
        verified = [
            {
                "claim": "Acme costs $50/mo.",
                "category": "price",
                "search_query": "Acme pricing",
                "status": "uncertain",
                "score": 52,
                "confidence": "low",
                "risk_level": "critical",
                "current_info": "Judges split on $50 vs $79 tier.",
                "reason": "Models disagree.",
                "dual_judge_used": True,
                "judge_models": ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"],
                "judge_verdicts": [
                    {"model": "openai/gpt-4o-mini", "status": "fresh", "score": 88, "current_info": "$50"},
                    {"model": "anthropic/claude-3-5-sonnet-20241022", "status": "outdated", "score": 20, "current_info": "$79"},
                ],
                "model_disagreement": True,
                "model_agreement_summary": "Models disagree",
                "judge_telemetry": [
                    {
                        "primary_model": "openai/gpt-4o-mini",
                        "model_used": "groq/llama-3.3-70b-versatile",
                        "fallback_used": True,
                        "latency_ms": 140.0,
                        "prompt_tokens": 60,
                        "completion_tokens": 20,
                        "total_tokens": 80,
                        "task": "judge",
                    },
                    {
                        "primary_model": "anthropic/claude-3-5-sonnet-20241022",
                        "model_used": "anthropic/claude-3-5-sonnet-20241022",
                        "fallback_used": False,
                        "latency_ms": 160.0,
                        "prompt_tokens": 70,
                        "completion_tokens": 30,
                        "total_tokens": 100,
                        "task": "judge",
                    },
                ],
                "sources": [{"title": "Acme", "url": "https://acme.dev/pricing"}],
                "model": "groq/llama-3.3-70b-versatile",
                "fallback_used": True,
            }
        ]

        with patch("api.extract_claims", return_value=(extracted, "gemini/gemini-2.5-flash-lite")):
            with patch("api.get_extract_metrics", return_value=extract_metrics):
                with patch("api.verify_claims", return_value=verified) as mock_ver:
                    with patch("api.is_use_bifrost", return_value=True):
                        resp = self.client.post(
                            "/verify",
                            json={"answer": "Acme costs $50/mo.", "dual_judge": True},
                        )

        self.assertEqual(resp.status_code, 200)
        mock_ver.assert_called_once_with(extracted, use_cache=True, dual_judge=True)
        body = resp.json()
        self.assertEqual(body["tokens_used"]["requests_count"], 3)
        self.assertEqual(body["tokens_used"]["fallback_count"], 1)
        self.assertEqual(body["tokens_used"]["avg_latency_ms"], 120.0)
        self.assertGreaterEqual(body["tokens_used"]["audit_duration_ms"], 0.0)

        claim0 = body["claims"][0]
        self.assertTrue(claim0["dual_judge_used"])
        self.assertTrue(claim0["model_disagreement"])
        self.assertEqual(claim0["model_agreement_summary"], "Models disagree")
        self.assertEqual(len(claim0["judge_verdicts"]), 2)


if __name__ == "__main__":
    unittest.main()
