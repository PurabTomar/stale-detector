"""
test_api.py - Unit tests for the FastAPI REST service in api.py.

Run with:
    python -m unittest test_api.py -v
"""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api


class TestVerifyAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

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


if __name__ == "__main__":
    unittest.main()
