"""Offline mocked unit tests for:
1. Freshness Guard (`stale_guard/` package + FastAPI `POST /guard` + chatbot wrapper)
2. Cited Corrections (`generate_corrected_answer`, `build_draft_corrected_answer`,
   Markdown export, and HTML export with [1], [2] primary-source citations & [unverified])
3. Deployment Readiness (`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `Makefile`,
   offline GitHub Actions CI, and README disclosures)
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import reporting
from stale_guard import (
    MAX_GUARD_ANSWER_CHARS,
    MAX_GUARD_CONTEXT_CHARS,
    MAX_GUARD_QUESTION_CHARS,
    compute_guard_verdict,
    guard,
)
import verifier


REPO_ROOT = Path(__file__).resolve().parent


class TestCitedCorrections(unittest.TestCase):
    """Feature 2: Numbered [1], [2] primary-source citations & untouched [unverified] claims."""

    def setUp(self):
        self.original = (
            "Python 3.11 is the latest stable release. "
            "OpenAI GPT-4 Turbo costs $10 per million input tokens. "
            "Acme Corp plans to launch Widget X on Mars next month."
        )
        self.results = [
            {
                "claim": "Python 3.11 is the latest stable release.",
                "category": "version",
                "risk_level": "moderate",
                "status": "outdated",
                "score": 15,
                "confidence": "high",
                "current_info": "Python 3.13 is the latest stable release.",
                "current_value": "Python 3.13",
                "reason": "Newer Python release is available.",
                "sources": [
                    {
                        "title": "Random Blog",
                        "url": "https://blog.example.com/python",
                        "domain": "blog.example.com",
                        "source_type": "Blog",
                    },
                    {
                        "title": "Python Downloads Official",
                        "url": "https://www.python.org/downloads/",
                        "domain": "python.org",
                        "source_type": "Primary",
                        "is_primary": True,
                        "published_date": "2026-10-01",
                    },
                ],
            },
            {
                "claim": "OpenAI GPT-4 Turbo costs $10 per million input tokens.",
                "category": "pricing",
                "risk_level": "critical",
                "status": "incorrect",
                "score": 5,
                "confidence": "high",
                "current_info": "OpenAI GPT-4o standard input pricing is $2.50 per million tokens.",
                "current_value": "$2.50 / 1M input",
                "reason": "Contradicted by OpenAI pricing page.",
                "sources": [
                    {
                        "title": "OpenAI API Pricing",
                        "url": "https://openai.com/api/pricing/",
                        "domain": "openai.com",
                        "source_type": "Primary",
                        "is_primary": True,
                    }
                ],
            },
            {
                "claim": "Acme Corp plans to launch Widget X on Mars next month.",
                "category": "company",
                "risk_level": "low",
                "status": "unverifiable",
                "score": None,
                "confidence": "low",
                "current_info": "No reliable web sources found.",
                "reason": "No sources corroborated this claim.",
                "sources": [],
            },
        ]

    def test_build_citation_map_picks_primary_sources_and_tracks_unverified(self):
        cmap = verifier.build_citation_map(self.results)
        self.assertEqual(len(cmap["citations"]), 2)
        # Claim 1 must pick Primary python.org over Blog
        self.assertEqual(cmap["citations"][0]["number"], 1)
        self.assertEqual(cmap["citations"][0]["url"], "https://www.python.org/downloads/")
        self.assertEqual(cmap["citations"][1]["number"], 2)
        self.assertEqual(cmap["citations"][1]["url"], "https://openai.com/api/pricing/")
        self.assertEqual(cmap["claim_to_citation"], {1: 1, 2: 2})
        self.assertEqual(len(cmap["unverified_claims"]), 1)
        self.assertEqual(cmap["unverified_claims"][0]["status"], "unverifiable")

    def test_generate_corrected_answer_includes_citations_sources_and_unverified_marker(self):
        mock_rewritten = (
            "Python 3.13 is the latest stable release [1]. "
            "OpenAI GPT-4o standard input pricing is $2.50 per million tokens [2]. "
            "Acme Corp plans to launch Widget X on Mars next month."
        )
        with patch("verifier.call_llm", return_value=(mock_rewritten, "gemini-3.1-flash-lite")):
            corrected, model = verifier.generate_corrected_answer(self.original, self.results)

        self.assertEqual(model, "gemini-3.1-flash-lite")
        self.assertIn("[1]", corrected)
        self.assertIn("[2]", corrected)
        self.assertIn("Sources:", corrected)
        self.assertIn("[1] Python Downloads Official — https://www.python.org/downloads/", corrected)
        self.assertIn("[2] OpenAI API Pricing — https://openai.com/api/pricing/", corrected)
        # Unverifiable claim stays untouched verbatim and is marked [unverified]
        self.assertIn("Acme Corp plans to launch Widget X on Mars next month. [unverified]", corrected)

    def test_draft_corrected_answer_and_markdown_html_exports_include_citations(self):
        draft = reporting.build_draft_corrected_answer(self.original, self.results)
        self.assertIn("[1]", draft)
        self.assertIn("[2]", draft)
        self.assertIn("Sources:", draft)
        self.assertIn("[unverified]", draft)

        payload = reporting.build_report_payload(
            mode="Audit",
            original_answer=self.original,
            verification_results=self.results,
            corrected_answer=draft,
        )
        self.assertEqual(len(payload["corrected_citations"]), 2)
        self.assertEqual(len(payload["unverified_claims_marked"]), 1)

        # Markdown export includes Cited Sources links and Unverified Claims section
        md = reporting.format_report_markdown(payload)
        self.assertIn("## Corrected Answer", md)
        self.assertIn("### Cited Sources", md)
        self.assertIn("[1] [Python Downloads Official](https://www.python.org/downloads/)", md)
        self.assertIn("[2] [OpenAI API Pricing](https://openai.com/api/pricing/)", md)
        self.assertIn("### Unverified Claims (Left Untouched)", md)
        self.assertIn("Acme Corp plans to launch Widget X on Mars next month.", md)

        # HTML export links inline [1], [2] to #cited-source-1, #cited-source-2 and renders Sources list
        html_out = reporting.format_report_html(payload)
        self.assertIn('href="#cited-source-1"', html_out)
        self.assertIn('href="#cited-source-2"', html_out)
        self.assertIn('id="cited-source-1"', html_out)
        self.assertIn('id="cited-source-2"', html_out)
        self.assertIn("https://www.python.org/downloads/", html_out)
        self.assertIn("https://openai.com/api/pricing/", html_out)
        self.assertIn("Unverified claims (left untouched)", html_out)

    def test_cited_html_renderer_escapes_xss_and_blocks_javascript_urls(self):
        html_out = reporting.render_cited_corrected_answer_html(
            "Updated <script>alert(1)</script> [1]. Unchecked [unverified].",
            citations=[{"number": 1, "title": "<img onerror=alert(1)>", "url": "javascript:alert(1)", "source_type": "Primary"}],
            unverified_claims=[{"index": 2, "claim": "<svg onload=alert(1)>", "status": "unverifiable"}],
        )
        self.assertNotIn("<script>", html_out)
        self.assertNotIn("javascript:alert", html_out)
        self.assertNotIn("<img ", html_out)
        self.assertNotIn("<svg ", html_out)


class TestFreshnessGuardSDKAndAPI(unittest.TestCase):
    """Feature 1: stale_guard.guard() SDK, FastAPI POST /guard, and chatbot wrapper."""

    def setUp(self):
        api._reset_guard_rate_limiter()
        self.client = TestClient(api.app)

    def tearDown(self):
        api._reset_guard_rate_limiter()

    def test_verdict_mapping_all_cases(self):
        self.assertEqual(compute_guard_verdict([], None), "no_claims")
        self.assertEqual(compute_guard_verdict([{"status": "fresh", "score": 92}], 92), "fresh")
        self.assertEqual(compute_guard_verdict([{"status": "outdated", "score": 15}], 15), "stale")
        self.assertEqual(compute_guard_verdict([{"status": "incorrect", "score": 5}], 5), "incorrect")
        self.assertEqual(compute_guard_verdict([{"status": "unverifiable", "score": None}], None), "needs_review")
        self.assertEqual(compute_guard_verdict([{"status": "verification_failed", "score": None}], None), "verification_failed")

    def test_guard_sdk_returns_verdict_claims_cited_correction_sources_usage(self):
        mock_extracted = [
            {"claim": "Python 3.11 is the latest release.", "category": "version", "search_query": "Python latest release"}
        ]
        mock_verified = [
            {
                "claim": "Python 3.11 is the latest release.",
                "category": "version",
                "risk_level": "moderate",
                "status": "outdated",
                "score": 12,
                "confidence": "high",
                "current_info": "Python 3.13 is the latest stable release.",
                "reason": "Superseded by Python 3.13.",
                "tavily_credits_used": 1.0,
                "sources": [
                    {
                        "title": "Python.org Downloads",
                        "url": "https://www.python.org/downloads/",
                        "domain": "python.org",
                        "source_type": "Primary",
                        "is_primary": True,
                    }
                ],
                "judge_telemetry": [
                    {
                        "model_used": "gemini-3.1-flash-lite",
                        "prompt_tokens": 110,
                        "completion_tokens": 45,
                        "total_tokens": 155,
                        "usage_returned": True,
                    }
                ],
            }
        ]
        mock_rewritten = (
            "Python 3.13 is the latest stable release [1].\n\n"
            "Sources:\n[1] Python.org Downloads — https://www.python.org/downloads/"
        )

        with patch("stale_guard.guard.extract_claims", return_value=(mock_extracted, "gemini-3.1-flash-lite")), \
             patch("stale_guard.guard.get_extract_metrics", return_value={"model_used": "gemini-3.1-flash-lite", "prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80, "usage_returned": True}), \
             patch("stale_guard.guard.verify_claims", return_value=mock_verified), \
             patch("stale_guard.guard.generate_corrected_answer", return_value=(mock_rewritten, "gemini-3.1-flash-lite")), \
             patch("stale_guard.guard.get_rewrite_metrics", return_value={"model_used": "gemini-3.1-flash-lite", "prompt_tokens": 70, "completion_tokens": 30, "total_tokens": 100, "usage_returned": True}):
            res = guard(
                question="What is the latest Python release?",
                answer="Python 3.11 is the latest release.",
                context="From docs snapshot 2023",
                max_claims=6,
            )

        self.assertEqual(res["verdict"], "stale")
        self.assertEqual(res["overall_score"], 12)
        self.assertEqual(len(res["claims"]), 1)
        self.assertIn("[1]", res["corrected_answer"])
        self.assertIn("https://www.python.org/downloads/", res["corrected_answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertTrue(res["sources"][0]["cited"])
        self.assertEqual(res["usage"]["llm_calls"], 3)
        self.assertEqual(res["usage"]["total_tokens"], 335)
        self.assertEqual(res["usage"]["tavily_credits_used"], 1.0)
        self.assertGreaterEqual(res["usage"]["latency_ms"], 0.0)
        self.assertTrue(res["context_provided"])

    def test_guard_sdk_enforces_size_limits(self):
        with self.assertRaises(ValueError):
            guard("", "Some answer")
        with self.assertRaises(ValueError):
            guard("Q" * (MAX_GUARD_QUESTION_CHARS + 1), "Some answer")
        with self.assertRaises(ValueError):
            guard("Question", "A" * (MAX_GUARD_ANSWER_CHARS + 1))
        with self.assertRaises(ValueError):
            guard("Question", "Answer", context="C" * (MAX_GUARD_CONTEXT_CHARS + 1))
        with self.assertRaises(ValueError):
            guard("Question", "Answer", max_claims=99)

    def test_chatbot_example_wrapper(self):
        from examples.chatbot_guard_example import safe_chatbot_reply

        fake_stale = {
            "verdict": "stale",
            "corrected_answer": "Python 3.13 is the latest stable release [1].",
        }
        with patch("examples.chatbot_guard_example.guard", return_value=fake_stale):
            reply = safe_chatbot_reply("Latest Python?", "Python 3.11 is latest.")
        self.assertEqual(reply, fake_stale["corrected_answer"])

        fake_fresh = {"verdict": "fresh", "corrected_answer": "Water boils at 100C."}
        with patch("examples.chatbot_guard_example.guard", return_value=fake_fresh):
            reply_ok = safe_chatbot_reply("Boiling point?", "Water boils at 100C.")
        self.assertEqual(reply_ok, "Water boils at 100C.")

    def test_post_guard_endpoint_auth_rate_limit_size_limits_and_openapi(self):
        fake_guard_res = {
            "verdict": "fresh",
            "overall_score": 95,
            "question": "Who is CEO of Microsoft?",
            "answer": "Satya Nadella is CEO of Microsoft.",
            "context_provided": False,
            "claims": [{"claim": "Satya Nadella is CEO of Microsoft.", "status": "fresh", "score": 95}],
            "corrected_answer": "Satya Nadella is CEO of Microsoft.",
            "sources": [
                {
                    "index": 1,
                    "title": "Microsoft Leadership",
                    "url": "https://news.microsoft.com/exec/satya-nadella/",
                    "domain": "microsoft.com",
                    "source_type": "Primary",
                    "published_date": None,
                    "cited": False,
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
                "llm_calls": 2,
                "tavily_credits_used": 1.0,
                "latency_ms": 215.4,
                "routing_mode": "direct_gemini",
                "models_used": ["gemini-3.1-flash-lite"],
            },
            "generated_at": "2026-11-18T08:00:00+00:00",
        }

        # 1. OpenAPI docs include /guard
        openapi = self.client.get("/openapi.json").json()
        self.assertIn("/guard", openapi["paths"])
        self.assertIn("post", openapi["paths"]["/guard"])

        # 2. Size limits rejected with HTTP 422
        r_big = self.client.post(
            "/guard",
            json={"question": "Q", "answer": "X" * (MAX_GUARD_ANSWER_CHARS + 5)},
        )
        self.assertEqual(r_big.status_code, 422)

        r_claims = self.client.post(
            "/guard",
            json={"question": "Q", "answer": "Valid answer", "max_claims": 99},
        )
        self.assertEqual(r_claims.status_code, 422)

        # 3. Optional API key from env (401 on missing/wrong, 200 on valid X-API-Key and Bearer)
        with patch("api.run_freshness_guard", return_value=fake_guard_res), \
             patch.dict(os.environ, {"GUARD_API_KEY": "test-secret-guard-key", "GUARD_RATE_LIMIT_REQUESTS": "2", "GUARD_RATE_LIMIT_WINDOW_SECONDS": "60"}, clear=False):
            r_no_key = self.client.post("/guard", json={"question": "Q", "answer": "A"})
            self.assertEqual(r_no_key.status_code, 401)

            r_bad_key = self.client.post("/guard", json={"question": "Q", "answer": "A"}, headers={"X-API-Key": "wrong"})
            self.assertEqual(r_bad_key.status_code, 401)

            r_ok1 = self.client.post(
                "/guard",
                json={"question": "Q", "answer": "A"},
                headers={"X-API-Key": "test-secret-guard-key", "X-Forwarded-For": "203.0.113.10"},
            )
            self.assertEqual(r_ok1.status_code, 200)
            self.assertEqual(r_ok1.json()["verdict"], "fresh")

            r_ok2 = self.client.post(
                "/guard",
                json={"question": "Q", "answer": "A"},
                headers={"Authorization": "Bearer test-secret-guard-key", "X-Forwarded-For": "203.0.113.10"},
            )
            self.assertEqual(r_ok2.status_code, 200)

            # 4. 3rd request from same IP hits per-IP rate limit (HTTP 429 + Retry-After)
            r_429 = self.client.post(
                "/guard",
                json={"question": "Q", "answer": "A"},
                headers={"X-API-Key": "test-secret-guard-key", "X-Forwarded-For": "203.0.113.10"},
            )
            self.assertEqual(r_429.status_code, 429)
            self.assertIn("retry-after", {k.lower() for k in r_429.headers.keys()})

            # Different IP is allowed cleanly
            r_other_ip = self.client.post(
                "/guard",
                json={"question": "Q", "answer": "A"},
                headers={"X-API-Key": "test-secret-guard-key", "X-Forwarded-For": "203.0.113.99"},
            )
            self.assertEqual(r_other_ip.status_code, 200)


class TestDeploymentReadinessFiles(unittest.TestCase):
    """Feature 3: Verify Dockerfile, docker-compose.yml, .dockerignore, Makefile, CI, README."""

    def test_dockerfile_python312_nonroot_healthcheck(self):
        df = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("FROM python:3.12-slim", df)
        self.assertIn("USER appuser", df)
        self.assertIn("HEALTHCHECK", df)
        self.assertIn("/_stcore/health", df)

    def test_docker_compose_app_and_optional_bifrost_profile(self):
        dc = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("app:", dc)
        self.assertIn("bifrost:", dc)
        self.assertIn('profiles: ["bifrost"]', dc)
        self.assertIn("maximhq/bifrost", dc)

    def test_dockerignore_excludes_secrets_and_caches(self):
        di = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".env", di)
        self.assertIn(".git", di)
        self.assertIn(".tavily_cache.db", di)

    def test_makefile_has_run_and_test_targets(self):
        mk = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn("run:", mk)
        self.assertIn("test:", mk)

    def test_ci_workflow_is_offline_only_and_needs_no_secrets(self):
        ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertNotIn("${{ secrets.", ci)
        self.assertIn("unittest discover", ci)
        self.assertIn("stale_guard", ci)

    def test_readme_sections_and_honest_untested_disclosure(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("## Add a freshness guard to your RAG or chatbot", readme)
        self.assertIn("POST /guard", readme)
        self.assertIn("Not tested locally:", readme)


if __name__ == "__main__":
    unittest.main()
