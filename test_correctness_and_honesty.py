"""Offline mocked tests for correctness, honesty, and failure handling.

Covers:
1. Failure states (Tavily/LLM connection errors with 3-try backoff, 429, 503, invalid JSON):
   - status == "verification_failed", score is None, excluded from overall score and status counts
   - overall header returns "Could not verify N of M claims" instead of "Mixed"
   - score ring renders "—" when all claims fail
   - stepper marks failed step red and later steps stay pending (never green)
   - Quota chip classification ("Offline" vs "Limited")
   - ledger and session gateway log exclude failed / 0-usage calls
2. No invented numbers:
   - measured step timings rendered, "(not run)" for steps that did not run
   - static guard test failing if hard-coded UI metrics or fabricated latencies reappear
3. Model Showdown:
   - empty state when no real recorded evidence/showdown_results.json exists
   - zero literal "**" markdown artifacts in winner summary HTML/plain text
   - scripts/generate_showdown.py saves timestamp, routing, models, raw responses (mocked)
4. History & Examples:
   - History starts empty (no fake seeded timestamps)
   - Failed/Partial audits recorded with status and score=None
   - Built-in examples loaded from demo_cache.json with real input_text (never empty quotes)
5. Route-aware labels, copy audit, and privacy settings:
   - "Bifrost gateway calls" vs "LLM calls (direct Gemini)" driven by real routing
   - .streamlit/config.toml sets gatherUsageStats = false
   - No unbacked absolute privacy strings or retired model copy
6. Categories & office-holder source filtering:
   - leadership / office holder, capital / geography, records / latest
   - office-holder verification keeps official gov/company pages & reputable news only
7. Source deduplication & single label:
   - Registered domain deduplicated with +N page(s)
   - Strictly ONE uppercase type label (PRIMARY, NEWS, BLOG, or FORUM)
   - "N of M independent sources agree" matches unique domain chips
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app
import extractor
import ledger
import llm
import showdown_data
import tab_components
import verifier


class TestFailureStatesAndRetry(unittest.TestCase):
    """Requirement 1: Failure states, retries, score exclusion, quota chip, ledger."""

    @patch("verifier.time.sleep", return_value=None)
    @patch("verifier.get_tavily_client")
    @patch("verifier._cache_get", return_value=None)
    def test_tavily_connection_error_retries_3_times_and_fails_cleanly(
        self, _mock_cache, mock_get_client, mock_sleep
    ):
        # Verify single search retries 3 times with 2 exponential backoff sleeps
        direct_client = MagicMock()
        direct_client.search.side_effect = ConnectionError("Connection refused to api.tavily.com")
        with self.assertRaises(ConnectionError):
            verifier._call_tavily_search_with_retry(direct_client, {"query": "Python latest"})
        self.assertEqual(direct_client.search.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

        # Verify end-to-end verify_claim marks claim verification_failed with score=None
        client = MagicMock()
        client.search.side_effect = ConnectionError("Connection refused to api.tavily.com")
        mock_get_client.return_value = client

        claim = {
            "claim": "Python 3.13 is the latest release.",
            "category": "version",
            "search_query": "Python latest stable release version",
            "detect_latency_ms": 42.5,
        }
        res = verifier.verify_claim(claim)

        self.assertEqual(res["status"], "verification_failed")
        self.assertIsNone(res["score"])
        self.assertEqual(res["failed_step"], "verify")
        self.assertIsNone(res["judge_latency_ms"])
        self.assertIsNone(res["score_latency_ms"])
        self.assertEqual(res["detect_latency_ms"], 42.5)
        self.assertIsNotNone(res["search_latency_ms"])
        self.assertEqual(res["judge_telemetry"], [])

    @patch("verifier.time.sleep", return_value=None)
    @patch("verifier.get_tavily_client")
    @patch("verifier._cache_get", return_value=None)
    def test_tavily_503_returns_verification_failed_with_none_score(
        self, _mock_cache, mock_get_client, _mock_sleep
    ):
        client = MagicMock()
        client.search.side_effect = RuntimeError("HTTP 503 Service Unavailable")
        mock_get_client.return_value = client

        res = verifier.verify_claim(
            {"claim": "Claude 4 costs $1 per million tokens.", "category": "pricing", "search_query": "Claude pricing"}
        )
        self.assertEqual(res["status"], "verification_failed")
        self.assertIsNone(res["score"])
        self.assertEqual(res["failed_step"], "verify")

    @patch("verifier._cache_set")
    @patch("verifier._cache_get", return_value=None)
    @patch("verifier._invoke_call_llm", side_effect=RuntimeError("HTTP 429 rate limit exceeded"))
    @patch("verifier.get_tavily_client")
    def test_llm_429_returns_verification_failed_and_no_fabricated_metrics(
        self, mock_get_client, _mock_llm, _mock_cache, _mock_store
    ):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {
                    "title": "Official Pricing",
                    "url": "https://openai.com/api/pricing/",
                    "content": "GPT-4o pricing is $2.50 per 1M input tokens.",
                    "score": 0.9,
                }
            ]
        }
        mock_get_client.return_value = client

        res = verifier.verify_claim(
            {"claim": "GPT-4o costs $10 per 1M tokens.", "category": "pricing", "search_query": "GPT-4o pricing"}
        )
        self.assertEqual(res["status"], "verification_failed")
        self.assertIsNone(res["score"])
        self.assertEqual(res["failed_step"], "verify")
        self.assertEqual(res["judge_telemetry"], [])
        self.assertIsNone(res["score_latency_ms"])

    @patch("verifier._cache_set")
    @patch("verifier._cache_get", return_value=None)
    @patch("verifier._invoke_call_llm", return_value=("not valid json {{{", {"model_used": "gemini-3.1-flash-lite", "latency_ms": 110.0, "total_tokens": 50, "usage_returned": True}))
    @patch("verifier.get_tavily_client")
    def test_llm_invalid_json_returns_verification_failed(
        self, mock_get_client, _mock_llm, _mock_cache, _mock_store
    ):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {
                    "title": "Official Docs",
                    "url": "https://docs.python.org/3/",
                    "content": "Python 3.13 documentation.",
                    "score": 0.92,
                }
            ]
        }
        mock_get_client.return_value = client

        res = verifier.verify_claim(
            {"claim": "Python 3.11 is the newest.", "category": "version", "search_query": "Python latest version"}
        )
        self.assertEqual(res["status"], "verification_failed")
        self.assertIsNone(res["score"])
        self.assertEqual(res["failed_step"], "verify")

    @patch("llm.time.sleep", return_value=None)
    @patch("llm._invoke_chat_completion")
    def test_llm_connection_error_retries_3_times_then_tries_fallback(
        self, mock_invoke, mock_sleep
    ):
        mock_msg = MagicMock()
        mock_msg.content = '{"status":"fresh"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        fallback_resp = MagicMock()
        fallback_resp.choices = [mock_choice]
        fallback_resp.usage = mock_usage
        fallback_resp.model = "google/gemini-3.5-flash"

        mock_invoke.side_effect = [
            ConnectionError("unreachable 1"),
            ConnectionError("unreachable 2"),
            ConnectionError("unreachable 3"),
            (fallback_resp, {}),
        ]
        mock_client = MagicMock()
        model_chain = ["google/gemini-3.1-flash-lite", "google/gemini-3.5-flash"]
        with patch("llm.get_llm_client", return_value=(mock_client, model_chain, "http://localhost:8080/v1")), \
             patch("llm.is_use_bifrost", return_value=True):
            text, meta = llm.call_llm_with_meta(
                [{"role": "user", "content": "hi"}],
                task="judge",
            )
        self.assertEqual(text, '{"status":"fresh"}')
        self.assertTrue(meta["fallback_used"])
        self.assertTrue(meta["usage_returned"])
        # 3 primary attempts + 1 fallback attempt
        self.assertEqual(mock_invoke.call_count, 4)
        # 2 sleeps between the 3 primary attempts (1s, 2s)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_score_exclusion_and_overall_header(self):
        # All failed -> score None, tier "Could not verify N of M claims"
        all_failed = [
            {"claim": "A", "status": "verification_failed", "score": None, "risk_level": "critical"},
            {"claim": "B", "status": "verification_failed", "score": None, "risk_level": "moderate"},
        ]
        self.assertIsNone(verifier.compute_risk_weighted_overall_score(all_failed))
        tier, color, desc = app.get_overall_tier(None, all_failed)
        self.assertEqual(tier, "Could not verify 2 of 2 claims")
        self.assertNotIn("Mixed", tier)
        ring_html = app.render_score_ring_html(None, [], results=all_failed)
        self.assertIn("—", ring_html)
        self.assertIn("Could not verify 2 of 2 claims", ring_html)

        # Partial failure -> score uses ONLY verified claim, header says "Could not verify 1 of 2 claims"
        partial = [
            {"claim": "A", "status": "fresh", "score": 92, "risk_level": "moderate"},
            {"claim": "B", "status": "verification_failed", "score": None, "risk_level": "critical"},
        ]
        self.assertEqual(verifier.compute_risk_weighted_overall_score(partial), 92)
        p_tier, _, _ = app.get_overall_tier(92, partial)
        self.assertEqual(p_tier, "Could not verify 1 of 2 claims")
        self.assertNotIn("Mixed", p_tier)

    def test_quota_chip_error_classification(self):
        self.assertEqual(app.classify_error_quota_status("HTTP 429 rate limit exceeded"), "Limited")
        self.assertEqual(app.classify_error_quota_status("Quota exhausted for today"), "Limited")
        self.assertEqual(app.classify_error_quota_status("Connection refused to host"), "Offline")
        self.assertEqual(app.classify_error_quota_status("HTTP 503 Service Unavailable"), "Offline")

    def test_ledger_and_gateway_log_exclude_failed_and_zero_usage_calls(self):
        items = [
            {
                "claim": "C1",
                "status": "fresh",
                "score": 90,
                "judge_telemetry": [
                    {"model": "gemini-3.1-flash-lite", "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140, "latency_ms": 200, "usage_returned": True},
                    {"model": "gemini-3.5-flash", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": "timeout", "usage_returned": False},
                ],
                "tavily_metrics": {"queries_sent": 1, "credits_used": 1.0, "cache_hit": False},
            },
            {
                "claim": "C2",
                "status": "verification_failed",
                "score": None,
                "judge_telemetry": [],
                "gateway_metrics": {"model": "gemini-3.1-flash-lite", "total_tokens": 0, "error": "503", "usage_returned": False},
            },
        ]
        summary = ledger.compute_audit_ledger(items, extract_meta={"total_tokens": 0, "usage_returned": False})
        self.assertEqual(summary["bifrost"]["calls"], 1)
        self.assertEqual(summary["bifrost"]["total_tokens"], 140)

        mock_ss: dict = {}
        with patch("streamlit.session_state", mock_ss):
            app.append_gateway_request({"operation": "Verify", "model_used": "m1", "total_tokens": 0, "error": "ConnectionError", "usage_returned": False})
            app.append_gateway_request({"operation": "Verify", "model_used": "m1", "total_tokens": 0, "usage_returned": False})
            app.append_gateway_request({"operation": "Verify", "model_used": "m1", "total_tokens": 55, "prompt_tokens": 30, "completion_tokens": 25, "usage_returned": True})
            self.assertEqual(len(mock_ss.get("gateway_requests", [])), 1)
            self.assertEqual(mock_ss["gateway_requests"][0]["total_tokens"], 55)


class TestStepperErrorStates(unittest.TestCase):
    """Requirement 1: Stepper marks failed step red and later steps stay pending."""

    def test_stepper_marks_detect_error_and_later_steps_pending(self):
        mock_ss = {"pasted_error": "Connection error during extraction"}
        with patch("streamlit.session_state", mock_ss):
            html = app.render_stepper_html(None, has_results=False, has_input=True, results=[])
        # Step 1 done, Step 2 error, Steps 3-5 pending
        self.assertRegex(html, r'pipe-step pipe-done[^>]*>.*?Input')
        self.assertRegex(html, r'pipe-step pipe-error[^>]*>.*?Detect')
        self.assertRegex(html, r'pipe-step pipe-pending[^>]*>.*?Verify')
        self.assertRegex(html, r'pipe-step pipe-pending[^>]*>.*?Score')
        self.assertRegex(html, r'pipe-step pipe-pending[^>]*>.*?Report')

    def test_stepper_marks_verify_error_and_score_report_pending(self):
        mock_ss = {}
        failed_results = [{"claim": "X", "status": "verification_failed", "score": None}]
        with patch("streamlit.session_state", mock_ss):
            html = app.render_stepper_html(None, has_results=True, has_input=True, results=failed_results)
        self.assertRegex(html, r'pipe-step pipe-done[^>]*>.*?Input')
        self.assertRegex(html, r'pipe-step pipe-done[^>]*>.*?Detect')
        self.assertRegex(html, r'pipe-step pipe-error[^>]*>.*?Verify')
        self.assertRegex(html, r'pipe-step pipe-pending[^>]*>.*?Score')
        self.assertRegex(html, r'pipe-step pipe-pending[^>]*>.*?Report')


class TestNoInventedNumbersAndTimingGuard(unittest.TestCase):
    """Requirement 2: Measured timings, '(not run)' for skipped steps, no hard-coded UI metrics."""

    def test_step_timeline_uses_measured_timings_and_not_run(self):
        item = {
            "detect_latency_ms": 38.4,
            "search_latency_ms": 195.2,
            "judge_latency_ms": None,
            "score_latency_ms": None,
            "status": "verification_failed",
        }
        html = app.format_step_timeline_html(item)
        self.assertIn('Detected <span class="timing">(38ms)</span>', html)
        self.assertIn('Searched <span class="timing">(195ms)</span>', html)
        self.assertIn('Judged <span class="timing">(not run)</span>', html)
        self.assertIn('Scored <span class="timing">(not run)</span>', html)
        # Must never show old invented defaults
        self.assertNotIn("(45ms)", html)
        self.assertNotIn("(210ms)", html)
        self.assertNotIn("(340ms)", html)
        self.assertNotIn("(12ms)", html)

    def test_no_hardcoded_ui_metrics_in_source_files(self):
        """Guard test: fails if fabricated UI latencies or fake telemetry percentages reappear."""
        repo_root = Path(__file__).resolve().parent
        forbidden_patterns = [
            (r"\bor\s+210\.0\b", "hard-coded 210ms search latency fallback"),
            (r"\bor\s+340\.0\b", "hard-coded 340ms judge latency fallback"),
            (r"\bor\s+45\.0\b", "hard-coded 45ms detect latency fallback"),
            (r"\bor\s+12\.0\b", "hard-coded 12ms score latency fallback"),
            (r"280ms verified", "hard-coded 280ms verified preview badge"),
            (r'"latency_ms":\s*285\b', "fabricated offline API playground latency_ms"),
            (r'"timestamp_ms":\s*142\.0\b', "fabricated Why Bifrost demo timestamp_ms"),
            (r'"Telemetry Health",\s*"100%"', "hard-coded 100% Telemetry Health metric"),
            (r"Legacy LLM \(Pre-2024", "fabricated Legacy LLM showdown entry"),
            (r"SAVED_SHOWDOWNS\s*=\s*\{", "hard-coded fabricated SAVED_SHOWDOWNS dict"),
        ]
        files_to_scan = [
            repo_root / "app.py",
            repo_root / "tab_components.py",
            repo_root / "showdown_data.py",
            repo_root / "lab" / "ui.py",
        ]
        violations = []
        for path in files_to_scan:
            text = path.read_text(encoding="utf-8")
            for pattern, label in forbidden_patterns:
                if re.search(pattern, text):
                    violations.append(f"{path.name}: found {label} ({pattern})")
        self.assertEqual(violations, [], f"Hard-coded UI metrics detected: {violations}")


class TestShowdownHonestyAndScript(unittest.TestCase):
    """Requirement 3: Empty state without recorded JSON, zero '**', generate_showdown.py."""

    def test_empty_state_when_no_recorded_file_exists(self):
        with patch("showdown_data.SHOWDOWN_RESULTS_FILE") as mock_file:
            mock_file.is_file.return_value = False
            self.assertEqual(showdown_data.get_saved_showdowns(), {})
            self.assertIsNone(showdown_data.get_saved_showdown("anything"))

    def test_evaluate_showdown_winner_has_no_raw_markdown_asterisks(self):
        models = {
            "gemini-3.1-flash-lite": {"display_name": "gemini-3.1-flash-lite", "score": 88, "status_counts": {"outdated": 0, "incorrect": 0}, "latency_ms": 310},
            "gemini-3.5-flash": {"display_name": "gemini-3.5-flash", "score": 70, "status_counts": {"outdated": 1, "incorrect": 0}, "latency_ms": 420},
        }
        res = showdown_data.evaluate_showdown_winner(models)
        self.assertNotIn("**", res["reason"])
        self.assertNotIn("**", res["reason_html"])
        self.assertIn("<strong>", res["reason_html"])

        tie_models = {
            "m1": {"display_name": "m1", "score": 90, "status_counts": {"outdated": 0, "incorrect": 0}, "latency_ms": 250},
            "m2": {"display_name": "m2", "score": 90, "status_counts": {"outdated": 0, "incorrect": 0}, "latency_ms": 500},
        }
        tie_res = showdown_data.evaluate_showdown_winner(tie_models)
        self.assertNotIn("**", tie_res["reason"])
        self.assertNotIn("**", tie_res["reason_html"])

    def test_generate_showdown_script_writes_recorded_json_mocked(self):
        from scripts.generate_showdown import generate_showdown_file

        fake_live_result = {
            "id": "live_custom",
            "title": "Test Q",
            "question": "Who is CEO of OpenAI?",
            "category": "leadership",
            "source_origin": "live",
            "recorded_at": "2026-11-18T08:00:00+00:00",
            "recorded_date": "Nov 18, 2026",
            "routing_mode": "direct_gemini",
            "models": {
                "gemini-3.1-flash-lite": {
                    "display_name": "gemini-3.1-flash-lite",
                    "raw_response": "Sam Altman is CEO of OpenAI.",
                    "score": 95,
                    "status_counts": {"fresh": 1, "outdated": 0, "incorrect": 0, "uncertain": 0},
                    "latency_ms": 315,
                    "tokens": 80,
                },
                "gemini-3.5-flash": {
                    "display_name": "gemini-3.5-flash",
                    "raw_response": "Sam Altman leads OpenAI as CEO.",
                    "score": 95,
                    "status_counts": {"fresh": 1, "outdated": 0, "incorrect": 0, "uncertain": 0},
                    "latency_ms": 480,
                    "tokens": 92,
                },
            },
            "winner": "gemini-3.1-flash-lite",
            "winner_score": 95,
            "winner_reason": "Tie on freshness.",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "showdown_results.json"
            with patch("scripts.generate_showdown.run_live_showdown", return_value=fake_live_result), \
                 patch("scripts.generate_showdown.is_use_bifrost", return_value=False):
                payload = generate_showdown_file(
                    output_path=out_path,
                    models=["gemini-3.1-flash-lite", "gemini-3.5-flash"],
                    questions=[{"id": "q1", "title": "OpenAI CEO", "question": "Who is CEO of OpenAI?", "description": "Leadership check"}],
                )
            self.assertTrue(out_path.exists())
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["routing_mode"], "direct_gemini")
            self.assertEqual(saved["models"], ["gemini-3.1-flash-lite", "gemini-3.5-flash"])
            self.assertIn("recorded_at", saved)
            self.assertIn("q1", saved["showdowns"])
            self.assertEqual(saved["showdowns"]["q1"]["models"]["gemini-3.1-flash-lite"]["raw_response"], "Sam Altman is CEO of OpenAI.")
            self.assertEqual(payload["routing_mode"], "direct_gemini")


class TestHistoryAndExamples(unittest.TestCase):
    """Requirement 4: Real session history only, Failed/Partial recording, real Examples."""

    def test_history_starts_empty_and_records_failed_and_partial_without_score(self):
        mock_ss: dict = {}
        with patch("streamlit.session_state", mock_ss):
            tab_components.init_history_if_needed()
            self.assertEqual(mock_ss["audit_history"], [])

            # Record full failure
            tab_components.record_audit_to_history(
                "Some failed text",
                [{"claim": "C1", "status": "verification_failed", "score": None}],
                source_label="Live",
                score=None,
                error_reason="Connection refused",
            )
            self.assertEqual(len(mock_ss["audit_history"]), 1)
            entry = mock_ss["audit_history"][0]
            self.assertEqual(entry["audit_status"], "Failed")
            self.assertIsNone(entry["score"])
            self.assertEqual(entry["failed_count"], 1)
            self.assertNotIn("verification_failed", entry["status_counts"])

            # Record partial failure
            tab_components.record_audit_to_history(
                "Some partial text",
                [
                    {"claim": "C1", "status": "fresh", "score": 90},
                    {"claim": "C2", "status": "verification_failed", "score": None},
                ],
                source_label="Live",
                score=90,
            )
            self.assertEqual(len(mock_ss["audit_history"]), 2)
            partial_entry = mock_ss["audit_history"][0]
            self.assertEqual(partial_entry["audit_status"], "Partial")
            self.assertIsNone(partial_entry["score"])
            self.assertEqual(partial_entry["status_counts"]["fresh"], 1)
            self.assertEqual(partial_entry["failed_count"], 1)

    def test_builtin_examples_have_real_input_text_and_no_fake_timestamp(self):
        examples = tab_components.get_builtin_examples_from_cache()
        self.assertGreater(len(examples), 0)
        for ex in examples:
            self.assertTrue(ex["input_text"].strip(), f"Empty input_text for example {ex['id']}")
            self.assertNotEqual(ex.get("recorded_date"), "2026-11-13 14:30:00")


class TestRouteAwareLabelsCopyAndPrivacy(unittest.TestCase):
    """Requirements 5 & 6: Route-aware labels, dynamic config models, privacy settings."""

    def test_route_aware_labels_bifrost_vs_direct_gemini(self):
        info_b = tab_components.get_real_model_and_routing_info(use_bifrost=True)
        self.assertTrue(info_b["use_bifrost"])
        self.assertEqual(info_b["calls_label"], "Bifrost gateway calls")
        self.assertEqual(info_b["judge_step_label"], "4. Judge via Bifrost")
        self.assertGreaterEqual(len(info_b["configured_models"]), 1)

        info_d = tab_components.get_real_model_and_routing_info(use_bifrost=False)
        self.assertFalse(info_d["use_bifrost"])
        self.assertEqual(info_d["calls_label"], "LLM calls (direct Gemini)")
        self.assertEqual(info_d["judge_step_label"], "4. Judge (Direct Gemini)")

    def test_streamlit_config_disables_usage_stats(self):
        cfg_path = Path(__file__).resolve().parent / ".streamlit" / "config.toml"
        content = cfg_path.read_text(encoding="utf-8")
        self.assertIn("[browser]", content)
        self.assertRegex(content, r"gatherUsageStats\s*=\s*false")

    def test_copy_audit_no_unbacked_claims_or_absolutes(self):
        repo_root = Path(__file__).resolve().parent
        app_text = (repo_root / "app.py").read_text(encoding="utf-8")
        tab_text = (repo_root / "tab_components.py").read_text(encoding="utf-8")
        combined = app_text + "\n" + tab_text
        self.assertNotIn("Gemini 2.5 Flash, Flash-Lite and OpenAI models", combined)
        self.assertNotIn("Zero PII Storage", combined)
        self.assertNotIn("Zero Persistent Storage", combined)
        self.assertIn("Architecture & Evidence", app_text)
        self.assertNotIn("in under 3 seconds", tab_text)

    def test_validation_gate_blocks_overlimit_text_and_private_url_in_apptest(self):
        """Regression test: invalid/over-limit inputs must show error and NEVER trigger busy_action."""
        from streamlit.testing.v1 import AppTest

        with patch(
            "settings.validate_text_input",
            return_value=(False, "Input text exceeds the maximum limit of 6,000 characters."),
        ):
            at = AppTest.from_file("app.py", default_timeout=60).run()
            at.text_area(key="answer_input").set_value("Some over-limit test payload")
            next(b for b in at.button if b.key == "check_freshness_btn").click().run()
            self.assertIsNone(at.session_state["busy_action"])
            self.assertTrue(any("6,000" in e.value or "exceeds" in e.value.lower() for e in at.error))

        at_url = AppTest.from_file("app.py", default_timeout=60).run()
        at_url.session_state["active_input_mode"] = "Paste a URL"
        at_url.run()
        at_url.text_input(key="url_input").set_value("http://127.0.0.1:8080/admin")
        next(b for b in at_url.button if b.key == "audit_url_btn").click().run()
        self.assertIsNone(at_url.session_state["busy_action"])
        self.assertTrue(len(at_url.error) >= 1)

    def test_network_disconnect_friendly_message_both_routes(self):
        with patch("app.is_use_bifrost", return_value=True):
            msg_b = app.format_friendly_ui_error(ConnectionError("network unreachable"), action="verifying your answer")
            self.assertIn("network offline", msg_b.lower())
            self.assertNotIn("Traceback", msg_b)
        with patch("app.is_use_bifrost", return_value=False):
            msg_d = app.format_friendly_ui_error(ConnectionError("network unreachable"), action="verifying your answer")
            self.assertIn("network offline", msg_d.lower())
            self.assertNotIn("Traceback", msg_d)


class TestCategoriesAndOfficeHolderFiltering(unittest.TestCase):
    """Requirement 7: leadership, geography, records categories & office-holder source rules."""

    def test_category_normalization_and_display_labels(self):
        raw = [
            {"claim": "Narendra Modi is the Prime Minister of India.", "category": "other", "search_query": "Prime Minister of India"},
            {"claim": "Satya Nadella is the CEO of Microsoft.", "category": "company", "search_query": "Microsoft CEO"},
            {"claim": "Canberra is the capital of Australia.", "category": "geography", "search_query": "capital of Australia"},
            {"claim": "Usain Bolt holds the 100m world record.", "category": "records", "search_query": "100m world record"},
        ]
        norm = extractor._normalize_claims(raw)
        self.assertEqual(norm[0]["category"], "leadership")
        self.assertEqual(norm[1]["category"], "leadership")
        self.assertEqual(norm[2]["category"], "geography")
        self.assertEqual(norm[3]["category"], "records")

        self.assertEqual(extractor.format_category_label("leadership"), "leadership / office holder")
        self.assertEqual(extractor.format_category_label("geography"), "capital / geography")
        self.assertEqual(extractor.format_category_label("records"), "records / latest")

    def test_office_holder_source_filtering_keeps_only_official_and_reputable_news(self):
        results = [
            {"url": "https://www.pmindia.gov.in/en/", "title": "PM India Official", "content": "Prime Minister Narendra Modi"},
            {"url": "https://www.reuters.com/world/india/pm-modi/", "title": "Reuters", "content": "Prime Minister Narendra Modi"},
            {"url": "https://medium.com/@random/india-pm", "title": "Random Blog", "content": "Opinion piece"},
            {"url": "https://www.reddit.com/r/india/comments/123/", "title": "Reddit", "content": "Forum post"},
            {"url": "https://random-untrusted-site.xyz/politics", "title": "Random site", "content": "Unverified"},
        ]
        filtered = verifier.filter_office_holder_sources(
            results,
            claim="Narendra Modi is the Prime Minister of India.",
        )
        urls = [r["url"] for r in filtered]
        self.assertIn("https://www.pmindia.gov.in/en/", urls)
        self.assertIn("https://www.reuters.com/world/india/pm-modi/", urls)
        self.assertNotIn("https://medium.com/@random/india-pm", urls)
        self.assertNotIn("https://www.reddit.com/r/india/comments/123/", urls)
        self.assertNotIn("https://random-untrusted-site.xyz/politics", urls)


class TestSourceDedupeAndSingleLabel(unittest.TestCase):
    """Requirement 8: Deduplicate by registered domain (+N pages), ONE label, matching count."""

    def test_extract_top_sources_dedupes_registered_domain_and_counts_pages(self):
        search_results = [
            {"url": "https://openai.com/api/pricing/", "title": "OpenAI Pricing", "content": "Pricing", "score": 0.9},
            {"url": "https://platform.openai.com/docs/models", "title": "OpenAI Models", "content": "Models", "score": 0.85},
            {"url": "https://help.openai.com/en/articles/123", "title": "OpenAI Help", "content": "Help", "score": 0.7},
            {"url": "https://www.reuters.com/technology/openai-pricing/", "title": "Reuters", "content": "News", "score": 0.8},
        ]
        top = verifier._extract_top_sources(search_results, limit=5)
        self.assertEqual(len(top), 2)
        openai_src = next(s for s in top if s["domain"] == "openai.com")
        reuters_src = next(s for s in top if s["domain"] == "reuters.com")
        self.assertEqual(openai_src["page_count"], 3)
        self.assertEqual(openai_src["extra_pages"], 2)
        self.assertEqual(reuters_src["page_count"], 1)
        self.assertEqual(reuters_src["extra_pages"], 0)

    def test_source_chips_render_one_label_per_domain_and_extra_pages(self):
        sources = [
            {
                "title": "OpenAI Pricing",
                "url": "https://openai.com/api/pricing/",
                "domain": "openai.com",
                "source_type": "Primary",
                "is_primary": True,
                "page_count": 3,
                "extra_pages": 2,
            },
            {
                "title": "OpenAI Docs (dup domain)",
                "url": "https://platform.openai.com/docs/",
                "domain": "openai.com",
                "source_type": "Primary",
                "is_primary": True,
                "page_count": 1,
                "extra_pages": 0,
            },
            {
                "title": "Reuters Tech",
                "url": "https://www.reuters.com/tech/article",
                "domain": "reuters.com",
                "source_type": "News",
                "is_primary": False,
                "page_count": 1,
                "extra_pages": 0,
            },
        ]
        html = app._format_source_chips_html(sources)
        # Only 2 chips rendered (openai.com deduped + reuters.com)
        self.assertEqual(html.count("sad-source-chip"), 2)
        # Never renders duplicate PRIMARY badge
        self.assertNotIn("sad-primary-badge", html)
        # Exactly one uppercase type label per chip
        self.assertEqual(html.count("sad-src-type"), 2)
        self.assertIn(">PRIMARY<", html)
        self.assertIn(">NEWS<", html)
        # Extra pages badge present for openai.com (+3 pages total across dup merge: 3+1-1 = +3 pages)
        self.assertIn("sad-src-pages", html)

    @patch("verifier._cache_set")
    @patch("verifier._cache_get", return_value=None)
    @patch("verifier._invoke_call_llm")
    @patch("verifier.get_tavily_client")
    def test_agreement_summary_matches_unique_domain_count(
        self, mock_get_client, mock_llm, _mock_cache, _mock_store
    ):
        client = MagicMock()
        client.search.return_value = {
            "results": [
                {"url": "https://openai.com/api/pricing/", "title": "OpenAI Pricing", "content": "GPT-4o is $2.50/1M", "score": 0.95},
                {"url": "https://platform.openai.com/docs/pricing", "title": "OpenAI Docs", "content": "GPT-4o is $2.50/1M", "score": 0.90},
                {"url": "https://www.reuters.com/tech/openai/", "title": "Reuters", "content": "OpenAI cuts GPT-4o price to $2.50/1M", "score": 0.88},
            ]
        }
        mock_get_client.return_value = client
        judge_json = json.dumps({
            "evidence_stance": "supports",
            "supporting_quote": "GPT-4o is $2.50/1M",
            "status": "fresh",
            "confidence": "high",
            "was_ever_true": "yes",
            "current_info": "GPT-4o costs $2.50 per 1M input tokens.",
            "reason": "Confirmed by OpenAI and Reuters.",
            "agreeing_sources": 5,  # Intentionally inflated; verifier must clamp to unique domains (2)
            "conflicting_sources": False,
            "source_evaluations": [
                {"url": "https://openai.com/api/pricing/", "stance": "support", "quote": "GPT-4o is $2.50/1M", "source_type": "primary"},
                {"url": "https://www.reuters.com/tech/openai/", "stance": "support", "quote": "OpenAI cuts GPT-4o price to $2.50/1M", "source_type": "news"},
            ],
        })
        mock_llm.return_value = (
            judge_json,
            {"model_used": "gemini-3.1-flash-lite", "latency_ms": 180.0, "prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180, "usage_returned": True},
        )

        res = verifier.verify_claim(
            {"claim": "GPT-4o costs $2.50 per 1M input tokens.", "category": "pricing", "search_query": "GPT-4o API pricing"}
        )
        self.assertEqual(len(res["sources"]), 2)
        self.assertEqual(res["sources_total"], 2)
        self.assertEqual(res["sources_agreeing"], 2)
        self.assertEqual(res["agreement_summary"], "2 of 2 independent sources agree")


if __name__ == "__main__":
    unittest.main()
