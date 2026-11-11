"""
test_llm.py - Unit tests for Bifrost gateway integration in llm.py.

Tests native per-request fallbacks (extra_body={"fallbacks": ...}), virtual key
headers (x-bf-vk), unreachable-only single-retry safety net, simulated provider
outage failover, per-request telemetry extraction (model, latency, tokens,
fallback_used), health check endpoint probing, and dashboard URL derivation.
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai

import llm


def _make_status_error(status_code: int, message: str = "API error") -> openai.APIStatusError:
    req = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    resp = httpx.Response(status_code=status_code, request=req)
    return openai.APIStatusError(message, response=resp, body=None)


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    req = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    resp = httpx.Response(status_code=400, request=req)
    return openai.BadRequestError(message, response=resp, body=None)


class TestBifrostLLMClient(unittest.TestCase):
    def setUp(self):
        llm.set_simulate_outage(False)
        llm.pop_last_gateway_metrics()

    def test_config_validation_and_virtual_key_header(self):
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_MODEL_FALLBACKS": ""},
            clear=False,
        ):
            with patch("llm.load_dotenv"):
                with self.assertRaises(ValueError) as ctx:
                    llm.get_llm_config()
                self.assertIn("LLM_API_KEY", str(ctx.exception))

        with patch("llm.load_dotenv"):
            with patch.dict(
                os.environ,
                {
                    "LLM_API_KEY": "bifrost-local",
                    "LLM_BASE_URL": "http://localhost:8080/v1",
                    "LLM_MODEL": "openai/gpt-4o-mini",
                    "LLM_MODEL_FALLBACKS": "anthropic/claude-3-5-sonnet, gemini/gemini-2.5-flash, openai/gpt-4o-mini",
                    "LLM_VIRTUAL_KEY": "",
                },
                clear=False,
            ):
                client1, chain, base_url = llm.get_llm_client()
                self.assertEqual(base_url, "http://localhost:8080/v1")
                self.assertEqual(
                    chain,
                    ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet", "gemini/gemini-2.5-flash"],
                )
                self.assertNotIn("x-bf-vk", client1.default_headers)

            with patch.dict(
                os.environ,
                {"LLM_API_KEY": "bifrost-local", "LLM_VIRTUAL_KEY": "bf-vk-demo"},
                clear=False,
            ):
                client2, _, _ = llm.get_llm_client()
                self.assertEqual(client2.default_headers["x-bf-vk"], "bf-vk-demo")

    def test_native_fallbacks_single_request_with_real_openai_transport(self):
        captured_requests: list[tuple[dict, dict]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            captured_requests.append((dict(request.headers), payload))
            return httpx.Response(
                status_code=200,
                headers={
                    "content-type": "application/json",
                    "x-bifrost-routing-info-provider": "anthropic",
                    "x-bifrost-routing-info-model": "claude-3-5-sonnet",
                    "x-bifrost-routing-info-is-fallback": "true",
                    "x-bifrost-latency": "342.5",
                },
                json={
                    "id": "chatcmpl-test-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "claude-3-5-sonnet-20241022",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 45,
                        "total_tokens": 165,
                    },
                    "extra_fields": {
                        "provider": "anthropic",
                        "latency": 342.5,
                        "routing_info": {
                            "provider": "anthropic",
                            "model": "claude-3-5-sonnet",
                            "is_fallback": True,
                            "primary_provider": "openai",
                            "primary_model": "gpt-4o-mini",
                        },
                    },
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        real_client = openai.OpenAI(
            api_key="bifrost-local",
            base_url="http://localhost:8080/v1",
            default_headers={"x-bf-vk": "bf-vk-999"},
            http_client=http_client,
            max_retries=0,
        )
        chain = ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet", "gemini/gemini-2.5-flash"]

        with patch("llm.load_dotenv"), patch.dict(os.environ, {"LLM_VIRTUAL_KEY": "bf-vk-999"}, clear=False):
            with patch("llm.get_llm_client", return_value=(real_client, chain, "http://localhost:8080/v1")):
                content, meta = llm.call_llm_with_meta("sys", "usr")

        # Single gateway HTTP call with exact Bifrost fallbacks + virtual key header.
        self.assertEqual(len(captured_requests), 1)
        req_headers, req_body = captured_requests[0]
        self.assertEqual(req_headers.get("x-bf-vk"), "bf-vk-999")
        self.assertEqual(req_body["model"], "openai/gpt-4o-mini")
        self.assertEqual(
            req_body["fallbacks"],
            ["anthropic/claude-3-5-sonnet", "gemini/gemini-2.5-flash"],
        )

        # Telemetry & metadata verification.
        self.assertEqual(content, '{"ok": true}')
        self.assertEqual(meta["primary_model"], "openai/gpt-4o-mini")
        self.assertEqual(meta["model_used"], "anthropic/claude-3-5-sonnet")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["latency_ms"], 342.5)
        self.assertEqual(meta["prompt_tokens"], 120)
        self.assertEqual(meta["completion_tokens"], 45)
        self.assertEqual(meta["total_tokens"], 165)

        popped = llm.pop_last_gateway_metrics()
        self.assertIsNotNone(popped)
        self.assertEqual(popped["model_used"], "anthropic/claude-3-5-sonnet")

    def test_unreachable_gateway_retries_once_and_non_unreachable_does_not_loop(self):
        # 1. Gateway unreachable on first try, succeeds on second (single safety-net retry).
        ok_resp = MagicMock()
        ok_resp.choices = [MagicMock(message=MagicMock(content="Gateway recovered"))]
        ok_resp.model = "openai/gpt-4o-mini"
        ok_resp.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ok_resp.model_extra = {}

        conn_err = openai.APIConnectionError(
            request=httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [conn_err, ok_resp]

        with patch(
            "llm.get_llm_client",
            return_value=(mock_client, ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"], "http://localhost:8080/v1"),
        ), patch("llm.time.sleep") as mock_sleep:
            text, model_used = llm.call_llm("sys", "user")

        self.assertEqual(text, "Gateway recovered")
        self.assertEqual(model_used, "openai/gpt-4o-mini")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once_with(1)
        # First call passes native fallbacks.
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["extra_body"],
            {"fallbacks": ["anthropic/claude-3-5-sonnet"]},
        )

        # 2. Gateway returns HTTP 500 when native fallbacks already configured: no duplicate app retry loop.
        mock_client_500 = MagicMock()
        mock_client_500.chat.completions.create.side_effect = _make_status_error(500, "Upstream failure")
        with patch(
            "llm.get_llm_client",
            return_value=(mock_client_500, ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"], "http://localhost:8080/v1"),
        ), patch("llm.time.sleep") as mock_sleep_2:
            with self.assertRaises(RuntimeError):
                llm.call_llm("sys", "user")

        self.assertEqual(mock_client_500.chat.completions.create.call_count, 1)
        mock_sleep_2.assert_not_called()

    def test_simulate_outage_switches_to_fallback_and_marks_route(self):
        # Case A: Gateway rejects synthetic outage model -> fails over to fallback[0].
        fb_resp = MagicMock()
        fb_resp.choices = [MagicMock(message=MagicMock(content="Answered by Claude"))]
        fb_resp.model = "anthropic/claude-3-5-sonnet"
        fb_resp.usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        fb_resp.model_extra = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_status_error(404, "Simulated outage model not found"),
            fb_resp,
        ]

        llm.set_simulate_outage(True)
        with patch(
            "llm.get_llm_client",
            return_value=(mock_client, ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"], "http://localhost:8080/v1"),
        ):
            text, meta = llm.call_llm_with_meta("sys", "user")

        self.assertEqual(text, "Answered by Claude")
        self.assertTrue(meta["fallback_used"])
        self.assertTrue(meta["simulated_outage"])
        self.assertEqual(meta["primary_model"], "openai/gpt-4o-mini")
        self.assertEqual(meta["model_used"], "anthropic/claude-3-5-sonnet")
        first_call = mock_client.chat.completions.create.call_args_list[0].kwargs
        second_call = mock_client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(first_call["model"], "openai/simulated-outage-gpt-4o-mini")
        self.assertEqual(first_call["extra_body"], {"fallbacks": ["anthropic/claude-3-5-sonnet"]})
        self.assertEqual(second_call["model"], "anthropic/claude-3-5-sonnet")

        # Case B: Bifrost natively switches to fallback in a single HTTP 200 response.
        native_fb_resp = MagicMock()
        native_fb_resp.choices = [MagicMock(message=MagicMock(content="Native Bifrost outage failover"))]
        native_fb_resp.model = "claude-3-5-sonnet"
        native_fb_resp.usage = {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}
        native_fb_resp.model_extra = {
            "extra_fields": {
                "provider": "anthropic",
                "latency": 190.2,
                "routing_info": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "is_fallback": True,
                },
            }
        }
        mock_native = MagicMock()
        mock_native.chat.completions.create.return_value = native_fb_resp
        llm.set_simulate_outage(True)
        with patch(
            "llm.get_llm_client",
            return_value=(mock_native, ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"], "http://localhost:8080/v1"),
        ):
            text_b, meta_b = llm.call_llm_with_meta("sys", "user")

        self.assertEqual(mock_native.chat.completions.create.call_count, 1)
        self.assertEqual(text_b, "Native Bifrost outage failover")
        self.assertTrue(meta_b["fallback_used"])
        self.assertTrue(meta_b["simulated_outage"])
        self.assertEqual(meta_b["primary_model"], "openai/gpt-4o-mini")
        self.assertEqual(meta_b["model_used"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(meta_b["latency_ms"], 190.2)

    def test_bad_request_about_fallbacks_strips_extra_body_on_non_bifrost(self):
        ok_resp = MagicMock()
        ok_resp.choices = [MagicMock(message=MagicMock(content="Plain OpenAI worked"))]
        ok_resp.model = "gpt-4o-mini"
        ok_resp.usage = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        ok_resp.model_extra = {}

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_bad_request_error("Unrecognized request argument supplied: fallbacks"),
            ok_resp,
        ]
        with patch(
            "llm.get_llm_client",
            return_value=(mock_client, ["gpt-4o-mini", "gpt-4o"], None),
        ):
            text, model = llm.call_llm("sys", "user")

        self.assertEqual(text, "Plain OpenAI worked")
        self.assertEqual(model, "gpt-4o-mini")
        self.assertNotIn("extra_body", mock_client.chat.completions.create.call_args_list[1].kwargs)

    def test_gateway_health_and_dashboard_url(self):
        self.assertEqual(llm.get_gateway_dashboard_url("http://localhost:8080/v1"), "http://localhost:8080")
        self.assertEqual(llm.get_gateway_dashboard_url("https://api.openai.com/v1"), "http://localhost:8080")
        with patch("llm.load_dotenv"), patch.dict(os.environ, {"LLM_BASE_URL": ""}, clear=False):
            self.assertEqual(llm.get_gateway_dashboard_url(), "http://localhost:8080")

        ok_http = httpx.Response(
            200,
            json={"status": "ok", "version": "1.2.0", "providers": ["openai", "anthropic"]},
        )
        with patch("llm.httpx.get", return_value=ok_http):
            health_ok = llm.check_gateway_health("http://localhost:8080/v1")
        self.assertTrue(health_ok["healthy"])
        self.assertEqual(health_ok["status"], "ok")
        self.assertEqual(health_ok["version"], "1.2.0")
        self.assertEqual(health_ok["health_url"], "http://localhost:8080/health")

        with patch("llm.httpx.get", side_effect=httpx.ConnectError("Connection refused")):
            health_down = llm.check_gateway_health("http://localhost:8080/v1")
        self.assertFalse(health_down["healthy"])
        self.assertEqual(health_down["status"], "unreachable")

    def test_ask_question_via_bifrost_and_validation(self):
        with self.assertRaises(ValueError):
            llm.ask_question("   ")

        ok_resp = MagicMock()
        ok_resp.choices = [MagicMock(message=MagicMock(content="Apple sells the iPhone 15 Pro Max for $1,199."))]
        ok_resp.model = "anthropic/claude-3-5-sonnet"
        ok_resp.usage = {"prompt_tokens": 35, "completion_tokens": 18, "total_tokens": 53}
        ok_resp.model_extra = {
            "extra_fields": {
                "provider": "anthropic",
                "latency": 215.4,
                "routing_info": {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet",
                    "is_fallback": True,
                },
            }
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = ok_resp

        with patch(
            "llm.get_llm_client",
            return_value=(mock_client, ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet"], "http://localhost:8080/v1"),
        ):
            answer, meta = llm.ask_question("  What is the latest iPhone and price?  ")

        self.assertEqual(answer, "Apple sells the iPhone 15 Pro Max for $1,199.")
        self.assertEqual(meta["operation"], "Ask & Audit Q&A")
        self.assertEqual(meta["model_used"], "anthropic/claude-3-5-sonnet")
        self.assertEqual(meta["primary_model"], "openai/gpt-4o-mini")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["latency_ms"], 215.4)
        self.assertEqual(meta["prompt_tokens"], 35)
        self.assertEqual(meta["completion_tokens"], 18)
        self.assertEqual(meta["total_tokens"], 53)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["messages"][0]["content"], llm.QA_SYSTEM_PROMPT)
        self.assertEqual(call_kwargs["messages"][1]["content"], "What is the latest iPhone and price?")
        self.assertEqual(call_kwargs["extra_body"], {"fallbacks": ["anthropic/claude-3-5-sonnet"]})
        # ask_question consumes the single telemetry pop so callers use its returned meta cleanly.
        self.assertIsNone(llm.pop_last_gateway_metrics())

    def test_use_bifrost_false_uses_direct_gemini_endpoint_and_app_fallbacks(self):
        # 1. USE_BIFROST=false overrides localhost Bifrost URL -> official Gemini OpenAI endpoint,
        # omits x-bf-vk, normalizes gemini/ prefix, and drops non-Gemini provider routes.
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {
                "USE_BIFROST": "false",
                "LLM_BASE_URL": "http://localhost:8080/v1",
                "LLM_API_KEY": "",
                "GEMINI_API_KEY": "AIza-gemini-test-key",
                "LLM_MODEL": "openai/gpt-4o-mini",
                "LLM_MODEL_FALLBACKS": "anthropic/claude-3-5-sonnet, gemini/gemini-2.5-flash, gemini-2.5-flash-lite",
                "LLM_VIRTUAL_KEY": "bf-vk-should-not-be-sent",
            },
            clear=False,
        ):
            client, chain, base_url = llm.get_llm_client()
            self.assertEqual(base_url, llm.GEMINI_OPENAI_BASE_URL)
            self.assertEqual(client.api_key, "AIza-gemini-test-key")
            self.assertEqual(chain, ["gemini-2.5-flash", "gemini-2.5-flash-lite"])
            self.assertNotIn("x-bf-vk", client.default_headers)
            self.assertFalse(llm.is_bifrost_gateway(base_url))

            # check_gateway_health skips localhost probing when USE_BIFROST=false
            with patch("llm.httpx.get") as mock_get:
                health = llm.check_gateway_health()
            mock_get.assert_not_called()
            self.assertTrue(health["healthy"])
            self.assertEqual(health["mode"], "direct_gemini")

        # 2. Direct Gemini mode uses app-level fallback chain on 429/500/503/404 without extra_body
        fb_resp = MagicMock()
        fb_resp.choices = [MagicMock(message=MagicMock(content="Answered by gemini-2.5-flash-lite"))]
        fb_resp.model = "gemini-2.5-flash-lite"
        fb_resp.usage = {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37}
        fb_resp.model_extra = {}

        mock_gemini_client = MagicMock()
        mock_gemini_client.chat.completions.create.side_effect = [
            _make_status_error(429, "Quota exhausted for primary"),
            fb_resp,
        ]

        with patch("llm.load_dotenv"), patch.dict(os.environ, {"USE_BIFROST": "false"}, clear=False):
            with patch(
                "llm.get_llm_client",
                return_value=(
                    mock_gemini_client,
                    ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                    llm.GEMINI_OPENAI_BASE_URL,
                ),
            ):
                content, meta = llm.call_llm_with_meta("sys", "user")

        self.assertEqual(content, "Answered by gemini-2.5-flash-lite")
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["routing_mode"], "direct_gemini")
        self.assertEqual(meta["primary_model"], "gemini-2.5-flash")
        self.assertEqual(meta["model_used"], "gemini-2.5-flash-lite")
        first_kwargs = mock_gemini_client.chat.completions.create.call_args_list[0].kwargs
        second_kwargs = mock_gemini_client.chat.completions.create.call_args_list[1].kwargs
        self.assertNotIn("extra_body", first_kwargs)
        self.assertNotIn("extra_body", second_kwargs)
        self.assertEqual(first_kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(second_kwargs["model"], "gemini-2.5-flash-lite")

    def test_auth_errors_do_not_loop_fallbacks_and_friendly_errors_strip_tracebacks(self):
        # 1. Invalid key (401) stops immediately without cascading through fallbacks
        mock_auth_fail = MagicMock()
        mock_auth_fail.chat.completions.create.side_effect = _make_status_error(401, "API_KEY_INVALID")

        with patch("llm.load_dotenv"), patch.dict(os.environ, {"USE_BIFROST": "false"}, clear=False):
            with patch(
                "llm.get_llm_client",
                return_value=(mock_auth_fail, ["gemini-2.5-flash", "gemini-2.5-flash-lite"], llm.GEMINI_OPENAI_BASE_URL),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    llm.call_llm("sys", "user")

        self.assertEqual(mock_auth_fail.chat.completions.create.call_count, 1)
        self.assertIn("Invalid Gemini API key", str(ctx.exception))
        self.assertNotIn("Traceback", str(ctx.exception))

        # 2. Bifrost connection failure produces actionable Bifrost-down guidance
        conn_err = openai.APIConnectionError(
            request=httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
        )
        msg_bifrost_down = llm.format_llm_error(
            conn_err,
            use_bifrost=True,
            base_url="http://localhost:8080/v1",
            primary_model="openai/gpt-4o-mini",
        )
        self.assertIn("Bifrost gateway is down", msg_bifrost_down)
        self.assertIn("USE_BIFROST=false", msg_bifrost_down)

        # 3. Raw multi-line traceback text is scrubbed cleanly
        dirty_exc = RuntimeError('Traceback (most recent call last):\n  File "app.py", line 99\nBoom internal failure')
        cleaned_msg = llm.format_llm_error(dirty_exc, use_bifrost=False)
        self.assertNotIn("Traceback", cleaned_msg)
        self.assertNotIn('File "app.py"', cleaned_msg)

    def test_task_based_routing_cross_provider_fallbacks_and_dual_judge_resolution(self):
        with patch("llm.load_dotenv"), patch.dict(
            os.environ,
            {
                "USE_BIFROST": "true",
                "LLM_API_KEY": "bifrost-local",
                "LLM_BASE_URL": "http://localhost:8080/v1",
                "LLM_MODEL": "gemini/gemini-2.5-flash",
                "LLM_MODEL_EXTRACT": "gemini/gemini-2.5-flash-lite",
                "LLM_MODEL_JUDGE": "openai/gpt-4o-mini",
                "LLM_MODEL_JUDGE_2": "",
                "LLM_MODEL_JUDGE_SECOND": "",
                "LLM_MODEL_DUAL_JUDGE": "",
                "LLM_MODEL_FALLBACKS": "anthropic/claude-3-5-sonnet-20241022, groq/llama-3.3-70b-versatile",
                "LLM_VIRTUAL_KEY": "bf-vk-cross",
            },
            clear=False,
        ):
            _, _, extract_chain = llm.get_llm_config(task="extract")
            _, _, judge_chain = llm.get_llm_config(task="judge")
            _, _, default_chain = llm.get_llm_config()

            self.assertEqual(
                extract_chain,
                [
                    "gemini/gemini-2.5-flash-lite",
                    "anthropic/claude-3-5-sonnet-20241022",
                    "groq/llama-3.3-70b-versatile",
                ],
            )
            self.assertEqual(
                judge_chain,
                [
                    "openai/gpt-4o-mini",
                    "anthropic/claude-3-5-sonnet-20241022",
                    "groq/llama-3.3-70b-versatile",
                ],
            )
            self.assertEqual(default_chain[0], "gemini/gemini-2.5-flash")

            summary = llm.get_task_routing_summary()
            self.assertEqual(summary["extract_model"], "gemini/gemini-2.5-flash-lite")
            self.assertEqual(summary["judge_model"], "openai/gpt-4o-mini")
            self.assertTrue(summary["dual_judge_configured"])
            self.assertEqual(
                llm.get_dual_judge_models(),
                ("openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"),
            )

            # Real HTTP transport check verifying x-bf-vk + cross-provider fallbacks + task routing.
            seen_bodies: list[dict] = []

            def handler(request: httpx.Request) -> httpx.Response:
                seen_bodies.append(json.loads(request.content.decode("utf-8")))
                return httpx.Response(
                    status_code=200,
                    json={
                        "id": "chatcmpl-task",
                        "object": "chat.completion",
                        "created": 1700000001,
                        "model": seen_bodies[-1]["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "[]"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                    },
                )

            http_client = httpx.Client(transport=httpx.MockTransport(handler))
            _, _, _ = llm.get_llm_client(task="extract")

            def _mock_task_client(task=None, model_override=None):
                _, b_url, ch = llm.get_llm_config(task=task, model_override=model_override)
                c = openai.OpenAI(
                    api_key="bifrost-local",
                    base_url=b_url,
                    default_headers={"x-bf-vk": "bf-vk-cross"},
                    http_client=http_client,
                    max_retries=0,
                )
                return c, ch, b_url

            with patch("llm.get_llm_client", side_effect=_mock_task_client):
                _, meta_ext = llm.call_llm_with_meta("sys", "usr", task="extract")
                _, meta_jdg = llm.call_llm_with_meta("sys", "usr", task="judge")

            self.assertEqual(seen_bodies[0]["model"], "gemini/gemini-2.5-flash-lite")
            self.assertEqual(seen_bodies[1]["model"], "openai/gpt-4o-mini")
            self.assertEqual(
                seen_bodies[1]["fallbacks"],
                ["anthropic/claude-3-5-sonnet-20241022", "groq/llama-3.3-70b-versatile"],
            )
            self.assertEqual(meta_ext["task"], "extract")
            self.assertEqual(meta_jdg["task"], "judge")

        # Same bare model across two providers must NOT be collapsed as identical.
        self.assertFalse(llm._models_refer_to_same("openai/gpt-4o-mini", "azure/gpt-4o-mini"))
        self.assertTrue(llm._models_refer_to_same("anthropic/claude-3-5-sonnet", "claude-3-5-sonnet"))

    def test_streamlit_secrets_use_bifrost_false_cloud_config(self):
        import sys
        import types
        import verifier

        fake_st = types.SimpleNamespace(
            secrets={
                "USE_BIFROST": "false",
                "GEMINI_API_KEY": "AIza-cloud-secret-key",
                "LLM_MODEL": "gemini-2.5-flash",
                "LLM_MODEL_EXTRACT": "gemini/gemini-2.5-flash-lite",
                "LLM_MODEL_JUDGE": "gemini/gemini-2.5-flash",
                "LLM_MODEL_FALLBACKS": "openai/gpt-4o-mini, gemini-2.5-flash-lite",
                "TAVILY_API_KEY": "tvly-cloud-secret-key",
            }
        )
        cleared_env = {
            "USE_BIFROST": "",
            "LLM_API_KEY": "",
            "GEMINI_API_KEY": "",
            "LLM_BASE_URL": "",
            "LLM_MODEL": "",
            "LLM_MODEL_EXTRACT": "",
            "LLM_MODEL_JUDGE": "",
            "LLM_MODEL_FALLBACKS": "",
            "LLM_VIRTUAL_KEY": "",
            "TAVILY_API_KEY": "",
        }
        with patch("llm.load_dotenv"), patch("verifier.load_dotenv"):
            with patch.dict(os.environ, cleared_env, clear=False):
                with patch.dict(sys.modules, {"streamlit": fake_st}):
                    self.assertFalse(llm.is_use_bifrost())
                    api_key, base_url, chain = llm.get_llm_config()
                    self.assertEqual(api_key, "AIza-cloud-secret-key")
                    self.assertEqual(base_url, llm.GEMINI_OPENAI_BASE_URL)
                    self.assertEqual(chain, ["gemini-2.5-flash", "gemini-2.5-flash-lite"])

                    _, _, extract_chain = llm.get_llm_config(task="extract")
                    self.assertEqual(extract_chain, ["gemini-2.5-flash-lite"])

                    with patch("verifier.TavilyClient") as mock_tavily_cls:
                        verifier.get_tavily_client()
                        mock_tavily_cls.assert_called_once_with(api_key="tvly-cloud-secret-key")


if __name__ == "__main__":
    unittest.main()
