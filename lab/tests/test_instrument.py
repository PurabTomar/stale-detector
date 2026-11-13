"""
lab/tests/test_instrument.py - Unit tests for secret redaction and instrumented clients.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from lab.instrument import InstrumentedLLMClient, InstrumentedTavilyClient, redact_dict, redact_secrets
from lab.schema import RunContext


class TestInstrument(unittest.TestCase):
    def test_redact_secrets_patterns(self):
        # API keys and bearer tokens
        t1 = "Bearer mysecrettoken1234567890abcdef and Authorization: Bearer xyz1234567890"
        redacted1 = redact_secrets(t1)
        self.assertNotIn("mysecrettoken1234567890abcdef", redacted1)
        self.assertIn("[REDACTED]", redacted1)

        # Virtual keys and headers
        t2 = "Calling Bifrost with x-bf-vk: my-virtual-key-secret-12345"
        redacted2 = redact_secrets(t2)
        self.assertNotIn("my-virtual-key-secret-12345", redacted2)
        self.assertIn("[REDACTED]", redacted2)

        # Active env secret
        os.environ["TAVILY_API_KEY"] = "my_temp_secret_test_tavily_val"
        try:
            t3 = "Using key my_temp_secret_test_tavily_val in request"
            redacted3 = redact_secrets(t3)
            self.assertNotIn("my_temp_secret_test_tavily_val", redacted3)
            self.assertIn("[REDACTED]", redacted3)
        finally:
            os.environ.pop("TAVILY_API_KEY", None)

    def test_redact_dict_recursive(self):
        d = {
            "api_key": "secret-value",
            "normal_field": "hello world",
            "nested": {
                "auth_header": "Bearer secrettoken123456789",
                "count": 5,
            },
        }
        cleaned = redact_dict(d)
        self.assertEqual(cleaned["api_key"], "[REDACTED]")
        self.assertEqual(cleaned["normal_field"], "hello world")
        self.assertEqual(cleaned["nested"]["auth_header"], "[REDACTED]")
        self.assertEqual(cleaned["nested"]["count"], 5)

    @patch("llm.call_llm_with_meta")
    def test_instrumented_llm_client_records_steps_and_tokens(self, mock_llm_call):
        mock_llm_call.return_value = (
            '{"verdict": "fresh"}',
            {
                "prompt_tokens": 150,
                "completion_tokens": 30,
                "model_used": "gpt-4o-mini",
                "fallback_used": False,
            },
        )
        ctx = RunContext(run_id="run-llm-test", agent_name="llm_agent")
        client = InstrumentedLLMClient(ctx)

        content, meta = client.call(
            system_prompt="sys prompt",
            user_prompt="user prompt",
            task="test_llm_step",
        )
        self.assertEqual(content, '{"verdict": "fresh"}')
        self.assertEqual(len(ctx.steps), 1)

        step = ctx.steps[0]
        self.assertEqual(step.kind, "llm")
        self.assertEqual(step.name, "test_llm_step")
        self.assertEqual(step.model, "gpt-4o-mini")
        self.assertEqual(step.tokens_in, 150)
        self.assertEqual(step.tokens_out, 30)

    @patch("llm.call_llm_with_meta")
    def test_instrumented_llm_client_emits_fallback_step(self, mock_llm_call):
        mock_llm_call.return_value = (
            '{"verdict": "outdated"}',
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "primary_model": "gpt-4o-mini",
                "model_used": "gemini-2.5-flash",
                "fallback_used": True,
            },
        )
        ctx = RunContext(run_id="run-fb-test", agent_name="fb_agent")
        client = InstrumentedLLMClient(ctx)

        client.call("sys", "user", task="judge")
        self.assertEqual(len(ctx.steps), 2)
        self.assertEqual(ctx.steps[0].kind, "llm")
        self.assertEqual(ctx.steps[1].kind, "fallback")
        self.assertEqual(ctx.steps[1].name, "gateway_fallback")
        self.assertEqual(ctx.steps[1].model, "gemini-2.5-flash")

    @patch("verifier.search_tavily_for_claim")
    def test_instrumented_tavily_client_records_tool_step(self, mock_search):
        mock_search.return_value = {
            "results": [
                {"title": "Doc", "url": "https://example.com/doc", "snippet": "info"}
            ],
            "cache_hit": True,
            "tavily_credits_used": 0.0,
        }
        ctx = RunContext(run_id="run-tav-test", agent_name="tav_agent")
        client = InstrumentedTavilyClient(ctx)

        res = client.search_claim(query="iPhone 16 price", category="price")
        self.assertEqual(len(res["results"]), 1)
        self.assertEqual(len(ctx.steps), 1)

        step = ctx.steps[0]
        self.assertEqual(step.kind, "tool")
        self.assertEqual(step.name, "tavily_search")
        self.assertTrue(step.cache_hit)
        self.assertIn("iPhone 16 price", step.input_summary)


if __name__ == "__main__":
    unittest.main()
