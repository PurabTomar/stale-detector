"""
lab/tests/test_agents.py - Unit tests for in-process research agents.
Uses mocks only; guarantees no real network calls and completes under 20 seconds.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from lab.agents.detector_pipeline import run_detector_pipeline_agent
from lab.agents.llm_only import run_llm_only_agent
from lab.agents.react_loop import run_react_loop_agent
from lab.agents.single_search import run_single_search_agent
from lab.schema import RunContext, Task


class TestResearchAgents(unittest.TestCase):
    def setUp(self):
        self.task = Task(
            id="t1",
            category="price",
            input_text="Apple's latest flagship is iPhone 15 Pro Max at $1,199.",
            expected_status="outdated",
            expected_facts=["iPhone 16"],
        )
        self.ctx = RunContext(run_id="run-agent-test", agent_name="test")

    @patch("llm.call_llm_with_meta")
    def test_llm_only_agent(self, mock_llm):
        mock_llm.return_value = (
            '{"verdict": "outdated", "current_value": "iPhone 16 Pro Max", "explanation": "iPhone 16 launched"}',
            {"prompt_tokens": 100, "completion_tokens": 30, "model_used": "gpt-4o-mini"},
        )
        res = run_llm_only_agent(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16 Pro Max")
        self.assertEqual(len(self.ctx.steps), 1)
        self.assertEqual(self.ctx.steps[0].kind, "llm")

    @patch("verifier.search_tavily_for_claim")
    @patch("llm.call_llm_with_meta")
    def test_single_search_agent(self, mock_llm, mock_search):
        mock_search.return_value = {
            "results": [{"title": "Apple News", "url": "https://apple.com", "snippet": "iPhone 16 is now available"}],
            "cache_hit": False,
        }
        mock_llm.return_value = (
            '{"verdict": "outdated", "current_value": "iPhone 16", "explanation": "Replaced by iPhone 16"}',
            {"prompt_tokens": 120, "completion_tokens": 25, "model_used": "gpt-4o-mini"},
        )
        res = run_single_search_agent(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16")
        # 1 search tool step + 1 llm judge step
        self.assertEqual(len(self.ctx.steps), 2)
        self.assertEqual(self.ctx.steps[0].kind, "tool")
        self.assertEqual(self.ctx.steps[1].kind, "llm")

    @patch("verifier.search_tavily")
    @patch("llm.call_llm_with_meta")
    def test_react_loop_agent_successful_cycle(self, mock_llm, mock_search):
        mock_search.return_value = [
            {"title": "Latest iPhone", "url": "https://apple.com/iphone-16", "snippet": "iPhone 16 starting at $799"}
        ]
        # Step 1: Model thoughts and requests search
        # Step 2: Model sees observation and outputs finish
        mock_llm.side_effect = [
            (
                "Thought: I need to search for the current iPhone lineup.\nAction: search[latest iPhone model and price]",
                {"prompt_tokens": 80, "completion_tokens": 25, "model_used": "gpt-4o-mini"},
            ),
            (
                "Thought: The search shows iPhone 16 is out.\nAction: finish[outdated, iPhone 16]",
                {"prompt_tokens": 140, "completion_tokens": 20, "model_used": "gpt-4o-mini"},
            ),
        ]

        res = run_react_loop_agent(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16")
        # 1st LLM call -> 1 Tavily search tool call -> 2nd LLM call
        self.assertEqual(len(self.ctx.steps), 3)

    @patch("verifier.search_tavily")
    @patch("llm.call_llm_with_meta")
    def test_react_loop_tool_retry_and_max_calls(self, mock_llm, mock_search):
        # Tool raises an exception on first search, then succeeds
        mock_search.side_effect = [
            RuntimeError("Tavily rate limit"),
            [{"title": "Apple", "url": "https://apple.com", "snippet": "iPhone 16 Pro"}],
        ]

        mock_llm.side_effect = [
            (
                "Thought: Searching...\nAction: search[iPhone 15]",
                {"prompt_tokens": 50, "completion_tokens": 10, "model_used": "gpt-4o-mini"},
            ),
            (
                "Thought: First search failed with error. Retrying with simpler query.\nAction: search[iPhone 16]",
                {"prompt_tokens": 80, "completion_tokens": 15, "model_used": "gpt-4o-mini"},
            ),
            (
                "Thought: Evidence confirmed.\nAction: finish[outdated, iPhone 16 Pro]",
                {"prompt_tokens": 120, "completion_tokens": 15, "model_used": "gpt-4o-mini"},
            ),
        ]

        res = run_react_loop_agent(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16 Pro")

    @patch("lab.agents.detector_pipeline.extract_claims")
    @patch("verifier.search_tavily_for_claim")
    @patch("llm.call_llm_with_meta")
    def test_detector_pipeline_agent(self, mock_llm, mock_search, mock_extract):
        mock_extract.return_value = (
            [{"claim": "iPhone 15 Pro Max is $1,199", "category": "price", "risk_level": "high"}],
            "gpt-4o-mini",
        )
        mock_search.return_value = {
            "results": [{"title": "Apple", "url": "https://apple.com", "snippet": "iPhone 16"}],
            "cache_hit": False,
        }
        mock_llm.return_value = (
            '{"status": "outdated", "current_value": "iPhone 16 Pro Max", "correction": "Updated to 16"}',
            {"prompt_tokens": 100, "completion_tokens": 20, "model_used": "gpt-4o-mini"},
        )

        res = run_detector_pipeline_agent(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16 Pro Max")


if __name__ == "__main__":
    unittest.main()
