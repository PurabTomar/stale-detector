"""
lab/tests/test_scoring.py - Unit tests for deterministic non-LLM scoring engine.
"""

import unittest
from lab.schema import AgentResult, StepRecord
from lab.scoring import compute_run_totals, estimate_run_cost, evaluate_success


class TestScoring(unittest.TestCase):
    def test_evaluate_success_exact_and_case_insensitive(self):
        # Match status and contains fact
        self.assertTrue(
            evaluate_success(
                verdict="outdated",
                expected_status="outdated",
                current_value="The current model is iPhone 16 Pro Max",
                expected_facts=["iPhone 16", "A18"],
            )
        )
        # Case insensitive fact matching
        self.assertTrue(
            evaluate_success(
                verdict="INCORRECT",
                expected_status="incorrect",
                current_value="the capital is CANBERRA, not sydney",
                expected_facts=["canberra"],
            )
        )

    def test_evaluate_success_verdict_mismatch_fails(self):
        self.assertFalse(
            evaluate_success(
                verdict="fresh",
                expected_status="outdated",
                current_value="iPhone 16",
                expected_facts=["iPhone 16"],
            )
        )

    def test_evaluate_success_fact_missing_fails(self):
        # Verdict matches but none of the expected facts appear in current_value
        self.assertFalse(
            evaluate_success(
                verdict="outdated",
                expected_status="outdated",
                current_value="Some other text that does not name the new phone",
                expected_facts=["iPhone 16", "A18"],
            )
        )

    def test_evaluate_success_empty_expected_facts(self):
        # When no expected facts are specified (e.g. uncertain), status match alone succeeds
        self.assertTrue(
            evaluate_success(
                verdict="uncertain",
                expected_status="uncertain",
                current_value=None,
                expected_facts=[],
            )
        )

    def test_estimate_run_cost_with_known_and_placeholder_pricing(self):
        mock_pricing = {
            "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
            "unpriced-model": None,
            "default": None,
        }

        # 1. Known model with pricing
        steps = [
            StepRecord(idx=1, kind="llm", model="gpt-4o-mini", tokens_in=1_000_000, tokens_out=1_000_000),
            StepRecord(idx=2, kind="tool", model=None, tokens_in=0, tokens_out=0),
        ]
        cost, display = estimate_run_cost(steps, pricing=mock_pricing)
        self.assertIsNotNone(cost)
        # 0.15 + 0.60 = 0.75 USD
        self.assertAlmostEqual(cost, 0.75, places=5)
        self.assertEqual(display, "$0.750000")

        # 2. Model with null pricing defaults to "cost not set"
        steps_unpriced = [
            StepRecord(idx=1, kind="llm", model="unpriced-model", tokens_in=100, tokens_out=50),
        ]
        cost_null, display_null = estimate_run_cost(steps_unpriced, pricing=mock_pricing)
        self.assertIsNone(cost_null)
        self.assertEqual(display_null, "cost not set")

        # 3. Model not in table and no default rate -> "cost not set"
        steps_unknown = [
            StepRecord(idx=1, kind="llm", model="mysterious-model", tokens_in=100, tokens_out=50),
        ]
        cost_unk, display_unk = estimate_run_cost(steps_unknown, pricing=mock_pricing)
        self.assertIsNone(cost_unk)
        self.assertEqual(display_unk, "cost not set")

    def test_compute_run_totals(self):
        steps = [
            StepRecord(idx=1, kind="tool", latency_ms=120.0),
            StepRecord(idx=2, kind="llm", model="gpt-4o-mini", tokens_in=200, tokens_out=50, latency_ms=450.0),
            StepRecord(idx=3, kind="tool", latency_ms=100.0),
            StepRecord(idx=4, kind="fallback", latency_ms=0.0),
        ]
        res = AgentResult(verdict="fresh", current_value="ok")
        totals = compute_run_totals(steps, duration_ms=700.0, result=res)

        self.assertEqual(totals["steps"], 4)
        self.assertEqual(totals["tool_calls"], 2)
        self.assertEqual(totals["llm_calls"], 1)
        self.assertEqual(totals["fallback_calls"], 1)
        self.assertEqual(totals["tokens_in"], 200)
        self.assertEqual(totals["tokens_out"], 50)
        self.assertEqual(totals["total_tokens"], 250)
        self.assertGreaterEqual(totals["total_latency_ms"], 670.0)


if __name__ == "__main__":
    unittest.main()
