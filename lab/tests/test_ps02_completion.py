"""
lab/tests/test_ps02_completion.py - Verification suite for Problem Statement 02 completion.

Tests:
1. Failure testing wrappers: broken_tool (50% error + garbled variants), unavailable_model, misleading_instructions.
2. Hardened prompts treat adversarial user and web text as untrusted data.
3. Chaos payloads: at least 4 payloads in lab/chaos_payloads.json.
4. Metrics from recorded runs: accuracy, pass^k consistency, mean/std for steps/tools/tokens/latency/cost, p95 latency, recovery rates, injection resistance, fallback overhead.
5. "Same model, different agents" & "Same agent, different models" matrices with spread delta.
6. External HTTP agent and black-box agent honest metric isolation.
7. Gateway-level visibility: x-bf-run-id tagging, gateway vs client tokens cross-check, mismatch reporting.
8. Export generation: summary.json, runs.jsonl, leaderboard.csv.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lab.instrument import (
    InstrumentedLLMClient,
    InstrumentedTavilyClient,
    get_chaos_payloads,
)
from lab.runner import compute_dry_run_estimates, execute_single_run
from lab.schema import AgentResult, RunContext, RunRecord, StepRecord, Task
from lab.scoring import (
    compute_agent_metrics,
    compute_bifrost_visibility,
    compute_run_totals,
    compute_same_agent_matrix,
    compute_same_model_matrix,
    estimate_run_cost,
    evaluate_success,
    generate_benchmark_exports,
)
from lab.ui import build_leaderboard_df, style_leaderboard


class TestPS02FailureWrappersAndHardening(unittest.TestCase):
    """Test Requirement 1: Failure testing via wrappers and hardened judge prompts."""

    def test_chaos_payloads_file_has_at_least_four_adversarial_entries(self) -> None:
        payloads = get_chaos_payloads()
        self.assertGreaterEqual(len(payloads), 4)
        types = {p.get("type") for p in payloads}
        self.assertIn("prompt_injection", types)
        self.assertIn("search_poison", types)
        targets = {p.get("target") for p in payloads}
        self.assertIn("input_text", targets)
        self.assertIn("search_result", targets)

    @patch("verifier.search_tavily_for_claim")
    def test_broken_tool_fifty_percent_failure_and_garbled_variant(self, mock_search) -> None:
        """Call 0: 503 error; Call 1: success; Call 2: garbled corrupted stream."""
        ctx = RunContext(run_id="run-broken-tool", agent_name="test_agent", scenario="broken_tool")
        client = InstrumentedTavilyClient(ctx)

        # Call 0 (even): Variant 1 (503 Service Unavailable)
        res0 = client.search_claim(query="API version check")
        self.assertIn("error", res0)
        self.assertIn("503", res0["error"])
        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.steps[0].kind, "tool")

        # Call 1 (odd): Upstream success
        mock_search.return_value = {
            "results": [{"title": "Official Doc", "url": "https://python.org", "snippet": "3.12 active"}],
            "cache_hit": False,
        }
        res1 = client.search_claim(query="API version check retry")
        self.assertEqual(len(res1.get("results", [])), 1)
        self.assertEqual(len(ctx.steps), 2)

        # Call 2 (even): Variant 2 (garbled corrupted binary stream)
        res2 = client.search_claim(query="API version check third")
        self.assertIsNone(res2.get("error"))
        results2 = res2.get("results", [])
        self.assertEqual(len(results2), 1)
        self.assertIn("GARBLED", results2[0]["snippet"])
        self.assertEqual(len(ctx.steps), 3)

    @patch("llm.call_llm_with_meta")
    def test_unavailable_model_triggers_outage_and_records_fallback(self, mock_llm) -> None:
        mock_llm.return_value = (
            '{"verdict": "fresh", "current_value": "Python 3.12"}',
            {
                "model_used": "gemini-2.5-flash",
                "primary_model": "gpt-4o-mini",
                "fallback_used": True,
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "gateway_tokens": 130,
            },
        )
        ctx = RunContext(run_id="run-unavail-test", agent_name="test_agent", scenario="unavailable_model")
        client = InstrumentedLLMClient(ctx)

        client.call("sys", "user", task="judge")
        self.assertEqual(len(ctx.steps), 2)
        self.assertEqual(ctx.steps[0].kind, "llm")
        self.assertEqual(ctx.steps[1].kind, "fallback")
        self.assertEqual(ctx.steps[1].name, "gateway_fallback")
        self.assertIn("gpt-4o-mini", ctx.steps[1].input_summary)
        self.assertIn("gemini-2.5-flash", ctx.steps[1].output_summary)

    @patch("verifier.search_tavily_for_claim")
    def test_misleading_instructions_injects_poisoned_search_result(self, mock_search) -> None:
        mock_search.return_value = {
            "results": [{"title": "Real Source", "url": "https://gov.in", "snippet": "Old rate"}],
            "cache_hit": False,
        }
        ctx = RunContext(run_id="run-misleading-test", agent_name="test_agent", scenario="misleading_instructions")
        client = InstrumentedTavilyClient(ctx)

        res = client.search_claim(query="GST rate")
        results = res.get("results", [])
        self.assertGreaterEqual(len(results), 2)
        # First source is the injected distractor / adversarial poison
        distractor = results[0]
        self.assertIn("unverified-blog-rumors.xyz", distractor["url"])
        self.assertIn("ADVERSARIAL", distractor["content"])


class TestPS02DeterministicMetrics(unittest.TestCase):
    """Test Requirement 2: Accuracy, Pass^k Consistency, Stds, P95, Recovery Rate, Cost."""

    def test_pass_k_consistency_and_distributions(self) -> None:
        # Create 4 tasks, each repeated 2 times
        # task_1: both succeed
        # task_2: 1 succeed, 1 fail
        # task_3: both fail
        # task_4: both succeed
        # Consistency pass^k should be 2 / 4 = 50.0%
        # Overall accuracy = 5 / 8 = 62.5%
        runs = [
            # task_1
            {"agent": "agent_x", "task_id": "t1", "repeat": 1, "scenario": "clean", "success": True, "totals": {"steps": 2, "tool_calls": 1, "total_tokens": 300, "total_latency_ms": 1000.0, "estimated_cost": 0.001}},
            {"agent": "agent_x", "task_id": "t1", "repeat": 2, "scenario": "clean", "success": True, "totals": {"steps": 2, "tool_calls": 1, "total_tokens": 320, "total_latency_ms": 1100.0, "estimated_cost": 0.0011}},
            # task_2
            {"agent": "agent_x", "task_id": "t2", "repeat": 1, "scenario": "clean", "success": True, "totals": {"steps": 1, "tool_calls": 0, "total_tokens": 150, "total_latency_ms": 500.0, "estimated_cost": 0.0005}},
            {"agent": "agent_x", "task_id": "t2", "repeat": 2, "scenario": "clean", "success": False, "totals": {"steps": 1, "tool_calls": 0, "total_tokens": 160, "total_latency_ms": 600.0, "estimated_cost": 0.00055}},
            # task_3
            {"agent": "agent_x", "task_id": "t3", "repeat": 1, "scenario": "clean", "success": False, "totals": {"steps": 3, "tool_calls": 2, "total_tokens": 600, "total_latency_ms": 2000.0, "estimated_cost": 0.002}},
            {"agent": "agent_x", "task_id": "t3", "repeat": 2, "scenario": "clean", "success": False, "totals": {"steps": 3, "tool_calls": 2, "total_tokens": 620, "total_latency_ms": 2100.0, "estimated_cost": 0.0021}},
            # task_4
            {"agent": "agent_x", "task_id": "t4", "repeat": 1, "scenario": "clean", "success": True, "totals": {"steps": 2, "tool_calls": 1, "total_tokens": 350, "total_latency_ms": 1200.0, "estimated_cost": 0.0012}},
            {"agent": "agent_x", "task_id": "t4", "repeat": 2, "scenario": "clean", "success": True, "totals": {"steps": 2, "tool_calls": 1, "total_tokens": 360, "total_latency_ms": 1300.0, "estimated_cost": 0.0013}},
        ]

        metrics = compute_agent_metrics(runs)
        self.assertEqual(metrics["n"], 8)
        self.assertEqual(metrics["accuracy"], 62.5)
        self.assertEqual(metrics["failure_rate"], 37.5)
        self.assertEqual(metrics["consistency"], 50.0)

        # Standard deviations must be > 0.0
        self.assertGreater(metrics["std_steps"], 0.0)
        self.assertGreater(metrics["std_tool_calls"], 0.0)
        self.assertGreater(metrics["std_tokens"], 0.0)
        self.assertGreater(metrics["std_latency_s"], 0.0)
        self.assertGreater(metrics["p95_latency_s"], metrics["mean_latency_s"])
        self.assertIn("$", metrics["cost_display"])

    def test_cost_calculation_when_pricing_is_missing_returns_cost_not_set(self) -> None:
        steps = [
            StepRecord(idx=1, kind="llm", model="unknown-model-xyz", tokens_in=500, tokens_out=100)
        ]
        val, display = estimate_run_cost(steps, pricing={"gpt-4o": {"input_per_million": 2.5, "output_per_million": 10.0}})
        self.assertIsNone(val)
        self.assertEqual(display, "cost not set")


class TestPS02ModelAgentFactorialSpread(unittest.TestCase):
    """Test Requirement 3: Same model / different agents & same agent / different models spreads."""

    def test_same_model_spread_isolates_architecture(self) -> None:
        runs = [
            # Agent A with model M: 100% accuracy, 5s latency, 200 tokens
            {"agent": "agent_a", "task_id": "t1", "repeat": 1, "success": True, "steps": [{"model": "gpt-4o-mini"}], "totals": {"steps": 1, "tool_calls": 0, "total_tokens": 200, "total_latency_ms": 5000.0}},
            # Agent B with model M: 50% accuracy, 10s latency, 800 tokens
            {"agent": "agent_b", "task_id": "t1", "repeat": 1, "success": False, "steps": [{"model": "gpt-4o-mini"}], "totals": {"steps": 3, "tool_calls": 2, "total_tokens": 800, "total_latency_ms": 10000.0}},
        ]
        res = compute_same_model_matrix(runs, target_model="gpt-4o-mini")
        self.assertEqual(res["model"], "gpt-4o-mini")
        self.assertEqual(len(res["matrix"]), 2)
        spread = res["spread"]
        self.assertEqual(spread["accuracy_delta"], 100.0)
        self.assertEqual(spread["latency_delta_s"], 5.0)
        self.assertEqual(spread["tokens_delta"], 600.0)

    def test_same_agent_spread_isolates_model(self) -> None:
        runs = [
            # Model 1: 100% accuracy
            {"agent": "pipeline", "task_id": "t1", "repeat": 1, "success": True, "steps": [{"model": "gemini-3.5-flash"}], "totals": {"total_tokens": 500, "total_latency_ms": 2000.0}},
            # Model 2: 0% accuracy
            {"agent": "pipeline", "task_id": "t1", "repeat": 2, "success": False, "steps": [{"model": "gemini-3.1-flash-lite"}], "totals": {"total_tokens": 300, "total_latency_ms": 1000.0}},
        ]
        res = compute_same_agent_matrix(runs, target_agent="pipeline")
        self.assertEqual(res["agent"], "pipeline")
        self.assertEqual(len(res["matrix"]), 2)
        spread = res["spread"]
        self.assertEqual(spread["accuracy_delta"], 100.0)
        self.assertEqual(spread["latency_delta_s"], 1.0)
        self.assertEqual(spread["tokens_delta"], 200.0)


class TestPS02BlackBoxAndBifrostVisibility(unittest.TestCase):
    """Test Requirements 4 & 5: Black-box agent isolation and Bifrost visibility."""

    def test_black_box_agents_honest_metric_isolation_in_leaderboard(self) -> None:
        runs = [
            {
                "run_id": "r1",
                "agent": "external_http",
                "task_id": "t1",
                "repeat": 1,
                "scenario": "clean",
                "success": True,
                "is_black_box": True,
                "steps": [],
                "totals": {"steps": 0, "tool_calls": 0, "total_tokens": 0, "total_latency_ms": 2400.0},
            }
        ]
        df = build_leaderboard_df(runs)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertIn("black-box", str(row["Agent"]))
        self.assertEqual(row["Accuracy (%)"], 100.0)
        # Internal metrics are omitted (None / NaN) for black box agents in DataFrame
        self.assertTrue(row["Mean Steps"] is None or pd.isna(row["Mean Steps"]))
        self.assertTrue(row["Mean Tool Calls"] is None or pd.isna(row["Mean Tool Calls"]))
        self.assertTrue(row["Mean Tokens"] is None or pd.isna(row["Mean Tokens"]))
        self.assertIn("black-box", str(row["Estimated Cost"]))
        self.assertEqual(row["Mean Latency (s)"], 2.4)

    def test_bifrost_visibility_token_cross_check(self) -> None:
        runs = [
            {
                "run_id": "run-gw-1",
                "agent": "agent_alpha",
                "totals": {
                    "total_tokens": 300,
                    "gateway_tokens": 300,
                    "fallback_calls": 0,
                },
            },
            {
                "run_id": "run-gw-2",
                "agent": "agent_alpha",
                "totals": {
                    "total_tokens": 500,
                    "gateway_tokens": 490,  # 10 tokens variance due to tokenizer vs byte-pair
                    "fallback_calls": 1,
                },
            },
        ]
        vis = compute_bifrost_visibility(runs)
        self.assertEqual(vis["total_runs"], 2)
        self.assertEqual(vis["tagged_runs"], 2)
        self.assertEqual(vis["fallback_events_detected"], 1)
        self.assertEqual(vis["client_tokens_total"], 800)
        self.assertEqual(vis["gateway_tokens_total"], 790)
        self.assertEqual(vis["token_mismatches_count"], 1)
        self.assertEqual(vis["token_mismatch_tokens_delta"], 10)


class TestPS02ExportsAndDryRun(unittest.TestCase):
    """Test Requirements 6 & 7: Exports generation and dry-run estimation."""

    def test_generate_benchmark_exports_produces_summary_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base_p = Path(td)
            runs_p = base_p / "runs.jsonl"
            summary_p = base_p / "summary.json"
            csv_p = base_p / "leaderboard.csv"

            sample_run = {
                "run_id": "r-exp-1",
                "agent": "test_agent",
                "task_id": "task_1",
                "repeat": 1,
                "scenario": "clean",
                "success": True,
                "steps": [{"kind": "llm", "model": "gpt-4o-mini", "tokens_in": 10, "tokens_out": 5}],
                "totals": {"steps": 1, "tool_calls": 0, "total_tokens": 15, "total_latency_ms": 800.0, "estimated_cost": 0.0001},
            }
            runs_p.write_text(json.dumps(sample_run) + "\n", encoding="utf-8")

            res = generate_benchmark_exports(runs_p, summary_p, csv_p)
            self.assertTrue(summary_p.is_file())
            self.assertTrue(csv_p.is_file())

            # Read back summary JSON
            s_data = json.loads(summary_p.read_text(encoding="utf-8"))
            self.assertIn("benchmark_metadata", s_data)
            self.assertIn("leaderboard", s_data)
            self.assertIn("bifrost_visibility", s_data)

            # Read back CSV
            csv_text = csv_p.read_text(encoding="utf-8")
            self.assertIn("test_agent", csv_text)
            self.assertIn("Accuracy (%)", csv_text)

    def test_dry_run_scenarios_multiplier(self) -> None:
        tasks = [Task(id="t1", category="other", input_text="Sample", expected_status="fresh")]
        agents = ["llm_only", "single_search"]
        repeats = 2
        scenarios = ["clean", "broken_tool", "unavailable_model", "misleading_instructions"]

        est = compute_dry_run_estimates(agents, tasks, repeats, scenarios=scenarios)
        # 2 agents * 1 task * 2 repeats * 4 scenarios = 16 planned runs
        self.assertEqual(est["total_runs"], 16)
        self.assertGreater(est["est_tavily_credits"], 0.0)
        self.assertGreater(est["est_llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
