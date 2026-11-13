"""
lab/tests/test_runner.py - Unit tests for benchmark runner orchestration, resume, and budgets.
Uses mocks only; guarantees no real network calls and completes under 20 seconds.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from lab.adapters import PythonAdapter
from lab.runner import (
    BudgetTracker,
    compute_dry_run_estimates,
    execute_single_run,
    load_completed_runs,
    run_benchmark,
)
from lab.schema import AgentResult, StepRecord, Task


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_path = Path(self.temp_dir.name) / "runs.jsonl"
        self.suite_path = Path(self.temp_dir.name) / "tasks.json"

        # Create 2 sample tasks in suite
        sample_tasks = [
            {
                "id": "t1",
                "category": "price",
                "input_text": "iPhone 15 is latest",
                "expected_status": "outdated",
                "expected_facts": ["iPhone 16"],
            },
            {
                "id": "t2",
                "category": "leadership",
                "input_text": "Rahul Gandhi is PM",
                "expected_status": "incorrect",
                "expected_facts": ["Modi"],
            },
        ]
        self.suite_path.write_text(json.dumps(sample_tasks), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dry_run_estimation(self):
        tasks = [
            Task(id="1", category="p", input_text="a", expected_status="fresh"),
            Task(id="2", category="p", input_text="b", expected_status="fresh"),
        ]
        est = compute_dry_run_estimates(
            agents=["llm_only", "single_search"],
            tasks=tasks,
            repeats=2,
        )
        # 2 agents * 2 tasks * 2 repeats = 8 runs
        self.assertEqual(est["total_runs"], 8)
        self.assertGreaterEqual(est["est_llm_calls"], 8)

        # Calling run_benchmark with dry_run=True should return empty list and write nothing
        runs = run_benchmark(
            agents=["llm_only"],
            suite_path=self.suite_path,
            results_path=self.results_path,
            repeats=1,
            dry_run=True,
        )
        self.assertEqual(len(runs), 0)
        self.assertFalse(self.results_path.exists())

    def test_resumable_skips_existing_runs(self):
        # Pre-populate results file with run for task 't1'
        existing_run = {
            "run_id": "run-existing-1",
            "agent": "llm_only",
            "task_id": "t1",
            "repeat": 1,
            "scenario": "clean",
            "success": True,
            "result": {"verdict": "outdated", "current_value": "iPhone 16"},
            "steps": [],
            "totals": {},
        }
        self.results_path.write_text(json.dumps(existing_run) + "\n", encoding="utf-8")

        completed = load_completed_runs(self.results_path)
        self.assertIn(("llm_only", "t1", 1, "clean"), completed)

        # Run benchmark with mocked agent
        def fake_agent(task, ctx):
            return AgentResult(verdict=task.expected_status, current_value="Modi")

        with patch("lab.runner.get_agent", return_value=PythonAdapter(fake_agent, name="llm_only")):
            runs = run_benchmark(
                agents=["llm_only"],
                suite_path=self.suite_path,
                results_path=self.results_path,
                repeats=1,
                concurrency=1,
            )

        # Only t2 should have run, since t1 was already completed
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].task_id, "t2")

    def test_budget_limits_stop_runner(self):
        budget_tracker = BudgetTracker(max_llm_calls=2)

        def agent_with_llm(task, ctx):
            ctx.add_step(StepRecord(idx=1, kind="llm", model="gpt-4o-mini", tokens_in=50, tokens_out=10))
            return AgentResult(verdict="fresh")

        adapter = PythonAdapter(agent_with_llm, name="test")
        task = Task(id="t1", category="c", input_text="claim", expected_status="fresh")
        pricing = {"gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60}}

        # 1st run: consumes 1 LLM call
        rec1 = execute_single_run(task, "test", adapter, 1, "clean", 30.0, True, budget_tracker, pricing)
        self.assertIsNotNone(rec1)
        self.assertFalse(budget_tracker.is_exhausted())

        # 2nd run: consumes 2nd LLM call -> triggers budget limit
        rec2 = execute_single_run(task, "test", adapter, 2, "clean", 30.0, True, budget_tracker, pricing)
        self.assertIsNotNone(rec2)
        self.assertTrue(budget_tracker.is_exhausted())

        # 3rd run: should be rejected immediately due to exhausted budget
        rec3 = execute_single_run(task, "test", adapter, 3, "clean", 30.0, True, budget_tracker, pricing)
        self.assertIsNone(rec3)

    def test_backoff_retry_on_transient_error(self):
        budget_tracker = BudgetTracker()
        task = Task(id="t1", category="c", input_text="claim", expected_status="fresh")
        pricing = {}

        call_count = [0]

        def flaky_agent(task, ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("HTTP 429 Too Many Requests: Rate limit exceeded")
            return AgentResult(verdict="fresh")

        adapter = PythonAdapter(flaky_agent, name="flaky")

        with patch("time.sleep", return_value=None):  # Fast forward sleep in test
            rec = execute_single_run(task, "flaky", adapter, 1, "clean", 30.0, True, budget_tracker, pricing)

        self.assertIsNotNone(rec)
        self.assertEqual(rec.result.verdict, "fresh")
        self.assertEqual(call_count[0], 2, "Should have retried after 429")


if __name__ == "__main__":
    unittest.main()
