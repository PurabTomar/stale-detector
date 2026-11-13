"""
lab/tests/test_schema.py - Unit tests for Agent Lab schemas.
"""

import unittest
from lab.schema import AgentResult, RunContext, RunRecord, StepRecord, Task, VALID_STATUSES


class TestAgentLabSchema(unittest.TestCase):
    def test_task_creation_and_validation(self):
        task = Task(
            id="t1",
            category="price",
            input_text="iPhone costs $999",
            expected_status="outdated",
            expected_facts=["$1,199"],
            needs_human_check=True,
            tags=["hardware"],
        )
        self.assertEqual(task.id, "t1")
        self.assertEqual(task.expected_status, "outdated")
        self.assertTrue(task.needs_human_check)

        d = task.to_dict()
        self.assertEqual(d["category"], "price")

        reconstructed = Task.from_dict(d)
        self.assertEqual(reconstructed.id, "t1")
        self.assertEqual(reconstructed.expected_facts, ["$1,199"])

        # Invalid status should raise ValueError
        with self.assertRaises(ValueError):
            Task(
                id="t2",
                category="price",
                input_text="bad status",
                expected_status="invalid_status",
            )

    def test_agent_result_normalization(self):
        # Valid status
        res1 = AgentResult(verdict="fresh", current_value="15")
        self.assertEqual(res1.verdict, "fresh")

        # Unknown status defensively normalized to uncertain
        res2 = AgentResult(verdict="unrecognized_gibberish", current_value=None)
        self.assertEqual(res2.verdict, "uncertain")

        d = res1.to_dict()
        reconstructed = AgentResult.from_dict(d)
        self.assertEqual(reconstructed.verdict, "fresh")
        self.assertEqual(reconstructed.current_value, "15")

    def test_step_record_serialization(self):
        step = StepRecord(
            idx=1,
            ts=1700000000.0,
            kind="llm",
            name="judge_step",
            input_summary="prompt text",
            output_summary="response text",
            model="gpt-4o-mini",
            tokens_in=100,
            tokens_out=25,
            latency_ms=350.5,
            cache_hit=False,
            error=None,
        )
        d = step.to_dict()
        self.assertEqual(d["kind"], "llm")
        self.assertEqual(d["tokens_in"], 100)

        reconstructed = StepRecord.from_dict(d)
        self.assertEqual(reconstructed.model, "gpt-4o-mini")
        self.assertEqual(reconstructed.latency_ms, 350.5)

    def test_run_record_serialization_and_json(self):
        step = StepRecord(idx=1, kind="tool", name="search", latency_ms=100.0)
        res = AgentResult(verdict="outdated", current_value="16 Pro")
        run = RunRecord(
            run_id="run-123",
            agent="react_loop",
            task_id="t1",
            repeat=1,
            scenario="clean",
            success=True,
            result=res,
            steps=[step],
            totals={"steps": 1, "tool_calls": 1},
            is_black_box=False,
        )
        json_str = run.to_json()
        self.assertIn('"run_id": "run-123"', json_str)
        self.assertIn('"agent": "react_loop"', json_str)

        d = run.to_dict()
        reconstructed = RunRecord.from_dict(d)
        self.assertEqual(reconstructed.run_id, "run-123")
        self.assertEqual(reconstructed.result.verdict, "outdated")
        self.assertEqual(len(reconstructed.steps), 1)
        self.assertTrue(reconstructed.success)

    def test_run_context_step_accumulator(self):
        ctx = RunContext(run_id="r1", agent_name="test_agent")
        self.assertEqual(len(ctx.steps), 0)

        s1 = StepRecord(idx=0, kind="tool", name="step1")
        ctx.add_step(s1)
        self.assertEqual(len(ctx.steps), 1)
        self.assertEqual(ctx.steps[0].idx, 1)

        s2 = StepRecord(idx=0, kind="llm", name="step2")
        ctx.add_step(s2)
        self.assertEqual(ctx.steps[1].idx, 2)


if __name__ == "__main__":
    unittest.main()
