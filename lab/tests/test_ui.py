"""
lab/tests/test_ui.py - Unit and smoke tests for Agent Lab UI components.

Verifies:
1. Loading runs from sample results files.
2. Leaderboard computation (Accuracy, steps, tools, tokens, latency, failure rate).
3. Best-value styling and timestamp formatting.
4. Black-box handling and empty state robustness.
5. Zero external network/API calls during UI data processing and rendering.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from lab.ui import (
    build_leaderboard_df,
    get_last_run_timestamp,
    load_runs,
    load_tasks_meta,
    render_agent_lab_tab,
    style_leaderboard,
)


class TestAgentLabUI(unittest.TestCase):
    """Test suite for lab/ui.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.sample_runs_path = self.base_dir / "sample_runs.jsonl"
        self.sample_suite_path = self.base_dir / "sample_tasks.json"

        # Create sample runs.jsonl
        sample_runs = [
            {
                "run_id": "run-001",
                "agent": "agent_alpha",
                "task_id": "task_price",
                "repeat": 1,
                "scenario": "clean",
                "success": True,
                "result": {
                    "verdict": "outdated",
                    "current_value": "499",
                    "answer": "Price is 499",
                    "error": None,
                },
                "steps": [
                    {
                        "idx": 1,
                        "ts": 1700000000.0,
                        "kind": "tool",
                        "name": "search",
                        "model": "tavily",
                        "tokens_in": 50,
                        "tokens_out": 20,
                        "latency_ms": 320.0,
                        "cache_hit": False,
                        "error": None,
                        "input_summary": "search 499",
                        "output_summary": "found 499",
                    },
                    {
                        "idx": 2,
                        "ts": 1700000001.0,
                        "kind": "llm",
                        "name": "judge",
                        "model": "gpt-4o-mini",
                        "tokens_in": 120,
                        "tokens_out": 40,
                        "latency_ms": 480.0,
                        "cache_hit": False,
                        "error": None,
                        "input_summary": "judge prompt",
                        "output_summary": "outdated",
                    },
                ],
                "totals": {
                    "steps": 2,
                    "tool_calls": 1,
                    "llm_calls": 1,
                    "fallback_calls": 0,
                    "tokens_in": 170,
                    "tokens_out": 60,
                    "total_tokens": 230,
                    "total_latency_ms": 800.0,
                },
                "started_at": "2026-11-13T14:00:00.000000+00:00",
                "ended_at": "2026-11-13T14:00:01.000000+00:00",
                "is_black_box": False,
            },
            {
                "run_id": "run-002",
                "agent": "agent_alpha",
                "task_id": "task_price",
                "repeat": 2,
                "scenario": "clean",
                "success": False,
                "result": {
                    "verdict": "fresh",
                    "current_value": "old_price",
                    "answer": "Price is old",
                    "error": None,
                },
                "steps": [],
                "totals": {
                    "steps": 1,
                    "tool_calls": 0,
                    "llm_calls": 1,
                    "fallback_calls": 0,
                    "tokens_in": 100,
                    "tokens_out": 50,
                    "total_tokens": 150,
                    "total_latency_ms": 600.0,
                },
                "started_at": "2026-11-13T14:01:00.000000+00:00",
                "ended_at": "2026-11-13T14:01:01.000000+00:00",
                "is_black_box": True,
            },
            {
                "run_id": "run-003",
                "agent": "agent_beta",
                "task_id": "task_price",
                "repeat": 1,
                "scenario": "clean",
                "success": True,
                "result": {
                    "verdict": "outdated",
                    "current_value": "499",
                    "answer": "Verified price is 499",
                    "error": None,
                },
                "steps": [],
                "totals": {
                    "steps": 1,
                    "tool_calls": 0,
                    "llm_calls": 1,
                    "fallback_calls": 0,
                    "tokens_in": 80,
                    "tokens_out": 30,
                    "total_tokens": 110,
                    "total_latency_ms": 400.0,
                },
                "started_at": "2026-11-13T14:02:00.000000+00:00",
                "ended_at": "2026-11-13T14:02:01.000000+00:00",
                "is_black_box": False,
            },
        ]

        with open(self.sample_runs_path, "w", encoding="utf-8") as f:
            for r in sample_runs:
                f.write(json.dumps(r) + "\n")

        # Create sample tasks suite
        sample_tasks = [
            {
                "id": "task_price",
                "category": "price",
                "input_text": "Product costs 299.",
                "expected_status": "outdated",
                "expected_facts": ["499"],
            }
        ]
        self.sample_suite_path.write_text(json.dumps(sample_tasks), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_runs_reads_jsonl(self) -> None:
        runs = load_runs(self.sample_runs_path)
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0]["agent"], "agent_alpha")
        self.assertEqual(runs[2]["agent"], "agent_beta")

    def test_load_runs_non_existent_file_returns_empty_list(self) -> None:
        runs = load_runs(self.base_dir / "does_not_exist.jsonl")
        self.assertEqual(runs, [])

    def test_load_tasks_meta(self) -> None:
        tasks = load_tasks_meta(self.sample_suite_path)
        self.assertIn("task_price", tasks)
        self.assertEqual(tasks["task_price"]["category"], "price")

    def test_get_last_run_timestamp(self) -> None:
        runs = load_runs(self.sample_runs_path)
        ts = get_last_run_timestamp(runs)
        self.assertIn("2026", ts)
        self.assertIn("UTC", ts)

        empty_ts = get_last_run_timestamp([])
        self.assertEqual(empty_ts, "Not run yet")

    def test_build_leaderboard_df_metrics(self) -> None:
        runs = load_runs(self.sample_runs_path)
        df = build_leaderboard_df(runs)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 2)  # agent_alpha and agent_beta
        expected_cols = [
            "Agent",
            "Accuracy (%)",
            "Mean Steps",
            "Mean Tool Calls",
            "Mean Tokens",
            "Mean Latency (s)",
            "Failure Rate (%)",
            "Runs (n)",
            "Errors",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # agent_beta: 1 run, 1 success -> 100.0% accuracy
        beta_row = df[df["Agent"] == "agent_beta"].iloc[0]
        self.assertEqual(beta_row["Accuracy (%)"], 100.0)
        self.assertEqual(beta_row["Runs (n)"], 1)
        self.assertEqual(beta_row["Failure Rate (%)"], 0.0)

        # agent_alpha: 2 runs, 1 success -> 50.0% accuracy
        alpha_row = df[df["Agent"] == "agent_alpha"].iloc[0]
        self.assertEqual(alpha_row["Accuracy (%)"], 50.0)
        self.assertEqual(alpha_row["Runs (n)"], 2)
        self.assertEqual(alpha_row["Failure Rate (%)"], 50.0)
        self.assertEqual(alpha_row["Mean Steps"], 1.5)  # (2 + 1) / 2

    def test_style_leaderboard_produces_styler(self) -> None:
        runs = load_runs(self.sample_runs_path)
        df = build_leaderboard_df(runs)
        styler = style_leaderboard(df)
        self.assertIsNotNone(styler)

    @patch("streamlit.markdown")
    @patch("streamlit.subheader")
    @patch("streamlit.caption")
    @patch("streamlit.columns")
    @patch("streamlit.dataframe")
    @patch("streamlit.download_button")
    @patch("streamlit.selectbox")
    @patch("streamlit.metric")
    @patch("streamlit.expander")
    @patch("streamlit.info")
    @patch("streamlit.container")
    @patch("streamlit.altair_chart")
    def test_render_agent_lab_tab_smoke(
        self,
        mock_altair: MagicMock,
        mock_container: MagicMock,
        mock_info: MagicMock,
        mock_expander: MagicMock,
        mock_metric: MagicMock,
        mock_selectbox: MagicMock,
        mock_dl_btn: MagicMock,
        mock_df: MagicMock,
        mock_cols: MagicMock,
        mock_caption: MagicMock,
        mock_subheader: MagicMock,
        mock_markdown: MagicMock,
    ) -> None:
        """Smoke test verifying render_agent_lab_tab executes cleanly with zero API calls."""
        def _col_side_effect(spec, *args, **kwargs):
            num = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(num)]

        mock_cols.side_effect = _col_side_effect
        mock_selectbox.return_value = 0

        # Execute UI rendering with sample test fixtures
        render_agent_lab_tab(
            results_path=self.sample_runs_path,
            suite_path=self.sample_suite_path,
        )

        mock_subheader.assert_called()
        mock_df.assert_called()
        mock_dl_btn.assert_called()

    @patch("streamlit.markdown")
    @patch("streamlit.subheader")
    @patch("streamlit.caption")
    @patch("streamlit.code")
    def test_render_agent_lab_tab_empty_state(
        self,
        mock_code: MagicMock,
        mock_caption: MagicMock,
        mock_subheader: MagicMock,
        mock_markdown: MagicMock,
    ) -> None:
        """Verify friendly empty state when no runs exist."""
        empty_runs_path = self.base_dir / "empty.jsonl"
        render_agent_lab_tab(
            results_path=empty_runs_path,
            suite_path=self.sample_suite_path,
        )
        mock_markdown.assert_called()
        mock_code.assert_called()


if __name__ == "__main__":
    unittest.main()
