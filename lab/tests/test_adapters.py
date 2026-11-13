"""
lab/tests/test_adapters.py - Unit tests for AgentAdapter implementations.
Tests PythonAdapter, HttpAdapter (with local test server), and CliAdapter.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from lab.adapters import CliAdapter, HttpAdapter, PythonAdapter
from lab.schema import AgentResult, RunContext, StepRecord, Task


class _MockAgentHandler(BaseHTTPRequestHandler):
    """Local HTTP mock server for testing HttpAdapter."""

    mode = "with_trace"  # "with_trace", "black_box", "error"

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8")
        req_data = json.loads(post_body)

        if self.mode == "error":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error": "Internal server error"}')
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if self.mode == "black_box":
            # No trace field
            resp = {
                "result": {
                    "verdict": "outdated",
                    "current_value": "iPhone 16",
                    "answer": "Answer without trace",
                }
            }
        else:
            # With trace steps
            resp = {
                "result": {
                    "verdict": "fresh",
                    "current_value": "Python 3.12",
                    "answer": "Current Python release",
                },
                "trace": [
                    {
                        "idx": 1,
                        "kind": "tool",
                        "name": "search",
                        "input_summary": "Python release",
                        "output_summary": "found 2 sources",
                        "latency_ms": 110.0,
                    },
                    {
                        "idx": 2,
                        "kind": "llm",
                        "name": "judge",
                        "input_summary": "judge python",
                        "output_summary": "fresh",
                        "tokens_in": 80,
                        "tokens_out": 20,
                        "latency_ms": 250.0,
                    },
                ],
            }
        self.wfile.write(json.dumps(resp).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Silence console logging in tests


class TestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start a local HTTP server on random free port
        cls.server = HTTPServer(("127.0.0.1", 0), _MockAgentHandler)
        cls.server_port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.task = Task(
            id="t_test",
            category="price",
            input_text="Test claim",
            expected_status="fresh",
            expected_facts=["Python 3.12"],
        )
        self.ctx = RunContext(run_id="run-test", agent_name="test")

    def test_python_adapter_success_and_exception(self):
        # 1. Success returning AgentResult
        def good_agent(task, ctx):
            ctx.add_step(StepRecord(idx=0, kind="tool", name="my_tool"))
            return AgentResult(verdict="fresh", current_value="ok")

        adapter = PythonAdapter(good_agent, name="good")
        res, steps = adapter.run(self.task, self.ctx)
        self.assertEqual(res.verdict, "fresh")
        self.assertEqual(len(steps), 1)

        # 2. Agent throwing exception
        def failing_agent(task, ctx):
            raise RuntimeError("Agent failure")

        fail_ctx = RunContext(run_id="run-fail", agent_name="fail")
        fail_adapter = PythonAdapter(failing_agent, name="fail")
        fail_res, fail_steps = fail_adapter.run(self.task, fail_ctx)
        self.assertEqual(fail_res.verdict, "uncertain")
        self.assertIn("Agent failure", str(fail_res.error))
        self.assertEqual(len(fail_steps), 1)
        self.assertEqual(fail_steps[0].kind, "error")

    def test_http_adapter_with_full_trace(self):
        _MockAgentHandler.mode = "with_trace"
        adapter = HttpAdapter(f"http://127.0.0.1:{self.server_port}/verify", timeout_seconds=5.0)

        res, steps = adapter.run(self.task, self.ctx)
        self.assertEqual(res.verdict, "fresh")
        self.assertEqual(res.current_value, "Python 3.12")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].kind, "tool")
        self.assertEqual(steps[1].kind, "llm")
        self.assertFalse(self.ctx.is_black_box)

    def test_http_adapter_black_box_when_no_trace(self):
        _MockAgentHandler.mode = "black_box"
        adapter = HttpAdapter(f"http://127.0.0.1:{self.server_port}/verify", timeout_seconds=5.0)

        res, steps = adapter.run(self.task, self.ctx)
        self.assertEqual(res.verdict, "outdated")
        self.assertEqual(res.current_value, "iPhone 16")
        self.assertTrue(self.ctx.is_black_box, "Should mark context as black-box when no trace")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].name, "http_agent")

    def test_http_adapter_server_error_handled_gracefully(self):
        _MockAgentHandler.mode = "error"
        adapter = HttpAdapter(f"http://127.0.0.1:{self.server_port}/verify", timeout_seconds=5.0)

        res, steps = adapter.run(self.task, self.ctx)
        self.assertEqual(res.verdict, "uncertain")
        self.assertIsNotNone(res.error)
        self.assertTrue(self.ctx.is_black_box)
        self.assertEqual(steps[0].kind, "error")

    @patch("subprocess.run")
    def test_cli_adapter_with_and_without_trace(self, mock_subproc):
        # 1. CLI with trace
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({
                "result": {"verdict": "fresh", "current_value": "100"},
                "trace": [{"idx": 1, "kind": "tool", "name": "cli_tool", "latency_ms": 50.0}],
            }),
            stderr="",
        )
        cli_adp = CliAdapter(["my_cli", "--flag"])
        res, steps = cli_adp.run(self.task, self.ctx)
        self.assertEqual(res.verdict, "fresh")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].name, "cli_tool")

        # 2. CLI without trace (black-box)
        mock_subproc.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"result": {"verdict": "incorrect", "current_value": "Modi"}}),
            stderr="",
        )
        ctx2 = RunContext(run_id="run-cli-2", agent_name="cli")
        res2, steps2 = cli_adp.run(self.task, ctx2)
        self.assertEqual(res2.verdict, "incorrect")
        self.assertTrue(ctx2.is_black_box)
        self.assertEqual(len(steps2), 1)


if __name__ == "__main__":
    unittest.main()
