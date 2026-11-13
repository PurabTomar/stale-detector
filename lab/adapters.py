"""
lab/adapters.py - Framework-independent agent adapters.

Supports:
1. PythonAdapter: in-process callable agents.
2. HttpAdapter: HTTP POST contract with optional trace support (or black-box fallback).
3. CliAdapter: Subprocess CLI contract with optional trace support (or black-box fallback).
"""

import json
import subprocess
import time
from typing import Any, Callable, Protocol, runtime_checkable
import urllib.error
import urllib.request

from lab.instrument import redact_secrets
from lab.schema import AgentResult, RunContext, StepRecord, Task


@runtime_checkable
class AgentAdapter(Protocol):
    """Protocol for framework-independent research agent execution."""

    def run(self, task: Task, ctx: RunContext) -> tuple[AgentResult, list[StepRecord]]:
        """Execute agent on task with telemetry context and return result + steps."""
        ...


class PythonAdapter:
    """Adapter for in-process callable agents."""

    def __init__(
        self,
        agent_fn: Callable[[Task, RunContext], AgentResult | tuple[AgentResult, list[StepRecord]]],
        name: str = "python_agent",
    ) -> None:
        self.agent_fn = agent_fn
        self.name = name

    def run(self, task: Task, ctx: RunContext) -> tuple[AgentResult, list[StepRecord]]:
        t0 = time.perf_counter()
        try:
            res = self.agent_fn(task, ctx)
            if isinstance(res, tuple) and len(res) == 2:
                result_obj, steps_list = res
                return result_obj, steps_list
            elif isinstance(res, AgentResult):
                return res, ctx.steps
            else:
                verdict = str(getattr(res, "verdict", "uncertain"))
                curr = getattr(res, "current_value", None)
                return AgentResult(verdict=verdict, current_value=curr), ctx.steps
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            err_step = StepRecord(
                idx=len(ctx.steps) + 1,
                ts=time.time(),
                kind="error",
                name=self.name,
                input_summary=redact_secrets(f"Task: {task.id}"),
                output_summary=redact_secrets(f"Exception: {exc}"),
                latency_ms=elapsed_ms,
                error=str(exc),
            )
            ctx.add_step(err_step)
            return AgentResult(verdict="uncertain", error=str(exc)), ctx.steps


class HttpAdapter:
    """
    Adapter that executes an external agent via HTTP POST:
    POST {"task": {...}, "run_id": "..."}
    Expects {"result": {...}, "trace": [optional steps]}
    If no trace is returned, records black-box metrics.
    """

    def __init__(self, endpoint_url: str, timeout_seconds: float = 90.0) -> None:
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds

    def run(self, task: Task, ctx: RunContext) -> tuple[AgentResult, list[StepRecord]]:
        t0 = time.perf_counter()
        payload = json.dumps(
            {"task": task.to_dict(), "run_id": ctx.run_id},
            ensure_ascii=False,
        ).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "AgentLab-HttpAdapter/1.0",
                "x-bf-run-id": ctx.run_id,
            },
            method="POST",
        )

        error_msg = None
        data: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.getcode()
                body = resp.read().decode("utf-8")
                if status_code >= 400:
                    raise RuntimeError(f"HTTP {status_code}: {body[:200]}")
                data = json.loads(body)
        except Exception as exc:
            error_msg = str(exc)
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if error_msg is not None:
            ctx.is_black_box = True
            err_step = StepRecord(
                idx=1,
                ts=time.time(),
                kind="error",
                name="http_agent",
                input_summary=redact_secrets(f"POST {self.endpoint_url}"),
                output_summary=redact_secrets(f"Error: {error_msg}"),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
            return AgentResult(verdict="uncertain", error=error_msg), [err_step]

        # Extract AgentResult
        res_dict = data.get("result")
        if not isinstance(res_dict, dict):
            res_dict = data
        result = AgentResult.from_dict(res_dict)

        # Extract trace steps
        trace_list = data.get("trace")
        if isinstance(trace_list, list) and trace_list:
            steps = [StepRecord.from_dict(s) for s in trace_list if isinstance(s, dict)]
            return result, steps

        # Black-box run: no internal trace provided
        ctx.is_black_box = True
        bb_step = StepRecord(
            idx=1,
            ts=time.time(),
            kind="tool",
            name="http_agent",
            input_summary=redact_secrets(f"POST {self.endpoint_url} (task_id={task.id})"),
            output_summary=redact_secrets(f"HTTP 200: verdict={result.verdict}"),
            latency_ms=elapsed_ms,
            error=result.error,
        )
        return result, [bb_step]


class CliAdapter:
    """
    Adapter that executes an external agent via subprocess CLI.
    Sends JSON task on stdin, reads JSON result from stdout.
    If no trace is returned, records black-box metrics.
    """

    def __init__(self, command: list[str] | str, timeout_seconds: float = 90.0) -> None:
        self.command = command if isinstance(command, list) else command.split()
        self.timeout_seconds = timeout_seconds

    def run(self, task: Task, ctx: RunContext) -> tuple[AgentResult, list[StepRecord]]:
        t0 = time.perf_counter()
        payload = json.dumps(
            {"task": task.to_dict(), "run_id": ctx.run_id},
            ensure_ascii=False,
        )

        error_msg = None
        data: dict[str, Any] = {}
        try:
            proc = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"CLI exited with {proc.returncode}: {proc.stderr[:200]}"
                )
            data = json.loads(proc.stdout)
        except Exception as exc:
            error_msg = str(exc)
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        if error_msg is not None:
            ctx.is_black_box = True
            err_step = StepRecord(
                idx=1,
                ts=time.time(),
                kind="error",
                name="cli_agent",
                input_summary=redact_secrets(f"CLI: {' '.join(self.command)}"),
                output_summary=redact_secrets(f"Error: {error_msg}"),
                latency_ms=elapsed_ms,
                error=error_msg,
            )
            return AgentResult(verdict="uncertain", error=error_msg), [err_step]

        res_dict = data.get("result")
        if not isinstance(res_dict, dict):
            res_dict = data
        result = AgentResult.from_dict(res_dict)

        trace_list = data.get("trace")
        if isinstance(trace_list, list) and trace_list:
            steps = [StepRecord.from_dict(s) for s in trace_list if isinstance(s, dict)]
            return result, steps

        # Black-box run: no internal trace provided
        ctx.is_black_box = True
        bb_step = StepRecord(
            idx=1,
            ts=time.time(),
            kind="tool",
            name="cli_agent",
            input_summary=redact_secrets(f"CLI: {' '.join(self.command)} (task_id={task.id})"),
            output_summary=redact_secrets(f"Exit 0: verdict={result.verdict}"),
            latency_ms=elapsed_ms,
            error=result.error,
        )
        return result, [bb_step]
