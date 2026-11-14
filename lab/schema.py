"""
lab/schema.py - Core data schemas for Agent Lab benchmarking.

Defines schemas for Tasks, AgentResults, StepRecords (telemetry traces),
RunRecords (full benchmark run execution), and RunContext.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import time
from typing import Any, Literal

StepKind = Literal["llm", "tool", "decision", "error", "fallback"]

VALID_STATUSES = {
    "fresh",
    "outdated",
    "incorrect",
    "partially_correct",
    "uncertain",
    "unverifiable",
}


@dataclass
class Task:
    """A research / fact-verification task with ground truth expectations."""

    id: str
    category: str
    input_text: str
    expected_status: str
    expected_facts: list[str] = field(default_factory=list)
    needs_human_check: bool = True
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.expected_status = self.expected_status.strip().lower()
        if self.expected_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid expected_status '{self.expected_status}'. "
                f"Must be one of {sorted(VALID_STATUSES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "other")),
            input_text=str(data.get("input_text", "")),
            expected_status=str(data.get("expected_status", "uncertain")),
            expected_facts=[str(f) for f in data.get("expected_facts", [])],
            needs_human_check=bool(data.get("needs_human_check", True)),
            tags=[str(t) for t in data.get("tags", [])],
        )


@dataclass
class AgentResult:
    """The normalized verdict and answer produced by any research agent."""

    verdict: str
    current_value: str | None = None
    answer: str = ""
    error: str | None = None

    def __post_init__(self) -> None:
        if self.verdict:
            self.verdict = self.verdict.strip().lower()
            if self.verdict not in VALID_STATUSES:
                # If agent returns an unmapped or noisy status, normalize defensively
                self.verdict = "uncertain"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult":
        return cls(
            verdict=str(data.get("verdict", "uncertain")),
            current_value=data.get("current_value"),
            answer=str(data.get("answer", "")),
            error=data.get("error"),
        )


@dataclass
class StepRecord:
    """Telemetry record for an individual agent step (LLM call, tool call, fallback, etc.)."""

    idx: int
    ts: float = field(default_factory=time.time)
    kind: StepKind = "tool"
    name: str = ""
    input_summary: str = ""
    output_summary: str = ""
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    error: str | None = None
    gateway_tokens: int | None = None
    token_mismatch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        return cls(
            idx=int(data.get("idx", 0)),
            ts=float(data.get("ts", time.time())),
            kind=data.get("kind", "tool"),
            name=str(data.get("name", "")),
            input_summary=str(data.get("input_summary", "")),
            output_summary=str(data.get("output_summary", "")),
            model=data.get("model"),
            tokens_in=int(data.get("tokens_in", 0) or 0),
            tokens_out=int(data.get("tokens_out", 0) or 0),
            latency_ms=float(data.get("latency_ms", 0.0) or 0.0),
            cache_hit=bool(data.get("cache_hit", False)),
            error=data.get("error"),
            gateway_tokens=data.get("gateway_tokens"),
            token_mismatch=data.get("token_mismatch"),
        )


@dataclass
class RunRecord:
    """Complete audit record of a single benchmark run on a task."""

    run_id: str
    agent: str
    task_id: str
    repeat: int = 1
    scenario: str = "clean"
    success: bool = False
    result: AgentResult = field(default_factory=lambda: AgentResult(verdict="uncertain"))
    steps: list[StepRecord] = field(default_factory=list)
    totals: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    is_black_box: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "task_id": self.task_id,
            "repeat": self.repeat,
            "scenario": self.scenario,
            "success": self.success,
            "result": self.result.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "totals": self.totals,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "is_black_box": self.is_black_box,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        res_data = data.get("result", {})
        result_obj = (
            AgentResult.from_dict(res_data)
            if isinstance(res_data, dict)
            else AgentResult(verdict=str(res_data or "uncertain"))
        )
        steps_objs = [
            StepRecord.from_dict(s)
            for s in data.get("steps", [])
            if isinstance(s, dict)
        ]
        return cls(
            run_id=str(data.get("run_id", "")),
            agent=str(data.get("agent", "")),
            task_id=str(data.get("task_id", "")),
            repeat=int(data.get("repeat", 1)),
            scenario=str(data.get("scenario", "clean")),
            success=bool(data.get("success", False)),
            result=result_obj,
            steps=steps_objs,
            totals=dict(data.get("totals", {})),
            started_at=str(data.get("started_at", "")),
            ended_at=str(data.get("ended_at", "")),
            is_black_box=bool(data.get("is_black_box", False)),
        )


@dataclass
class RunContext:
    """Execution context provided to agents and adapters for trace recording."""

    run_id: str
    agent_name: str
    repeat: int = 1
    scenario: str = "clean"
    bypass_cache: bool = True
    timeout_seconds: float = 90.0
    steps: list[StepRecord] = field(default_factory=list)
    is_black_box: bool = False

    def add_step(self, step: StepRecord) -> None:
        step.idx = len(self.steps) + 1
        self.steps.append(step)
