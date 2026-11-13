"""
lab/runner.py - Orchestrator for running benchmark suites across agents.

Features:
- Resumable: skips runs already persisted in lab/results/runs.jsonl.
- Budget guards: stops gracefully when Tavily credits or LLM calls limit is hit.
- Exponential backoff retry on HTTP 429/503 rate-limits or transient outages.
- Configurable concurrency with thread safety.
- Dry-run mode estimating resource usage.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from lab.adapters import AgentAdapter
from lab.agents import get_agent, list_agents
from lab.schema import AgentResult, RunContext, RunRecord, StepRecord, Task
from lab.scoring import compute_run_totals, evaluate_success, load_pricing_table

_DEFAULT_SUITE = Path(__file__).resolve().parent / "tasks" / "research_freshness.json"
_DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "runs.jsonl"


class BudgetExceededError(Exception):
    """Raised when configured budget limit has been reached."""
    pass


class BudgetTracker:
    """Thread-safe resource budget accumulator."""

    def __init__(
        self,
        max_tavily_credits: float | None = None,
        max_llm_calls: int | None = None,
    ) -> None:
        self.max_tavily_credits = max_tavily_credits
        self.max_llm_calls = max_llm_calls
        self.credits_used = 0.0
        self.llm_calls = 0
        self._lock = threading.Lock()
        self.exhausted = False
        self.exhausted_reason = ""

    def add(self, credits: float, llm_calls: int) -> None:
        with self._lock:
            self.credits_used += credits
            self.llm_calls += llm_calls
            if self.max_tavily_credits is not None and self.credits_used >= self.max_tavily_credits:
                self.exhausted = True
                self.exhausted_reason = f"Tavily credits budget ({self.max_tavily_credits}) reached."
            if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
                self.exhausted = True
                self.exhausted_reason = f"LLM calls budget ({self.max_llm_calls}) reached."

    def is_exhausted(self) -> bool:
        with self._lock:
            return self.exhausted


def load_tasks(suite_path: Path | str) -> list[Task]:
    """Load and validate benchmark tasks from JSON file."""
    p = Path(suite_path)
    if not p.is_file():
        raise FileNotFoundError(f"Task suite file not found: {p}")
    raw_data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw_data, list):
        raise ValueError(f"Task suite JSON must be a list of task objects: {p}")
    return [Task.from_dict(item) for item in raw_data if isinstance(item, dict)]


def load_completed_runs(results_path: Path | str) -> set[tuple[str, str, int, str]]:
    """
    Read results JSONL and return set of completed run signatures:
    (agent_name, task_id, repeat, scenario).
    """
    completed: set[tuple[str, str, int, str]] = set()
    p = Path(results_path)
    if not p.is_file():
        return completed

    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record_dict = json.loads(line_str)
                    sig = (
                        str(record_dict.get("agent")),
                        str(record_dict.get("task_id")),
                        int(record_dict.get("repeat", 1)),
                        str(record_dict.get("scenario", "clean")),
                    )
                    completed.add(sig)
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        pass

    return completed


_FILE_LOCK = threading.Lock()


def append_run_record(record: RunRecord, results_path: Path | str) -> None:
    """Thread-safe append of a RunRecord to runs.jsonl."""
    p = Path(results_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _FILE_LOCK:
        with open(p, "a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")


def execute_single_run(
    task: Task,
    agent_name: str,
    adapter: AgentAdapter,
    repeat: int,
    scenario: str,
    timeout_seconds: float,
    bypass_cache: bool,
    budget_tracker: BudgetTracker,
    pricing: dict[str, Any],
) -> RunRecord | None:
    """Execute a single run of an agent on a task with retries on rate limits."""
    if budget_tracker.is_exhausted():
        return None

    run_id = f"run-{agent_name}-{task.id}-r{repeat}-{uuid.uuid4().hex[:6]}"
    ctx = RunContext(
        run_id=run_id,
        agent_name=agent_name,
        repeat=repeat,
        scenario=scenario,
        bypass_cache=bypass_cache,
        timeout_seconds=timeout_seconds,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    # Retry loop with exponential backoff on transient 429/503
    max_retries = 3
    backoff_delay = 1.0
    result: AgentResult
    steps: list[StepRecord]

    for attempt in range(max_retries + 1):
        if budget_tracker.is_exhausted():
            return None
        try:
            result, steps = adapter.run(task, ctx)
            if result and result.error:
                is_transient = any(code in result.error for code in ["429", "503", "rate_limit", "resource_exhausted"])
                if is_transient and attempt < max_retries:
                    time.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue
            break
        except Exception as exc:
            err_str = str(exc)
            is_transient = any(code in err_str for code in ["429", "503", "rate_limit", "resource_exhausted"])
            if is_transient and attempt < max_retries:
                time.sleep(backoff_delay)
                backoff_delay *= 2.0
                continue
            # Non-retryable error
            result = AgentResult(verdict="uncertain", error=err_str)
            steps = ctx.steps
            break

    ended_at = datetime.now(timezone.utc).isoformat()
    duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Deterministic scoring
    success = evaluate_success(
        verdict=result.verdict,
        expected_status=task.expected_status,
        current_value=result.current_value,
        expected_facts=task.expected_facts,
    )

    totals = compute_run_totals(steps, duration_ms, result, pricing=pricing)

    # Update budget tracking
    tool_calls = totals.get("tool_calls", 0)
    llm_calls = totals.get("llm_calls", 0)
    budget_tracker.add(credits=float(tool_calls), llm_calls=int(llm_calls))

    return RunRecord(
        run_id=run_id,
        agent=agent_name,
        task_id=task.id,
        repeat=repeat,
        scenario=scenario,
        success=success,
        result=result,
        steps=steps,
        totals=totals,
        started_at=started_at,
        ended_at=ended_at,
        is_black_box=ctx.is_black_box,
    )


def compute_dry_run_estimates(
    agents: list[str],
    tasks: list[Task],
    repeats: int,
) -> dict[str, Any]:
    """Estimate Tavily credits and LLM calls for a planned benchmark run."""
    estimates_per_agent = {
        "llm_only": {"llm_calls": 1.0, "tavily_credits": 0.0},
        "single_search": {"llm_calls": 1.0, "tavily_credits": 1.0},
        "react_loop": {"llm_calls": 3.0, "tavily_credits": 2.0},
        "detector_pipeline": {"llm_calls": 2.0, "tavily_credits": 2.0},
        "external_http": {"llm_calls": 2.0, "tavily_credits": 2.0},
    }

    total_runs = len(agents) * len(tasks) * repeats
    est_credits = 0.0
    est_llm_calls = 0

    agent_breakdowns = {}
    for a in agents:
        rate = estimates_per_agent.get(a, {"llm_calls": 2.0, "tavily_credits": 1.0})
        agent_runs = len(tasks) * repeats
        c = agent_runs * rate["tavily_credits"]
        l = int(agent_runs * rate["llm_calls"])
        est_credits += c
        est_llm_calls += l
        agent_breakdowns[a] = {
            "runs": agent_runs,
            "est_tavily_credits": c,
            "est_llm_calls": l,
        }

    return {
        "total_agents": len(agents),
        "total_tasks": len(tasks),
        "repeats": repeats,
        "total_runs": total_runs,
        "est_tavily_credits": est_credits,
        "est_llm_calls": est_llm_calls,
        "per_agent": agent_breakdowns,
    }


def run_benchmark(
    agents: list[str] | None = None,
    suite_path: Path | str | None = None,
    results_path: Path | str | None = None,
    repeats: int = 3,
    scenarios: list[str] | None = None,
    concurrency: int = 2,
    timeout_per_run: float = 90.0,
    budget_tavily_credits: float | None = None,
    budget_llm_calls: int | None = None,
    dry_run: bool = False,
    bypass_cache: bool = True,
    pricing_path: Path | str | None = None,
    progress_callback: Any = None,
) -> list[RunRecord]:
    """
    Run Agent Lab benchmarking across specified agents and tasks.
    Supports dry-run, concurrency, resume, and budget limit stops.
    """
    suite_p = Path(suite_path) if suite_path else _DEFAULT_SUITE
    tasks = load_tasks(suite_p)

    agent_names = agents or ["llm_only", "single_search", "react_loop", "detector_pipeline"]
    scenario_list = scenarios or ["clean"]
    results_p = Path(results_path) if results_path else _DEFAULT_RESULTS
    pricing = load_pricing_table(pricing_path)

    # 1. Dry Run Mode
    if dry_run:
        est = compute_dry_run_estimates(agent_names, tasks, repeats)
        print("=" * 60)
        print("AGENT LAB BENCHMARK DRY RUN ESTIMATE")
        print("=" * 60)
        print(f"Tasks: {est['total_tasks']} | Agents: {est['total_agents']} | Repeats: {repeats}")
        print(f"Total planned runs: {est['total_runs']}")
        print(f"Estimated Tavily credits: {est['est_tavily_credits']:.1f}")
        print(f"Estimated LLM calls:     {est['est_llm_calls']}")
        print("\nBreakdown per agent:")
        for a_name, a_info in est["per_agent"].items():
            print(f"  - {a_name:18}: {a_info['runs']:3} runs | ~{a_info['est_tavily_credits']:4.1f} credits | ~{a_info['est_llm_calls']:4} LLM calls")
        print("=" * 60)
        return []

    # 2. Resumability: check completed runs in results file
    completed = load_completed_runs(results_p)
    budget_tracker = BudgetTracker(
        max_tavily_credits=budget_tavily_credits,
        max_llm_calls=budget_llm_calls,
    )

    # Build queue of (task, agent_name, repeat, scenario)
    planned_runs = []
    skipped_count = 0
    for scenario in scenario_list:
        for repeat in range(1, repeats + 1):
            for agent_name in agent_names:
                for task in tasks:
                    sig = (agent_name, task.id, repeat, scenario)
                    if sig in completed:
                        skipped_count += 1
                    else:
                        planned_runs.append((task, agent_name, repeat, scenario))

    if skipped_count > 0:
        print(f"[Agent Lab] Resuming: skipped {skipped_count} already completed run(s).")
    print(f"[Agent Lab] Executing {len(planned_runs)} run(s) with concurrency={concurrency}...")

    # Instantiate agent adapters
    adapters: dict[str, AgentAdapter] = {}
    for a_name in agent_names:
        try:
            adapters[a_name] = get_agent(a_name)
        except Exception as exc:
            print(f"[Agent Lab Warning] Could not initialize agent '{a_name}': {exc}")

    records: list[RunRecord] = []

    def _worker(item: tuple[Task, str, int, str]) -> RunRecord | None:
        t_obj, a_nm, rep, scen = item
        adp = adapters.get(a_nm)
        if adp is None:
            return None
        return execute_single_run(
            task=t_obj,
            agent_name=a_nm,
            adapter=adp,
            repeat=rep,
            scenario=scen,
            timeout_seconds=timeout_per_run,
            bypass_cache=bypass_cache,
            budget_tracker=budget_tracker,
            pricing=pricing,
        )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {executor.submit(_worker, item): item for item in planned_runs}
        for future in as_completed(futures):
            if budget_tracker.is_exhausted():
                print(f"[Agent Lab Notice] Budget limit reached: {budget_tracker.exhausted_reason}")
                break
            try:
                rec = future.result()
                if rec is not None:
                    records.append(rec)
                    append_run_record(rec, results_p)
                    if progress_callback:
                        progress_callback(rec)
            except Exception as e:
                print(f"[Agent Lab Run Error] {e}")

    return records
