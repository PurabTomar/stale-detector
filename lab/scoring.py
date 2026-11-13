"""
lab/scoring.py - Deterministic, non-LLM scoring engine for Agent Lab.

Calculates:
1. Success: verdict == expected_status AND at least one expected_fact in current_value.
2. Run totals: steps count, tool_calls count, token sums, latency, and estimated cost.
"""

import json
from pathlib import Path
from typing import Any

from lab.schema import AgentResult, StepRecord

_PRICING_FILE = Path(__file__).resolve().parent / "pricing.json"


def load_pricing_table(path: Path | str | None = None) -> dict[str, Any]:
    """Load model token pricing table from JSON file."""
    p = Path(path) if path else _PRICING_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def evaluate_success(
    verdict: str,
    expected_status: str,
    current_value: str | None,
    expected_facts: list[str],
) -> bool:
    """
    Deterministic evaluation:
    - verdict must equal expected_status (case-insensitive).
    - at least one expected_fact must appear in current_value (case-insensitive).
      (If expected_facts is empty, verdict match alone is sufficient).
    """
    clean_verdict = (verdict or "").strip().lower()
    clean_expected = (expected_status or "").strip().lower()
    if clean_verdict != clean_expected:
        return False

    if not expected_facts:
        return True

    val = (current_value or "").lower()
    for fact in expected_facts:
        f_clean = str(fact).strip().lower()
        if f_clean and f_clean in val:
            return True

    return False


def estimate_run_cost(
    steps: list[StepRecord],
    pricing: dict[str, Any] | None = None,
) -> tuple[float | None, str]:
    """
    Compute estimated cost in USD based on tokens_in and tokens_out per LLM step.
    If any model has null pricing or is missing, returns (None, 'cost not set').
    """
    if pricing is None:
        pricing = load_pricing_table()

    total_cost = 0.0
    has_llm_steps = False

    for s in steps:
        if s.kind != "llm":
            continue
        has_llm_steps = True
        model = (s.model or "").strip()
        model_rate = pricing.get(model)
        if not model_rate or not isinstance(model_rate, dict):
            # Fall back to default rate if explicitly configured
            model_rate = pricing.get("default")

        if not model_rate or not isinstance(model_rate, dict):
            return None, "cost not set"

        in_rate = model_rate.get("input_per_million")
        out_rate = model_rate.get("output_per_million")
        if in_rate is None or out_rate is None:
            return None, "cost not set"

        step_cost = (
            (float(s.tokens_in) * float(in_rate))
            + (float(s.tokens_out) * float(out_rate))
        ) / 1_000_000.0
        total_cost += step_cost

    if not has_llm_steps:
        return 0.0, "$0.000000"

    rounded = round(total_cost, 6)
    return rounded, f"${rounded:.6f}"


def compute_run_totals(
    steps: list[StepRecord],
    duration_ms: float,
    result: AgentResult,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute aggregated metrics and cost for a RunRecord."""
    tool_calls = sum(1 for s in steps if s.kind == "tool")
    llm_calls = sum(1 for s in steps if s.kind == "llm")
    fallback_calls = sum(1 for s in steps if s.kind == "fallback")
    tokens_in = sum(s.tokens_in for s in steps)
    tokens_out = sum(s.tokens_out for s in steps)
    total_tokens = tokens_in + tokens_out

    step_latencies = sum(s.latency_ms for s in steps)
    total_latency_ms = round(max(float(duration_ms), float(step_latencies)), 2)

    cost_val, cost_display = estimate_run_cost(steps, pricing=pricing)

    return {
        "steps": len(steps),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "fallback_calls": fallback_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency_ms,
        "estimated_cost": cost_val,
        "estimated_cost_display": cost_display,
    }
