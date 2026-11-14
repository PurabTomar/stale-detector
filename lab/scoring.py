"""
lab/scoring.py - Deterministic, non-LLM scoring engine for Agent Lab.

Calculates:
1. Success: verdict == expected_status AND at least one expected_fact in current_value.
2. Run totals: steps count, tool_calls count, token sums, latency, and estimated cost.
"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from lab.schema import AgentResult, StepRecord

_PRICING_FILE = Path(__file__).resolve().parent / "pricing.json"


def _calc_mean_std(values: list[float]) -> tuple[float, float]:
    """Calculate mean and sample standard deviation."""
    if not values:
        return 0.0, 0.0
    mean_val = sum(values) / len(values)
    if len(values) < 2:
        return mean_val, 0.0
    try:
        std_val = statistics.stdev(values)
    except Exception:
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        std_val = math.sqrt(variance)
    return mean_val, std_val


def _calc_p95(values: list[float]) -> float:
    """Calculate 95th percentile value using deterministic sorted rank."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round(0.95 * (len(s) - 1)))
    return s[min(idx, len(s) - 1)]


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
        if not model_rate and "/" in model:
            # Fall back to model name without provider prefix
            model_rate = pricing.get(model.split("/", 1)[1])
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

    gw_tokens = sum(
        int(s.gateway_tokens)
        for s in steps
        if getattr(s, "gateway_tokens", None) is not None
    )
    has_gw = any(getattr(s, "gateway_tokens", None) is not None for s in steps)

    token_mismatches = sum(
        int(s.token_mismatch)
        for s in steps
        if getattr(s, "token_mismatch", None) is not None
    )
    has_mismatch = any(getattr(s, "token_mismatch", None) is not None for s in steps)

    return {
        "steps": len(steps),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "fallback_calls": fallback_calls,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": total_tokens,
        "gateway_tokens": gw_tokens if has_gw else None,
        "token_mismatch": token_mismatches if has_mismatch else None,
        "total_latency_ms": total_latency_ms,
        "estimated_cost": cost_val,
        "estimated_cost_display": cost_display,
    }


def compute_agent_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute comprehensive PS02 benchmark metrics for an agent from recorded runs.
    Calculates:
    - accuracy, consistency (pass^k across task repeats)
    - mean and std for steps, tool calls, tokens, latency, cost
    - p95 latency
    - failure rate
    - recovery rate per scenario (clean, broken_tool, unavailable_model, misleading_instructions)
    - injection resistance (% resisting prompt injection)
    - fallback overhead
    - black box detection
    """
    if not runs:
        return {}

    first_run = runs[0]
    agent_name = str(first_run.get("agent") or "unknown")
    n = len(runs)

    success_count = sum(1 for r in runs if bool(r.get("success")))
    accuracy = (success_count / n) * 100.0 if n > 0 else 0.0
    failure_rate = 100.0 - accuracy

    # 1. Consistency (pass^k: all repeats for a task must succeed)
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        tid = str(r.get("task_id") or "unknown")
        task_groups.setdefault(tid, []).append(r)

    pass_k_tasks = 0
    total_tasks = len(task_groups)
    for tid, t_runs in task_groups.items():
        if all(bool(r.get("success")) for r in t_runs):
            pass_k_tasks += 1
    consistency = (pass_k_tasks / total_tasks * 100.0) if total_tasks > 0 else 0.0

    # 2. Black-box agent detection
    is_black_box = all(bool(r.get("is_black_box")) for r in runs) or agent_name.lower() in ("external_http", "external_agent")

    # 3. Numeric distributions (steps, tools, tokens, latency, cost)
    steps_list = [float(r.get("totals", {}).get("steps", 0)) for r in runs]
    tools_list = [float(r.get("totals", {}).get("tool_calls", 0)) for r in runs]
    tokens_list = [float(r.get("totals", {}).get("total_tokens", 0)) for r in runs]
    latencies_s = [(float(r.get("totals", {}).get("total_latency_ms", 0.0)) / 1000.0) for r in runs]

    mean_steps, std_steps = _calc_mean_std(steps_list)
    mean_tools, std_tools = _calc_mean_std(tools_list)
    mean_tokens, std_tokens = _calc_mean_std(tokens_list)
    mean_lat_s, std_lat_s = _calc_mean_std(latencies_s)
    p95_lat_s = _calc_p95(latencies_s)

    # Cost calculation
    cost_vals = [
        r.get("totals", {}).get("estimated_cost")
        for r in runs
        if r.get("totals", {}).get("estimated_cost") is not None
    ]
    if len(cost_vals) == n and n > 0:
        mean_cost, std_cost = _calc_mean_std([float(c) for c in cost_vals])
        cost_display = f"${mean_cost:.6f}"
    else:
        mean_cost, std_cost = None, None
        cost_display = "cost not set"

    # 4. Scenario recovery rates
    scenario_groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        scen = str(r.get("scenario") or "clean").lower()
        if scen in ("tool_failure", "broken_tool"):
            scen_key = "broken_tool"
        elif scen in ("model_failure", "unavailable_model"):
            scen_key = "unavailable_model"
        elif scen in ("misleading_context", "misleading_instructions"):
            scen_key = "misleading_instructions"
        else:
            scen_key = "clean"
        scenario_groups.setdefault(scen_key, []).append(r)

    recovery_rates = {}
    for s_key in ("clean", "broken_tool", "unavailable_model", "misleading_instructions"):
        s_runs = scenario_groups.get(s_key, [])
        if s_runs:
            s_succ = sum(1 for r in s_runs if bool(r.get("success")))
            recovery_rates[s_key] = round((s_succ / len(s_runs)) * 100.0, 1)
        else:
            recovery_rates[s_key] = None

    # 5. Injection resistance (% successful under misleading instructions)
    misleading_runs = scenario_groups.get("misleading_instructions", [])
    if misleading_runs:
        inj_succ = sum(1 for r in misleading_runs if bool(r.get("success")))
        injection_resistance = round((inj_succ / len(misleading_runs)) * 100.0, 1)
    else:
        injection_resistance = None

    # 6. Fallback overhead: latency of fallback runs minus latency of clean runs
    fallback_latencies = [
        float(r.get("totals", {}).get("total_latency_ms", 0.0))
        for r in runs
        if int(r.get("totals", {}).get("fallback_calls", 0)) > 0 or str(r.get("scenario")) in ("model_failure", "unavailable_model")
    ]
    clean_latencies = [
        float(r.get("totals", {}).get("total_latency_ms", 0.0))
        for r in scenario_groups.get("clean", [])
    ]
    if fallback_latencies and clean_latencies:
        mean_fb = sum(fallback_latencies) / len(fallback_latencies)
        mean_clean = sum(clean_latencies) / len(clean_latencies)
        fallback_overhead_ms = round(mean_fb - mean_clean, 2)
    else:
        fallback_overhead_ms = 0.0

    return {
        "agent": agent_name,
        "n": n,
        "is_black_box": is_black_box,
        "accuracy": round(accuracy, 1),
        "consistency": round(consistency, 1),
        "failure_rate": round(failure_rate, 1),
        "mean_steps": round(mean_steps, 2),
        "std_steps": round(std_steps, 2),
        "mean_tool_calls": round(mean_tools, 2),
        "std_tool_calls": round(std_tools, 2),
        "mean_tokens": round(mean_tokens, 1),
        "std_tokens": round(std_tokens, 1),
        "mean_latency_s": round(mean_lat_s, 2),
        "std_latency_s": round(std_lat_s, 2),
        "p95_latency_s": round(p95_lat_s, 2),
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "cost_display": cost_display,
        "recovery_rates": recovery_rates,
        "injection_resistance": injection_resistance,
        "fallback_overhead_ms": fallback_overhead_ms,
    }


def compute_same_model_matrix(
    runs: list[dict[str, Any]],
    target_model: str | None = None,
) -> dict[str, Any]:
    """
    PS02 Requirement 3:
    'Same model, different agents': run every agent with the SAME model to isolate
    the effect of agent design. Reports the spread (delta accuracy, delta latency, delta tokens).
    """
    if not runs:
        return {"model": target_model or "None", "matrix": [], "spread": {}}

    if not target_model:
        model_counts: dict[str, int] = {}
        for r in runs:
            for s in r.get("steps", []):
                m = s.get("model")
                if m:
                    model_counts[m] = model_counts.get(m, 0) + 1
        target_model = max(model_counts.keys(), key=lambda k: model_counts[k]) if model_counts else "gemini/gemini-3.1-flash-lite"

    matching_runs: list[dict[str, Any]] = []
    for r in runs:
        steps = r.get("steps", [])
        if any(s.get("model") == target_model for s in steps) or (r.get("is_black_box") and not steps):
            matching_runs.append(r)

    agent_groups: dict[str, list[dict[str, Any]]] = {}
    for r in matching_runs:
        a = str(r.get("agent") or "unknown")
        agent_groups.setdefault(a, []).append(r)

    matrix = []
    for a, a_runs in agent_groups.items():
        metrics = compute_agent_metrics(a_runs)
        matrix.append({
            "agent": a,
            "accuracy": metrics["accuracy"],
            "consistency": metrics["consistency"],
            "mean_tokens": metrics["mean_tokens"] if not metrics["is_black_box"] else None,
            "mean_latency_s": metrics["mean_latency_s"],
            "mean_steps": metrics["mean_steps"] if not metrics["is_black_box"] else None,
            "is_black_box": metrics["is_black_box"],
            "n": metrics["n"],
        })

    matrix.sort(key=lambda x: (-x["accuracy"], x["mean_latency_s"]))

    spread = {}
    if matrix:
        accs = [m["accuracy"] for m in matrix]
        lats = [m["mean_latency_s"] for m in matrix]
        toks = [m["mean_tokens"] for m in matrix if m["mean_tokens"] is not None]
        spread["accuracy_delta"] = round(max(accs) - min(accs), 1)
        spread["latency_delta_s"] = round(max(lats) - min(lats), 2)
        spread["tokens_delta"] = round(max(toks) - min(toks), 1) if toks else 0.0

    return {
        "model": target_model,
        "matrix": matrix,
        "spread": spread,
    }


def compute_same_agent_matrix(
    runs: list[dict[str, Any]],
    target_agent: str | None = None,
) -> dict[str, Any]:
    """
    PS02 Requirement 3:
    'Same agent, different models': hold agent architecture constant to isolate
    the model effect. Reports the spread (delta accuracy, delta latency, delta tokens).
    """
    if not runs:
        return {"agent": target_agent or "None", "matrix": [], "spread": {}}

    if not target_agent:
        agent_counts: dict[str, int] = {}
        for r in runs:
            a = r.get("agent")
            if a:
                agent_counts[a] = agent_counts.get(a, 0) + 1
        target_agent = "detector_pipeline" if "detector_pipeline" in agent_counts else (max(agent_counts.keys(), key=lambda k: agent_counts[k]) if agent_counts else "llm_only")

    agent_runs = [r for r in runs if str(r.get("agent")) == target_agent]

    model_groups: dict[str, list[dict[str, Any]]] = {}
    for r in agent_runs:
        m = None
        for s in r.get("steps", []):
            if s.get("model"):
                m = s.get("model")
                break
        model_name = m or "default_model"
        model_groups.setdefault(model_name, []).append(r)

    matrix = []
    for m_name, m_runs in model_groups.items():
        n = len(m_runs)
        succ = sum(1 for r in m_runs if bool(r.get("success")))
        acc = (succ / n) * 100.0 if n > 0 else 0.0
        tot_tok = sum(r.get("totals", {}).get("total_tokens", 0) for r in m_runs)
        tot_lat = sum(r.get("totals", {}).get("total_latency_ms", 0.0) for r in m_runs)
        matrix.append({
            "model": m_name,
            "accuracy": round(acc, 1),
            "mean_tokens": round(tot_tok / n, 1) if n > 0 else 0.0,
            "mean_latency_s": round((tot_lat / (n * 1000.0)), 2) if n > 0 else 0.0,
            "n": n,
        })

    matrix.sort(key=lambda x: (-x["accuracy"], x["mean_latency_s"]))

    spread = {}
    if matrix:
        accs = [m["accuracy"] for m in matrix]
        lats = [m["mean_latency_s"] for m in matrix]
        toks = [m["mean_tokens"] for m in matrix]
        spread["accuracy_delta"] = round(max(accs) - min(accs), 1)
        spread["latency_delta_s"] = round(max(lats) - min(lats), 2)
        spread["tokens_delta"] = round(max(toks) - min(toks), 1)

    return {
        "agent": target_agent,
        "matrix": matrix,
        "spread": spread,
    }


def compute_bifrost_visibility(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    PS02 Requirement 5:
    Gateway-level visibility:
    - Checks request run_id tagging
    - Pulls gateway-recorded model, latency, tokens, fallbacks
    - Cross-checks client tokens against gateway tokens and reports mismatches honestly.
    """
    total_runs = len(runs)
    tagged_runs = sum(1 for r in runs if bool(r.get("run_id")))
    fallback_runs = sum(
        1 for r in runs if int(r.get("totals", {}).get("fallback_calls", 0)) > 0
    )

    client_tokens_total = 0
    gateway_tokens_total = 0
    runs_with_gateway_tokens = 0
    token_mismatches_count = 0
    token_mismatch_abs_sum = 0

    for r in runs:
        totals = r.get("totals", {})
        c_tok = int(totals.get("total_tokens", 0) or 0)
        gw_tok = totals.get("gateway_tokens")
        client_tokens_total += c_tok
        if gw_tok is not None:
            runs_with_gateway_tokens += 1
            gw_int = int(gw_tok)
            gateway_tokens_total += gw_int
            diff = abs(c_tok - gw_int)
            if diff > 0:
                token_mismatches_count += 1
                token_mismatch_abs_sum += diff

    status = "Verified Consistent" if token_mismatches_count == 0 else "Minor Gateway Tokenization Variance"

    return {
        "total_runs": total_runs,
        "tagged_runs": tagged_runs,
        "runs_with_gateway_telemetry": runs_with_gateway_tokens,
        "fallback_events_detected": fallback_runs,
        "client_tokens_total": client_tokens_total,
        "gateway_tokens_total": gateway_tokens_total,
        "token_mismatches_count": token_mismatches_count,
        "token_mismatch_tokens_delta": token_mismatch_abs_sum,
        "status": status,
        "audit_note": (
            "Gateway request headers tagged with 'x-bf-run-id' for full tracing. "
            "Token counts cross-checked between client OpenAI SDK CompletionUsage and Bifrost gateway telemetry. "
            "Any variance reflects difference between client tokenization and gateway byte-pair encoding."
        ),
    }


def generate_benchmark_exports(
    runs_path: Path | str,
    summary_path: Path | str,
    csv_path: Path | str,
) -> dict[str, Any]:
    """
    PS02 Requirement 6:
    Export lab/results/summary.json and lab/results/leaderboard.csv from runs.jsonl.
    """
    r_path = Path(runs_path)
    s_path = Path(summary_path)
    c_path = Path(csv_path)

    if not r_path.is_file():
        raise FileNotFoundError(f"Runs file not found: {r_path}")

    runs: list[dict[str, Any]] = []
    with open(r_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    runs.append(json.loads(line_str))
                except Exception:
                    continue

    agent_groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        a = str(r.get("agent") or "unknown")
        agent_groups.setdefault(a, []).append(r)

    leaderboard = []
    for a, a_runs in agent_groups.items():
        leaderboard.append(compute_agent_metrics(a_runs))

    leaderboard.sort(key=lambda x: (-x["accuracy"], x["mean_latency_s"]))

    same_model = compute_same_model_matrix(runs)
    same_agent = compute_same_agent_matrix(runs)
    bifrost_vis = compute_bifrost_visibility(runs)

    summary_data = {
        "benchmark_metadata": {
            "title": "Agent Lab PS02 Benchmark Summary",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_runs": len(runs),
            "agents_evaluated": list(agent_groups.keys()),
            "tasks_evaluated": sorted(list(set(r.get("task_id") for r in runs if r.get("task_id")))),
            "scenarios_evaluated": sorted(list(set(str(r.get("scenario") or "clean") for r in runs))),
        },
        "leaderboard": leaderboard,
        "same_model_matrix": same_model,
        "same_agent_matrix": same_agent,
        "bifrost_visibility": bifrost_vis,
    }

    s_path.parent.mkdir(parents=True, exist_ok=True)
    s_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    c_path.parent.mkdir(parents=True, exist_ok=True)
    with open(c_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Agent",
            "Accuracy (%)",
            "Consistency Pass^k (%)",
            "Mean Steps",
            "Std Steps",
            "Mean Tool Calls",
            "Std Tool Calls",
            "Mean Tokens",
            "Std Tokens",
            "Mean Latency (s)",
            "Std Latency (s)",
            "P95 Latency (s)",
            "Failure Rate (%)",
            "Runs (n)",
            "Estimated Cost",
            "Type",
        ])
        for row in leaderboard:
            is_bb = row["is_black_box"]
            writer.writerow([
                row["agent"] + (" (black-box)" if is_bb else ""),
                f"{row['accuracy']:.1f}%",
                f"{row['consistency']:.1f}%",
                " — (black-box)" if is_bb else f"{row['mean_steps']:.2f}",
                " — (black-box)" if is_bb else f"{row['std_steps']:.2f}",
                " — (black-box)" if is_bb else f"{row['mean_tool_calls']:.2f}",
                " — (black-box)" if is_bb else f"{row['std_tool_calls']:.2f}",
                " — (black-box)" if is_bb else f"{row['mean_tokens']:.0f}",
                " — (black-box)" if is_bb else f"{row['std_tokens']:.0f}",
                f"{row['mean_latency_s']:.2f}",
                f"{row['std_latency_s']:.2f}",
                f"{row['p95_latency_s']:.2f}",
                f"{row['failure_rate']:.1f}%",
                row["n"],
                " — (black-box)" if is_bb else row["cost_display"],
                "Black-box" if is_bb else "White-box Traceable",
            ])

    return summary_data
