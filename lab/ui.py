"""
lab/ui.py - Agent Lab UI Tab for the Stale-Answer Detector Streamlit App.

Implements Problem Statement 02: "Build a Benchmarking Platform for AI Agents, Not Just Models".
Reads exclusively from saved local files in lab/results/ (runs.jsonl and optional summary.json).
Performs ZERO API calls and adheres to the app's clean light SaaS aesthetic.
"""

from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

_RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DEFAULT_RUNS_PATH = _RESULTS_DIR / "runs.jsonl"
_DEFAULT_SUMMARY_PATH = _RESULTS_DIR / "summary.json"
_DEFAULT_SUITE_PATH = Path(__file__).resolve().parent / "tasks" / "research_freshness.json"


# ---------------------------------------------------------------------------
# Data Loading & Processing (Read-Only, Zero API Calls)
# ---------------------------------------------------------------------------

def load_runs(results_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load all completed run records from JSONL file."""
    path = Path(results_path) if results_path else _DEFAULT_RUNS_PATH
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        runs: list[dict[str, Any]] = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except Exception:
                    continue
        return runs
    except Exception:
        return []


def load_tasks_meta(suite_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Load task metadata dictionary keyed by task_id."""
    path = Path(suite_path) if suite_path else _DEFAULT_SUITE_PATH
    if not path.is_file():
        return {}
    try:
        tasks = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(tasks, list):
            return {t["id"]: t for t in tasks if isinstance(t, dict) and "id" in t}
        return {}
    except Exception:
        return {}


def get_last_run_timestamp(runs: list[dict[str, Any]]) -> str:
    """Extract and format the timestamp of the most recent run."""
    if not runs:
        return "Not run yet"
    last_ts_str = None
    for r in reversed(runs):
        ts = r.get("ended_at") or r.get("started_at")
        if ts:
            last_ts_str = ts
            break
    if not last_ts_str:
        return "Unknown"
    try:
        # Parse ISO 8601 string
        dt = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y %H:%M UTC")
    except Exception:
        return str(last_ts_str)[:19]


def build_leaderboard_df(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Compute leaderboard metrics per agent:
    Accuracy (%), Mean Steps, Mean Tool Calls, Mean Tokens, Mean Latency (s),
    Failure Rate (%), and Runs (n).
    """
    if not runs:
        return pd.DataFrame(
            columns=[
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
        )

    agent_groups = defaultdict(list)
    for r in runs:
        agent_name = str(r.get("agent") or "unknown")
        agent_groups[agent_name].append(r)

    rows = []
    for agent_name, a_runs in agent_groups.items():
        n = len(a_runs)
        succ_count = sum(1 for r in a_runs if bool(r.get("success")))
        acc = (succ_count / n) * 100.0 if n > 0 else 0.0
        fail_rate = 100.0 - acc

        tot_steps = sum(r.get("totals", {}).get("steps", 0) for r in a_runs)
        tot_tools = sum(r.get("totals", {}).get("tool_calls", 0) for r in a_runs)
        tot_tokens = sum(r.get("totals", {}).get("total_tokens", 0) for r in a_runs)
        tot_latency_ms = sum(r.get("totals", {}).get("total_latency_ms", 0.0) for r in a_runs)
        error_count = sum(1 for r in a_runs if bool(r.get("result", {}).get("error")))

        mean_steps = tot_steps / n if n > 0 else 0.0
        mean_tools = tot_tools / n if n > 0 else 0.0
        mean_tokens = tot_tokens / n if n > 0 else 0.0
        mean_latency_s = (tot_latency_ms / (n * 1000.0)) if n > 0 else 0.0

        rows.append(
            {
                "Agent": agent_name,
                "Accuracy (%)": round(acc, 1),
                "Mean Steps": round(mean_steps, 2),
                "Mean Tool Calls": round(mean_tools, 2),
                "Mean Tokens": int(round(mean_tokens)),
                "Mean Latency (s)": round(mean_latency_s, 2),
                "Failure Rate (%)": round(fail_rate, 1),
                "Runs (n)": n,
                "Errors": error_count,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Accuracy (%)", "Mean Latency (s)"], ascending=[False, True]).reset_index(drop=True)
    return df


def style_leaderboard(df: pd.DataFrame) -> Any:
    """Highlight best values in green with good text contrast."""
    if df.empty:
        return df

    def _highlight_col(col: pd.Series, lower_is_better: bool = False) -> list[str]:
        if col.name not in df.columns or col.empty or not pd.api.types.is_numeric_dtype(col):
            return ["" for _ in col]
        best_val = col.min() if lower_is_better else col.max()
        return [
            "background-color: #DCFCE7; color: #166534; font-weight: 700;"
            if v == best_val
            else ""
            for v in col
        ]

    styler = df.style
    styler.apply(_highlight_col, subset=["Accuracy (%)"], lower_is_better=False)
    styler.apply(_highlight_col, subset=["Mean Steps"], lower_is_better=True)
    styler.apply(_highlight_col, subset=["Mean Tool Calls"], lower_is_better=True)
    styler.apply(_highlight_col, subset=["Mean Tokens"], lower_is_better=True)
    styler.apply(_highlight_col, subset=["Mean Latency (s)"], lower_is_better=True)
    styler.apply(_highlight_col, subset=["Failure Rate (%)"], lower_is_better=True)
    return styler


# ---------------------------------------------------------------------------
# UI Components & Sections
# ---------------------------------------------------------------------------

def _render_kpi_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: var(--bg-card, #FFFFFF);
            border: 1px solid var(--border-main, #E5E7EF);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: var(--shadow-sm, 0 1px 2px rgba(15,23,42,0.05));
        ">
            <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-muted, #64748B); text-transform: uppercase; letter-spacing: 0.05em;">{html.escape(title)}</div>
            <div style="font-size: 1.85rem; font-weight: 800; color: var(--text-main, #0F172A); margin: 4px 0;">{html.escape(str(value))}</div>
            <div style="font-size: 0.82rem; color: var(--text-subtle, #94A3B8);">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_status_badge(status: str) -> str:
    status_lower = str(status).strip().lower()
    color_map = {
        "fresh": ("#16A34A", "#DCFCE7", "Fresh"),
        "outdated": ("#EA580C", "#FFEDD5", "Outdated"),
        "incorrect": ("#DC2626", "#FEE2E2", "Incorrect"),
        "partially_correct": ("#D97706", "#FEF3C7", "Partially Correct"),
        "uncertain": ("#475569", "#E2E8F0", "Uncertain"),
        "unverifiable": ("#6B7280", "#F3F4F6", "Unverifiable"),
    }
    text_color, bg_color, label = color_map.get(status_lower, ("#475569", "#E2E8F0", status.capitalize()))
    return (
        f'<span style="background-color: {bg_color}; color: {text_color}; '
        f'padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem; '
        f'display: inline-block;">{html.escape(label)}</span>'
    )


def _render_scatter_charts(df: pd.DataFrame) -> None:
    """Render dual scatter charts for Accuracy vs Tokens and Accuracy vs Latency."""
    if df.empty or len(df) < 2:
        st.info("Insufficient agent runs to generate scatter charts.", icon=":material/bar_chart:")
        return

    col1, col2 = st.columns(2, gap="medium")

    # Chart 1: Accuracy vs Tokens
    with col1:
        st.markdown("**Accuracy vs. Mean Tokens per Run**")
        try:
            points1 = (
                alt.Chart(df)
                .mark_circle(size=140)
                .encode(
                    x=alt.X("Mean Tokens:Q", title="Mean Tokens / Run", scale=alt.Scale(zero=False)),
                    y=alt.Y("Accuracy (%):Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 105])),
                    color=alt.Color("Agent:N", legend=None),
                    tooltip=["Agent", "Accuracy (%)", "Mean Tokens", "Mean Latency (s)", "Runs (n)"],
                )
            )
            labels1 = (
                alt.Chart(df)
                .mark_text(dx=10, dy=-6, fontSize=11, fontWeight=600)
                .encode(
                    x="Mean Tokens:Q",
                    y="Accuracy (%):Q",
                    text="Agent:N",
                    color=alt.value("#0F172A"),
                )
            )
            chart1 = (points1 + labels1).properties(height=280)
            st.altair_chart(chart1, use_container_width=True)
        except Exception:
            st.scatter_chart(df, x="Mean Tokens", y="Accuracy (%)", color="Agent")

    # Chart 2: Accuracy vs Latency
    with col2:
        st.markdown("**Accuracy vs. Mean Latency (seconds)**")
        try:
            points2 = (
                alt.Chart(df)
                .mark_circle(size=140)
                .encode(
                    x=alt.X("Mean Latency (s):Q", title="Mean Latency (seconds)", scale=alt.Scale(zero=False)),
                    y=alt.Y("Accuracy (%):Q", title="Accuracy (%)", scale=alt.Scale(domain=[0, 105])),
                    color=alt.Color("Agent:N", legend=None),
                    tooltip=["Agent", "Accuracy (%)", "Mean Latency (s)", "Mean Tokens", "Runs (n)"],
                )
            )
            labels2 = (
                alt.Chart(df)
                .mark_text(dx=10, dy=-6, fontSize=11, fontWeight=600)
                .encode(
                    x="Mean Latency (s):Q",
                    y="Accuracy (%):Q",
                    text="Agent:N",
                    color=alt.value("#0F172A"),
                )
            )
            chart2 = (points2 + labels2).properties(height=280)
            st.altair_chart(chart2, use_container_width=True)
        except Exception:
            st.scatter_chart(df, x="Mean Latency (s)", y="Accuracy (%)", color="Agent")


def _render_run_explorer(runs: list[dict[str, Any]], tasks_meta: dict[str, dict[str, Any]]) -> None:
    """Explore individual run executions with step-by-step traces."""
    st.subheader("🔍 Run Explorer & Trace Inspector", divider="gray")

    agents = sorted(set(str(r.get("agent") or "") for r in runs if r.get("agent")))
    tasks = sorted(set(str(r.get("task_id") or "") for r in runs if r.get("task_id")))

    if not agents or not tasks:
        st.info("No runs available to inspect.", icon=":material/search:")
        return

    c1, c2, c3, c4 = st.columns([1.5, 2.5, 1, 1.2], gap="small")
    with c1:
        sel_agent = st.selectbox("Agent", options=agents, index=0, key="exp_agent")
    with c2:
        task_labels = [f"{t} ({tasks_meta.get(t, {}).get('category', 'task')})" for t in tasks]
        sel_task_idx = st.selectbox(
            "Task",
            options=range(len(tasks)),
            format_func=lambda i: task_labels[i],
            key="exp_task",
        )
        sel_task = tasks[sel_task_idx]
    with c3:
        repeats_avail = sorted(set(int(r.get("repeat", 1)) for r in runs if r.get("agent") == sel_agent and r.get("task_id") == sel_task))
        if not repeats_avail:
            repeats_avail = [1]
        sel_repeat = st.selectbox("Repeat", options=repeats_avail, index=0, key="exp_repeat")
    with c4:
        scenarios_avail = sorted(set(str(r.get("scenario") or "clean") for r in runs if r.get("agent") == sel_agent and r.get("task_id") == sel_task))
        if not scenarios_avail:
            scenarios_avail = ["clean"]
        st.selectbox("Scenario", options=scenarios_avail, index=0, key="exp_scenario", help="Chaos scenarios: not run yet")

    # Find matching run
    matched_run = None
    for r in runs:
        if (
            r.get("agent") == sel_agent
            and r.get("task_id") == sel_task
            and int(r.get("repeat", 1)) == sel_repeat
        ):
            matched_run = r
            break

    if not matched_run:
        st.warning(f"No run found for `{sel_agent}` on task `{sel_task}` (repeat {sel_repeat}).", icon=":material/warning:")
        return

    task_info = tasks_meta.get(sel_task, {})
    res_obj = matched_run.get("result", {})
    totals = matched_run.get("totals", {})
    success = bool(matched_run.get("success"))
    is_black_box = bool(matched_run.get("is_black_box")) or len(matched_run.get("steps", [])) == 0

    # Summary card
    with st.container(border=True):
        sc_col1, sc_col2 = st.columns([2.5, 1.2], gap="medium")
        with sc_col1:
            st.markdown(f"**Task Claim:** *\"{task_info.get('input_text', sel_task)}\"*")
            st.markdown(
                f"**Expected Status:** {_render_status_badge(task_info.get('expected_status', 'N/A'))} &nbsp;&bull;&nbsp; "
                f"**Expected Facts:** `{task_info.get('expected_facts', [])}`",
                unsafe_allow_html=True,
            )
            val_disp = res_obj.get("current_value") or "*(None)*"
            st.markdown(f"**Agent Current Value:** `{val_disp}`")
            if res_obj.get("error"):
                st.markdown(f'<span style="color: #DC2626; font-weight: 600;">Error: {html.escape(str(res_obj["error"]))}</span>', unsafe_allow_html=True)

        with sc_col2:
            badge_html = (
                '<span style="background:#DCFCE7; color:#166534; padding:4px 12px; border-radius:9999px; font-weight:800; font-size:0.85rem;">PASSED GROUND TRUTH</span>'
                if success
                else '<span style="background:#FEE2E2; color:#991B1B; padding:4px 12px; border-radius:9999px; font-weight:800; font-size:0.85rem;">MISMATCH</span>'
            )
            st.markdown(f"**Outcome:** {badge_html}", unsafe_allow_html=True)
            st.markdown(
                f"**Agent Verdict:** {_render_status_badge(res_obj.get('verdict', 'unknown'))}",
                unsafe_allow_html=True,
            )
            dur_s = (totals.get("total_latency_ms", 0.0)) / 1000.0
            st.caption(
                f"**Latency:** `{dur_s:.2f}s` • **Tokens:** `{totals.get('total_tokens', 0):,}` • "
                f"**Steps:** `{totals.get('steps', 0)}` • **Tool calls:** `{totals.get('tool_calls', 0)}`"
            )

    # Granular Trace Steps
    steps = matched_run.get("steps", [])
    if is_black_box or not steps:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center; color: #64748B;">
                <div style="font-size: 1.6rem; margin-bottom: 6px;">📦</div>
                <div style="font-weight: 700; font-size: 1rem; color: #334155;">Black-box Run</div>
                <div style="font-size: 0.85rem; margin-top: 4px;">This agent operated externally and returned no internal trace steps. Metrics were recorded at the boundary level.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**Execution Trace ({len(steps)} steps recorded):**")
        step_icons = {
            "llm": "🤖",
            "tool": "🔍",
            "decision": "⚖️",
            "fallback": "⚠️",
            "error": "❌",
        }
        for s in steps:
            s_idx = s.get("idx", 1)
            s_kind = str(s.get("kind") or "step").lower()
            s_icon = step_icons.get(s_kind, "⚡")
            s_name = s.get("name") or s_kind
            s_model = s.get("model") or "N/A"
            s_lat = s.get("latency_ms", 0.0)
            s_tin = s.get("tokens_in", 0)
            s_tout = s.get("tokens_out", 0)
            s_err = s.get("error")
            s_cache = "Hit" if s.get("cache_hit") else "Miss"

            exp_label = f"Step {s_idx}: {s_icon} [{s_kind.upper()}] {s_name} — {s_lat:.1f}ms ({s_model})"
            with st.expander(exp_label, expanded=(bool(s_err) or s_idx == 1)):
                m_cols = st.columns(4)
                m_cols[0].metric("Model", s_model)
                m_cols[1].metric("Latency", f"{s_lat:.1f} ms")
                m_cols[2].metric("Tokens", f"{s_tin + s_tout:,}", f"{s_tin} in / {s_tout} out")
                m_cols[3].metric("Tavily Cache", s_cache)

                if s_kind == "fallback":
                    st.warning("⚠️ Gateway fallback event: Primary model was unavailable; secondary fallback model answered.", icon=":material/warning:")

                if s_err:
                    st.error(f"❌ Step Error: {html.escape(str(s_err))}", icon=":material/error:")

                if s.get("input_summary"):
                    st.caption("**Input Summary:**")
                    st.code(s["input_summary"], language="text")
                if s.get("output_summary"):
                    st.caption("**Output Summary:**")
                    st.code(s["output_summary"], language="text")


def _render_compare_view(runs: list[dict[str, Any]], tasks_meta: dict[str, dict[str, Any]]) -> None:
    """Compare two agents side-by-side on the same task."""
    st.subheader("⚖️ Agent Head-to-Head Comparison", divider="gray")

    agents = sorted(set(str(r.get("agent") or "") for r in runs if r.get("agent")))
    tasks = sorted(set(str(r.get("task_id") or "") for r in runs if r.get("task_id")))

    if len(agents) < 2 or not tasks:
        st.info("At least two evaluated agents are required to run side-by-side comparison.", icon=":material/compare:")
        return

    c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 1], gap="small")
    with c1:
        task_labels = [f"{t} ({tasks_meta.get(t, {}).get('category', 'task')})" for t in tasks]
        sel_idx = st.selectbox(
            "Select Task for Comparison",
            options=range(len(tasks)),
            format_func=lambda i: task_labels[i],
            key="cmp_task",
        )
        sel_task = tasks[sel_idx]
    with c2:
        agent_a = st.selectbox("Agent A", options=agents, index=0, key="cmp_agent_a")
    with c3:
        b_idx = 1 if len(agents) > 1 else 0
        agent_b = st.selectbox("Agent B", options=agents, index=b_idx, key="cmp_agent_b")
    with c4:
        sel_repeat = st.selectbox("Repeat", options=[1, 2, 3], index=0, key="cmp_repeat")

    task_info = tasks_meta.get(sel_task, {})
    st.markdown(f"**Claim under test:** *\"{task_info.get('input_text', sel_task)}\"*")
    st.markdown(
        f"**Expected Status:** {_render_status_badge(task_info.get('expected_status', 'N/A'))} &bull; "
        f"**Expected Ground Facts:** `{task_info.get('expected_facts', [])}`",
        unsafe_allow_html=True,
    )

    run_a = next(
        (r for r in runs if r.get("agent") == agent_a and r.get("task_id") == sel_task and int(r.get("repeat", 1)) == sel_repeat),
        None,
    )
    run_b = next(
        (r for r in runs if r.get("agent") == agent_b and r.get("task_id") == sel_task and int(r.get("repeat", 1)) == sel_repeat),
        None,
    )

    col_a, col_b = st.columns(2, gap="large")

    def _render_agent_card(agent_name: str, run_data: dict[str, Any] | None) -> None:
        st.markdown(f"### `{agent_name}`")
        if not run_data:
            st.info(f"No run recorded for `{agent_name}` on this task (repeat {sel_repeat}).")
            return

        res = run_data.get("result", {})
        tot = run_data.get("totals", {})
        succ = bool(run_data.get("success"))

        badge = (
            '<span style="background:#DCFCE7; color:#166534; padding:3px 10px; border-radius:9999px; font-weight:700; font-size:0.8rem;">PASSED</span>'
            if succ
            else '<span style="background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:9999px; font-weight:700; font-size:0.8rem;">FAILED</span>'
        )

        with st.container(border=True):
            st.markdown(f"**Ground Truth:** {badge}", unsafe_allow_html=True)
            st.markdown(f"**Verdict:** {_render_status_badge(res.get('verdict', 'unknown'))}", unsafe_allow_html=True)
            st.markdown(f"**Current Value:** `{res.get('current_value') or 'None'}`")

            lat_s = (tot.get("total_latency_ms", 0.0)) / 1000.0
            st.metric("Latency", f"{lat_s:.2f}s")
            st.metric("Total Tokens", f"{tot.get('total_tokens', 0):,}")
            st.metric("Steps / Tool Calls", f"{tot.get('steps', 0)} / {tot.get('tool_calls', 0)}")

            if res.get("error"):
                st.error(f"Error: {res['error']}")

    with col_a:
        _render_agent_card(agent_a, run_a)
    with col_b:
        _render_agent_card(agent_b, run_b)

    # Agreement / Disagreement conclusion
    if run_a and run_b:
        va = str(run_a.get("result", {}).get("verdict", "")).lower()
        vb = str(run_b.get("result", {}).get("verdict", "")).lower()
        if va == vb:
            st.success(f"🤝 **Consensus**: Both `{agent_a}` and `{agent_b}` reached the exact same verdict: **{va}**.", icon=":material/handshake:")
        else:
            st.warning(f"⚔️ **Disagreement**: `{agent_a}` concluded **{va}**, while `{agent_b}` concluded **{vb}**.", icon=":material/compare_arrows:")


def _render_consistency_view(runs: list[dict[str, Any]], tasks_meta: dict[str, dict[str, Any]]) -> None:
    """Inspect repeat consistency of an agent across multiple runs of the same task."""
    st.subheader("🔁 Multi-Repeat Consistency", divider="gray")

    agents = sorted(set(str(r.get("agent") or "") for r in runs if r.get("agent")))
    tasks = sorted(set(str(r.get("task_id") or "") for r in runs if r.get("task_id")))

    if not agents or not tasks:
        st.info("No runs available for consistency analysis.", icon=":material/repeat:")
        return

    c1, c2 = st.columns([1.5, 3], gap="medium")
    with c1:
        sel_agent = st.selectbox("Select Agent for Consistency", options=agents, index=0, key="cons_agent")
    with c2:
        task_labels = [f"{t} ({tasks_meta.get(t, {}).get('category', 'task')})" for t in tasks]
        sel_idx = st.selectbox(
            "Select Task for Consistency",
            options=range(len(tasks)),
            format_func=lambda i: task_labels[i],
            key="cons_task",
        )
        sel_task = tasks[sel_idx]

    task_info = tasks_meta.get(sel_task, {})
    st.markdown(f"**Task Claim:** *\"{task_info.get('input_text', sel_task)}\"*")

    matching = [r for r in runs if r.get("agent") == sel_agent and r.get("task_id") == sel_task]
    matching.sort(key=lambda r: int(r.get("repeat", 1)))

    if not matching:
        st.info(f"No runs recorded for `{sel_agent}` on task `{sel_task}`.")
        return

    rows = []
    verdicts = []
    for r in matching:
        rep = int(r.get("repeat", 1))
        v = str(r.get("result", {}).get("verdict", "unknown"))
        val = str(r.get("result", {}).get("current_value") or "None")
        succ = "Passed" if r.get("success") else "Mismatch"
        lat = f"{(r.get('totals', {}).get('total_latency_ms', 0.0) / 1000.0):.2f}s"
        toks = f"{r.get('totals', {}).get('total_tokens', 0):,}"
        verdicts.append(v)
        rows.append({
            "Repeat #": rep,
            "Verdict": v,
            "Current Value Returned": val,
            "Ground Truth": succ,
            "Latency": lat,
            "Tokens": toks,
        })

    c_df = pd.DataFrame(rows)
    st.dataframe(c_df, hide_index=True, width="stretch")

    # Consistency score
    unique_v = set(verdicts)
    if len(unique_v) == 1:
        st.success(f"🎯 **100% Deterministic Consistency**: All {len(verdicts)} repeats returned the verdict **'{verdicts[0]}'**.")
    else:
        most_common = max(unique_v, key=verdicts.count)
        freq = verdicts.count(most_common)
        pct = (freq / len(verdicts)) * 100.0
        st.warning(f"⚠️ **Inconsistency Observed**: {freq}/{len(verdicts)} repeats ({pct:.1f}%) returned '{most_common}', but other repeats produced: {list(unique_v - {most_common})}.")


def _render_how_to_add_agent_expander() -> None:
    """Collapsible guide with short Python, HTTP, and CLI adapter snippets."""
    with st.expander("🛠️ How to Add Your Own Research Agent (Python, HTTP, CLI)", expanded=False):
        st.markdown(
            """
            Agent Lab evaluates any AI research or fact-checking agent using a standardized `(Task) -> (AgentResult, Steps)` contract.
            You can register an agent in seconds via three framework-independent adapters:
            """
        )

        p_tab, h_tab, c_tab = st.tabs(["1. Python Adapter (In-Process)", "2. HTTP Adapter (REST Endpoint)", "3. CLI Adapter (JSON stdin/stdout)"])

        with p_tab:
            st.markdown("**Wrap any Python function or agent class in `lab/adapters.py`:**")
            st.code(
                '''from lab.schema import Task, AgentResult, StepRecord
from lab.adapters import PythonAdapter

def my_custom_agent(task: Task, ctx) -> tuple[AgentResult, list[StepRecord]]:
    # 1. Access the claim to audit: task.input_text
    claim = task.input_text
    
    # 2. Execute your agentic loop / tools...
    # 3. Return normalized AgentResult:
    result = AgentResult(
        verdict="outdated",  # fresh | outdated | incorrect | partially_correct | uncertain | unverifiable
        current_value="iPhone 16 Pro Max",
        answer="Apple released the iPhone 16 series in September 2024.",
    )
    return result, []

# Register in lab/agents/__init__.py:
adapter = PythonAdapter(my_custom_agent, name="my_custom_agent")
''',
                language="python",
            )

        with h_tab:
            st.markdown("**Point Agent Lab to any remote or local HTTP microservice (e.g. FastAPI on port 9001):**")
            st.code(
                '''from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/verify")
async def verify_endpoint(request: Request):
    payload = await request.json()
    task = payload["task"]       # Task dictionary
    run_id = payload["run_id"]   # Unique trace execution ID
    
    # Your proprietary agent logic...
    return {
        "result": {
            "verdict": "fresh",
            "current_value": "Python 3.12 is actively supported",
            "answer": "Python 3.12 was released in October 2023.",
            "error": None
        },
        "trace": []  # Optional granular StepRecord objects
    }
''',
                language="python",
            )
            st.caption("Agent Lab POSTs `{\"task\": {...}, \"run_id\": \"...\"}` and accepts `{result: {...}, trace: [...]}`.")

        with c_tab:
            st.markdown("**Wrap any CLI executable or script that reads JSON from `stdin` and writes JSON to `stdout`:**")
            st.code(
                '''# Example CLI script in any language (Python, Node, Go, Rust):
import sys, json

input_data = json.load(sys.stdin)
claim = input_data["input_text"]

# Produce verdict...
output = {
    "result": {
        "verdict": "incorrect",
        "current_value": "Canberra",
        "answer": "The capital of Australia is Canberra, not Sydney."
    },
    "trace": []
}
print(json.dumps(output))
''',
                language="python",
            )

        st.markdown("**Run your agent against the Agent Lab benchmark suite:**")
        st.code("python -m lab.run --agents my_agent --repeats 3 --concurrency 2 --cache-tavily --yes", language="bash")


# ---------------------------------------------------------------------------
# Main Tab Renderer Entrypoint
# ---------------------------------------------------------------------------

def render_agent_lab_tab(
    results_path: Path | str | None = None,
    suite_path: Path | str | None = None,
) -> None:
    """
    Render the Agent Lab tab in Streamlit.
    Strictly read-only; reads only saved files and makes no API calls.
    """
    # 1. Environment & Bifrost Check
    try:
        from llm import is_use_bifrost
        bifrost_active = is_use_bifrost()
    except Exception:
        bifrost_active = os.environ.get("USE_BIFROST", "false").lower() == "true"

    runs = load_runs(results_path)
    tasks_meta = load_tasks_meta(suite_path)
    last_run_date = get_last_run_timestamp(runs)

    # 2. Header Section
    st.markdown(
        """
        <div style="margin-bottom: 8px;">
            <span style="background: linear-gradient(135deg, #5B5BF0, #8B5CF6); color: white; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                Problem Statement 02 &bull; Research Agents
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader(
        "Agent Lab: Benchmarking AI Agents, Not Just Models",
        icon=":material/science:",
    )
    st.caption(
        "Framework-independent benchmark evaluating research and fact-verification agents on autonomous freshness detection, tool usage, latency, and reasoning consistency."
    )

    # If USE_BIFROST is false (deployed app), show real-run notice
    if not bifrost_active:
        st.info(
            f"ℹ️ Results are from a real benchmark run on **{last_run_date}** using a local Bifrost gateway.",
            icon=":material/info:",
        )

    # If no runs exist, show friendly empty state
    if not runs:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E5E7EF; border-radius: 16px; padding: 40px; text-align: center; margin: 20px 0;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">🧪</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">No Benchmark Runs Found</div>
                <div style="color: #64748B; max-width: 500px; margin: 8px auto 20px auto; font-size: 0.92rem;">
                    No results found in <code>lab/results/runs.jsonl</code>. Run the benchmark suite via the terminal to populate Agent Lab metrics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code("python -m lab.run --agents all --repeats 3 --concurrency 2 --cache-tavily --yes", language="bash")
        _render_how_to_add_agent_expander()
        return

    # 3. KPI Header Grid
    unique_agents = sorted(set(r.get("agent") for r in runs if r.get("agent")))
    unique_tasks = sorted(set(r.get("task_id") for r in runs if r.get("task_id")))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4, gap="small")
    with kpi1:
        _render_kpi_card("Agents Evaluated", str(len(unique_agents)), "In-process & HTTP")
    with kpi2:
        _render_kpi_card("Suite Tasks", str(len(unique_tasks)), "15 Ground truth facts")
    with kpi3:
        _render_kpi_card("Total Executed Runs", f"{len(runs):,}", "Recorded in runs.jsonl")
    with kpi4:
        _render_kpi_card("Last Benchmark", last_run_date, "Clean scenario (chaos: not run yet)")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 4. Leaderboard Table
    df = build_leaderboard_df(runs)

    lead_head_col, download_col = st.columns([3, 1], vertical_alignment="bottom")
    with lead_head_col:
        st.subheader("🏆 Agent Leaderboard", divider="gray")
    with download_col:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="agent_lab_leaderboard.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )

    st.caption("Green highlights denote the best value across each metric (highest accuracy, lowest latency/tokens/steps/failure rate).")
    styled_df = style_leaderboard(df)
    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Agent": st.column_config.TextColumn("Agent", width="medium"),
            "Accuracy (%)": st.column_config.NumberColumn("Accuracy (%)", format="%.1f%%"),
            "Mean Steps": st.column_config.NumberColumn("Mean Steps", format="%.2f"),
            "Mean Tool Calls": st.column_config.NumberColumn("Mean Tools", format="%.2f"),
            "Mean Tokens": st.column_config.NumberColumn("Mean Tokens", format="%d"),
            "Mean Latency (s)": st.column_config.NumberColumn("Mean Latency", format="%.2fs"),
            "Failure Rate (%)": st.column_config.NumberColumn("Failure Rate", format="%.1f%%"),
            "Runs (n)": st.column_config.NumberColumn("Runs (n)", format="%d"),
            "Errors": st.column_config.NumberColumn("Errors", format="%d"),
        },
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 5. Scatter Charts
    _render_scatter_charts(df)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 6. Run Explorer & Traces
    _render_run_explorer(runs, tasks_meta)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 7. Compare Two Agents Side-by-Side
    _render_compare_view(runs, tasks_meta)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 8. Consistency View
    _render_consistency_view(runs, tasks_meta)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 9. How to add your own agent expander
    _render_how_to_add_agent_expander()
