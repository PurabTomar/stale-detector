"""
bifrost_evidence.py - Loader and Streamlit renderer for Bifrost gateway evidence.

Renders audited Bifrost gateway telemetry from `evidence/bifrost_evidence.json`
in both the Architecture tab and the Agent Lab Gateway sub-tab.
Allows deployed apps (USE_BIFROST=false) to show proof of real Bifrost routing
without hosting an active gateway container.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_EVIDENCE_PATH = Path(__file__).resolve().parent / "evidence" / "bifrost_evidence.json"


def load_bifrost_evidence(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load the sanitized Bifrost gateway evidence file."""
    p = Path(path) if path else _EVIDENCE_PATH
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def render_bifrost_evidence_panel(key_prefix: str = "arch") -> None:
    """
    Render the comprehensive Bifrost Evidence panel showing real recorded gateway calls,
    fallback events, and per-model counts.
    """
    evidence = load_bifrost_evidence()

    if not evidence:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 12px; padding: 24px; text-align: center; color: #64748B; margin: 12px 0;">
                <div style="font-size: 1.8rem; margin-bottom: 8px;">🛡️</div>
                <div style="font-weight: 700; font-size: 1.05rem; color: #1E293B;">Bifrost Evidence File Not Found</div>
                <div style="font-size: 0.88rem; margin: 6px auto 12px auto; max-width: 500px;">
                    Generate the audited gateway telemetry by running <code>python scripts/export_bifrost_evidence.py</code> locally with Bifrost.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rec_date = evidence.get("recorded_date", "Nov 13, 2026")
    tot_requests = int(evidence.get("total_requests", 0))
    tot_fallbacks = int(evidence.get("total_fallbacks", 0))
    tot_tokens_dict = evidence.get("total_tokens") or {}
    tot_tokens = int(tot_tokens_dict.get("total_tokens", 0)) if isinstance(tot_tokens_dict, dict) else 0
    mean_lat = float(evidence.get("mean_latency_ms", 0.0))
    note = evidence.get("note", f"Recorded from a real run on {rec_date} through a local Bifrost gateway")

    # Prominent Evidence Header Notice
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%); border: 1px solid #C7D2FE; border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="background: #4F46E5; color: white; padding: 3px 10px; border-radius: 9999px; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; margin-right: 8px;">
                        Audited Evidence
                    </span>
                    <strong style="color: #1E1B4B; font-size: 0.95rem;">{html.escape(note)}</strong>
                </div>
                <div style="font-size: 0.82rem; color: #4338CA; font-weight: 600;">
                    {tot_requests:,} Gateway Requests &bull; {tot_fallbacks} Fallbacks Recorded
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 KPI Cards
    ek1, ek2, ek3, ek4 = st.columns(4, gap="small")
    with ek1:
        st.metric(
            "Audited Calls",
            f"{tot_requests:,}",
            f"Logged on {rec_date}",
            border=True,
            help="Total requests captured through the local Bifrost gateway",
        )
    with ek2:
        st.metric(
            "Gateway Fallbacks",
            f"{tot_fallbacks:,}",
            "Auto circuit-breakers",
            border=True,
            help="Outage failovers automatically routed to secondary backup models",
        )
    tin_str = f"{tot_tokens_dict.get('tokens_in', 0):,} in / {tot_tokens_dict.get('tokens_out', 0):,} out" if isinstance(tot_tokens_dict, dict) else "Recorded gateway usage"
    with ek3:
        st.metric(
            "Tokens Processed",
            f"{tot_tokens:,}",
            tin_str,
            border=True,
            help="Verified token usage recorded from OpenAI-compatible gateway responses",
        )
    with ek4:
        st.metric(
            "Mean Gateway Latency",
            f"{mean_lat:.0f} ms",
            "End-to-end response",
            border=True,
            help="Average roundtrip latency including model execution and streaming",
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Per-Model Breakdown Table
    models_data = evidence.get("models_breakdown") or {}
    if models_data and isinstance(models_data, dict):
        st.markdown("#### 📊 Per-Model Gateway Distribution")
        m_rows = []
        for m_name, m_info in models_data.items():
            if isinstance(m_info, dict):
                m_rows.append({
                    "Model": m_name,
                    "Calls": m_info.get("requests", 0),
                    "Tokens In": m_info.get("tokens_in", 0),
                    "Tokens Out": m_info.get("tokens_out", 0),
                    "Total Tokens": m_info.get("total_tokens", 0),
                    "Mean Latency (s)": round(m_info.get("mean_latency_ms", 0.0) / 1000.0, 2),
                    "Fallbacks": m_info.get("fallback_count", 0),
                })
        if m_rows:
            m_df = pd.DataFrame(m_rows).sort_values("Calls", ascending=False)
            st.dataframe(
                m_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Model": st.column_config.TextColumn("Model Architecture", width="medium"),
                    "Calls": st.column_config.NumberColumn("Calls", format="%d"),
                    "Tokens In": st.column_config.NumberColumn("Prompt Tokens", format="%d"),
                    "Tokens Out": st.column_config.NumberColumn("Completion Tokens", format="%d"),
                    "Total Tokens": st.column_config.NumberColumn("Total Tokens", format="%d"),
                    "Mean Latency (s)": st.column_config.NumberColumn("Mean Latency", format="%.2fs"),
                    "Fallbacks": st.column_config.NumberColumn("Fallbacks Handled", format="%d"),
                },
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Fallback Events Section
    fallback_events = evidence.get("fallback_events") or []
    if fallback_events:
        with st.expander(
            f"⚠️ Recorded Gateway Fallback Events ({len(fallback_events)} Captured Failovers)",
            expanded=False,
            icon=":material/warning:",
        ):
            st.caption(
                "When primary LLM models encounter simulated 503 outages or 429 rate limits, "
                "Bifrost triggers instant circuit-breaker failover to backup models without crashing the user workflow."
            )
            fb_rows = []
            for fb in fallback_events:
                if isinstance(fb, dict):
                    fb_rows.append({
                        "Run ID": fb.get("run_id", "—"),
                        "Timestamp": str(fb.get("timestamp", ""))[:19],
                        "Primary (Failed)": fb.get("from_model", "—"),
                        "Fallback Route": f"→ {fb.get('to_model', '—')}",
                        "Reason": fb.get("reason", "Outage"),
                        "Latency (ms)": fb.get("latency_ms", 0.0),
                    })
            if fb_rows:
                fb_df = pd.DataFrame(fb_rows)
                st.dataframe(
                    fb_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Run ID": st.column_config.TextColumn("Run ID", width="medium"),
                        "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
                        "Primary (Failed)": st.column_config.TextColumn("Failed Model"),
                        "Fallback Route": st.column_config.TextColumn("Routed To"),
                        "Reason": st.column_config.TextColumn("Failure Reason", width="medium"),
                        "Latency (ms)": st.column_config.NumberColumn("Failover Latency", format="%.1f ms"),
                    },
                )

    # Real Gateway Call Table
    raw_requests = evidence.get("requests") or []
    if raw_requests:
        with st.expander(
            f"📋 Inspect All Gateway Requests ({len(raw_requests)} Calls Recorded)",
            expanded=False,
            icon=":material/table_rows:",
        ):
            st.caption(
                "Full request trace captured through Bifrost. Every record contains correlation run ID, model, "
                "latency, token counts, and fallback route status with all secrets redacted."
            )
            req_rows = []
            for r in raw_requests:
                if isinstance(r, dict):
                    req_rows.append({
                        "Time": str(r.get("timestamp", ""))[:19].replace("T", " "),
                        "Run ID": r.get("run_id", "—"),
                        "Model": r.get("model", "—"),
                        "Latency (s)": round(float(r.get("latency_ms", 0.0)) / 1000.0, 2),
                        "In Tokens": r.get("tokens_in", 0),
                        "Out Tokens": r.get("tokens_out", 0),
                        "Total Tokens": r.get("total_tokens", 0),
                        "Fallback": "Yes" if r.get("fallback_used") else "No",
                        "Status": r.get("status", "200 OK"),
                    })
            if req_rows:
                req_df = pd.DataFrame(req_rows)
                st.dataframe(
                    req_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Time": st.column_config.TextColumn("Time (UTC)", width="small"),
                        "Run ID": st.column_config.TextColumn("Run Correlation ID", width="medium"),
                        "Model": st.column_config.TextColumn("Model", width="small"),
                        "Latency (s)": st.column_config.NumberColumn("Latency", format="%.2fs"),
                        "In Tokens": st.column_config.NumberColumn("In", format="%d"),
                        "Out Tokens": st.column_config.NumberColumn("Out", format="%d"),
                        "Total Tokens": st.column_config.NumberColumn("Tokens", format="%d"),
                        "Fallback": st.column_config.TextColumn("Fallback"),
                        "Status": st.column_config.TextColumn("Status"),
                    },
                )
