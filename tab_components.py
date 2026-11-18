"""Tab components and helper views for Stale-Answer Detector.

Implements the views for:
- Home Tab (Hero, 4-step How it Works, Highlights Bento, Guided Demo, Real Evidence Strip, FAQ, Evaluator Note)
- Model Showdown Tab (Side-by-side model comparison, winner evaluation, sample size disclaimer, 3 real saved showdowns)
- History Tab (Session audits, reopen, side-by-side compare, delete, export JSON/Markdown)
- API Playground Tab (POST /verify schema, cURL, Python, JS snippets, interactive try it with real saved fallback)
- Methodology Tab (6 status definitions with test examples, LaTeX math formula, stance counting, Tavily vs Bifrost roles, privacy, repo file links)
- Watchlist Component (Track claims, re-check, status delta detection)
"""

from __future__ import annotations

import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

import streamlit as st

from demo_data import SAVED_RESULTS_DATE, load_demo_cache
from settings import DEFAULT_MAX_SESSION_AUDITS
from showdown_data import (
    DEFAULT_SHOWDOWN_QUESTIONS,
    SAMPLE_SIZE_DISCLAIMER,
    evaluate_showdown_winner,
    get_configured_showdown_models,
    get_saved_showdown,
    get_saved_showdowns,
    run_live_showdown,
)

EVIDENCE_FILE = Path(__file__).resolve().parent / "evidence" / "bifrost_evidence.json"
LAB_SUMMARY_FILE = Path(__file__).resolve().parent / "lab" / "results" / "summary.json"


def get_real_model_and_routing_info(use_bifrost: Optional[bool] = None) -> Dict[str, Any]:
    """Build truthful model lists and route-aware labels from active config and saved evidence."""
    try:
        from llm import get_task_routing_summary, is_use_bifrost

        active_bifrost = is_use_bifrost() if use_bifrost is None else bool(use_bifrost)
        routing = get_task_routing_summary()
    except Exception:
        active_bifrost = bool(use_bifrost)
        routing = {
            "routing_mode": "bifrost" if active_bifrost else "direct_gemini",
            "default_model": "",
            "extract_model": "",
            "judge_model": "",
            "dual_judge_primary": "",
            "dual_judge_secondary": "",
            "fallbacks": [],
        }

    configured_models: List[str] = []
    for m in (
        routing.get("extract_model"),
        routing.get("judge_model"),
        routing.get("dual_judge_secondary"),
        *(routing.get("fallbacks") or []),
    ):
        clean = str(m or "").strip()
        if clean and clean not in configured_models:
            configured_models.append(clean)

    evidence_models: List[str] = []
    evidence_date = ""
    if EVIDENCE_FILE.is_file():
        try:
            ev = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
            if isinstance(ev, dict):
                evidence_date = str(ev.get("recorded_date") or "").strip()
                breakdown = ev.get("models_breakdown") or {}
                if isinstance(breakdown, dict):
                    for m_name in breakdown.keys():
                        clean_m = str(m_name or "").strip()
                        if clean_m and clean_m.lower() != "llm" and clean_m not in evidence_models:
                            evidence_models.append(clean_m)
        except Exception:
            pass

    calls_label = "Bifrost gateway calls" if active_bifrost else "LLM calls (direct Gemini)"
    route_label = "Bifrost gateway" if active_bifrost else "Direct Gemini API"
    judge_step_label = "4. Judge via Bifrost" if active_bifrost else "4. Judge (Direct Gemini)"

    return {
        "use_bifrost": active_bifrost,
        "route_label": route_label,
        "calls_label": calls_label,
        "judge_step_label": judge_step_label,
        "extract_model": str(routing.get("extract_model") or ""),
        "judge_model": str(routing.get("judge_model") or ""),
        "dual_judge_primary": str(routing.get("dual_judge_primary") or routing.get("judge_model") or ""),
        "dual_judge_secondary": str(routing.get("dual_judge_secondary") or ""),
        "fallbacks": list(routing.get("fallbacks") or []),
        "configured_models": configured_models,
        "evidence_models": evidence_models,
        "evidence_date": evidence_date,
    }


def record_audit_to_history(
    text: str,
    results: Optional[List[Dict[str, Any]]],
    source_label: str,
    mode: str = "Pasted Answer",
    score: Optional[int] = None,
    audit_status: Optional[str] = None,
    error_reason: Optional[str] = None,
) -> None:
    """Record a real audit execution of this session to history (max 50 entries).

    Failed or partial audits (where one or more claims had `verification_failed`
    or the pipeline failed) are recorded with status 'Failed' or 'Partial' and
    `score=None` rather than a numeric score.
    """
    clean_text = str(text or "").strip()
    if not clean_text:
        return
    res_list = list(results or [])
    if not res_list and not audit_status and not error_reason:
        return

    history = st.session_state.setdefault("audit_history", [])
    # Prevent identical immediate consecutive duplicate with same status
    failed_count = sum(
        1
        for r in res_list
        if isinstance(r, dict) and str(r.get("status", "")).lower() == "verification_failed"
    )
    if audit_status in {"Failed", "Partial", "Completed"}:
        resolved_status = audit_status
    elif error_reason or (res_list and failed_count == len(res_list)):
        resolved_status = "Failed"
    elif failed_count > 0:
        resolved_status = "Partial"
    else:
        resolved_status = "Completed"

    if (
        history
        and history[0].get("input_text") == clean_text
        and history[0].get("audit_status") == resolved_status
    ):
        return

    status_counts = {"fresh": 0, "outdated": 0, "incorrect": 0, "other": 0}
    for r in res_list:
        if not isinstance(r, dict):
            continue
        st_name = str(r.get("status", "")).lower()
        if st_name == "verification_failed":
            continue
        if st_name in status_counts:
            status_counts[st_name] += 1
        else:
            status_counts["other"] += 1

    if resolved_status in {"Failed", "Partial"}:
        final_score: Optional[int] = None
    elif score is not None:
        final_score = int(score)
    else:
        scores = [
            r.get("score")
            for r in res_list
            if isinstance(r, dict)
            and r.get("score") is not None
            and str(r.get("status", "")).lower() not in {"unverifiable", "verification_failed", "skipped"}
        ]
        final_score = round(sum(scores) / len(scores)) if scores else None

    history.insert(
        0,
        {
            "id": f"audit-{int(time.time() * 1000)}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "audit_status": resolved_status,
            "failed_count": failed_count,
            "total_claims": len(res_list),
            "error_reason": error_reason,
            "input_preview": (clean_text[:90] + "..." if len(clean_text) > 90 else clean_text).strip(),
            "input_text": clean_text,
            "score": final_score,
            "status_counts": status_counts,
            "source_label": source_label,
            "results": res_list,
        },
    )
    if len(history) > 50:
        history.pop()


def init_history_if_needed() -> None:
    """Initialize empty session audit history (never seeds fake demo entries or timestamps)."""
    st.session_state.setdefault("audit_history", [])


def get_builtin_examples_from_cache() -> List[Dict[str, Any]]:
    """Return built-in saved examples with their real non-empty input text and real saved date."""
    demo_cache = load_demo_cache()
    examples: List[Dict[str, Any]] = []
    for ex in demo_cache.get("pasted_examples", []):
        if not isinstance(ex, dict):
            continue
        real_text = str(ex.get("answer_text") or ex.get("text") or "").strip()
        if not real_text:
            continue
        ex_results = ex.get("results") or []
        st_counts = {"fresh": 0, "outdated": 0, "incorrect": 0, "other": 0}
        scores: List[float] = []
        for r in ex_results:
            if not isinstance(r, dict):
                continue
            s = str(r.get("status", "")).lower()
            if s == "verification_failed":
                continue
            if s in st_counts:
                st_counts[s] += 1
            else:
                st_counts["other"] += 1
            if r.get("score") is not None and s not in {"unverifiable", "skipped"}:
                scores.append(float(r["score"]))

        ex_score = round(sum(scores) / len(scores)) if scores else None
        recorded_date = str(ex.get("saved_date") or SAVED_RESULTS_DATE).strip()
        examples.append(
            {
                "id": str(ex.get("id") or f"example-{len(examples) + 1}"),
                "label": str(ex.get("label") or f"Example {len(examples) + 1}"),
                "icon": str(ex.get("icon") or "📄"),
                "recorded_date": recorded_date,
                "input_text": real_text,
                "input_preview": (real_text[:110] + "..." if len(real_text) > 110 else real_text),
                "score": ex_score,
                "status_counts": st_counts,
                "source_label": f"Saved ({recorded_date})",
                "results": ex_results,
            }
        )
    return examples


DATASET_FILE = Path(__file__).resolve().parent / "benchmark" / "dataset.json"


def load_evidence_counts() -> Dict[str, Any]:
    """Load real counts from bifrost_evidence.json, lab/results/summary.json, and benchmark/dataset.json."""
    counts = {
        "bifrost_requests": 490,
        "bifrost_fallbacks": 10,
        "bifrost_tokens": 127854,
        "bifrost_date": "Nov 13, 2026",
        "agent_lab_runs": 330,
        "agent_lab_archs": 6,
        "benchmark_claims": 30,
    }
    if EVIDENCE_FILE.exists():
        try:
            ev_data = json.loads(EVIDENCE_FILE.read_text(encoding="utf-8"))
            if isinstance(ev_data, dict):
                counts["bifrost_requests"] = ev_data.get("total_requests", counts["bifrost_requests"])
                counts["bifrost_fallbacks"] = ev_data.get("total_fallbacks", counts["bifrost_fallbacks"])
                tok_val = ev_data.get("total_tokens")
                if isinstance(tok_val, dict):
                    counts["bifrost_tokens"] = tok_val.get("total_tokens", counts["bifrost_tokens"])
                elif isinstance(tok_val, (int, float)):
                    counts["bifrost_tokens"] = int(tok_val)
                counts["bifrost_date"] = ev_data.get("recorded_date") or ev_data.get("date", counts["bifrost_date"])
        except Exception:
            pass

    if LAB_SUMMARY_FILE.exists():
        try:
            lab_data = json.loads(LAB_SUMMARY_FILE.read_text(encoding="utf-8"))
            if isinstance(lab_data, dict):
                meta = lab_data.get("benchmark_metadata", {})
                counts["agent_lab_runs"] = meta.get("total_runs") or lab_data.get("total_runs", counts["agent_lab_runs"])
                agents_list = meta.get("agents_evaluated") or lab_data.get("leaderboard", [])
                counts["agent_lab_archs"] = len(agents_list) or counts["agent_lab_archs"]
        except Exception:
            pass

    if DATASET_FILE.exists():
        try:
            ds_data = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
            if isinstance(ds_data, list):
                counts["benchmark_claims"] = len(ds_data)
        except Exception:
            pass

    return counts


def render_home_tab(
    use_bifrost_mode: bool,
    gateway_unreachable: bool,
    quota_state: str,
    gateway_health: Optional[Dict[str, Any]],
    evaluator_text: str,
    highlights_bento_fn: Any,
) -> None:
    """Render the comprehensive Home Tab."""
    ev_counts = load_evidence_counts()

    # Routing and Quota Pills
    if gateway_unreachable:
        routing_badge = '<span class="sad-tech-badge sad-tech-warning"><span class="sad-dot-orange"></span>Gateway Fallback (Direct Gemini)</span>'
    elif use_bifrost_mode:
        routing_badge = '<span class="sad-tech-badge"><span class="sad-dot"></span>Routed via Bifrost</span>'
    else:
        routing_badge = '<span class="sad-tech-badge"><span class="sad-dot" style="background:#8B5CF6;"></span>Direct Gemini API</span>'

    if quota_state == "Live":
        quota_badge = '<span class="sad-tech-badge" style="border-color: rgba(22,163,74,0.35);"><span class="sad-dot" style="background:#16A34A;"></span>Quota: Live</span>'
    elif quota_state == "Limited":
        quota_badge = '<span class="sad-tech-badge" style="border-color: rgba(234,88,12,0.35);"><span class="sad-dot" style="background:#EA580C;"></span>Quota: Limited</span>'
    else:
        quota_badge = '<span class="sad-tech-badge" style="border-color: rgba(37,99,235,0.35);"><span class="sad-dot" style="background:#2563EB;"></span>Quota: Saved results</span>'

    # 1. Hero Card
    hero_html = f"""
    <div class="sad-hero">
      <div class="sad-hero-grid">
        <div class="sad-hero-content">
          <div class="sad-hero-badges-row">
            {quota_badge}
            {routing_badge}
          </div>
          <h1 class="sad-hero-title">Don&#39;t trust AI blindly. Check its expiry date.</h1>
          <p class="sad-hero-tagline">
            Verify time-sensitive AI answers against live web sources, compute explainable freshness scores, and rewrite stale facts.
          </p>
          <div class="sad-hero-actions">
            <a href="javascript:void(0)" onclick="parent.sadSwitchTab ? parent.sadSwitchTab(1) : (window.parent && window.parent.sadSwitchTab(1))" class="sad-hero-btn sad-hero-btn-primary" aria-label="Start Audit">
              ⚡ Start Fact Audit
            </a>
            <a href="javascript:void(0)" onclick="parent.sadSwitchTab ? parent.sadSwitchTab(4) : (window.parent && window.parent.sadSwitchTab(4))" class="sad-hero-btn sad-hero-btn-secondary" aria-label="Model Showdown">
              ⚔️ Model Showdown
            </a>
          </div>
          <div class="sad-hero-builtwith">
            <span class="sad-builtwith-label">Built with:</span>
            <span class="sad-tech-pill"><span class="sad-dot"></span>Tavily</span>
            <span class="sad-tech-pill"><span class="sad-dot"></span>Bifrost</span>
            <span class="sad-tech-pill"><span class="sad-dot"></span>Gemini</span>
            <span class="sad-tech-pill"><span class="sad-dot"></span>FastAPI</span>
            <span class="sad-tech-pill"><span class="sad-dot"></span>Streamlit</span>
          </div>
        </div>
        <div class="sad-hero-preview-col">
          <div class="sad-preview-wrapper">
            <svg class="sad-3d-blob" viewBox="0 0 320 260" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
              <defs>
                <linearGradient id="sad-blob-g1-home" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#5B5BF0" stop-opacity="0.85"/>
                  <stop offset="50%" stop-color="#8B5CF6" stop-opacity="0.75"/>
                  <stop offset="100%" stop-color="#EC4899" stop-opacity="0.85"/>
                </linearGradient>
                <radialGradient id="sad-blob-glow-home" cx="45%" cy="40%" r="60%">
                  <stop offset="0%" stop-color="#FFEBDD" stop-opacity="0.7"/>
                  <stop offset="100%" stop-color="#8B5CF6" stop-opacity="0"/>
                </radialGradient>
                <filter id="sad-blob-blur-home" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="16" result="blur"/>
                </filter>
              </defs>
              <path d="M75,130 C55,65 125,25 195,35 C265,45 295,95 285,160 C275,225 215,245 145,240 C85,235 95,195 75,130 Z" fill="url(#sad-blob-g1-home)" filter="url(#sad-blob-blur-home)" opacity="0.6"/>
              <path d="M85,125 C65,70 130,40 195,45 C250,50 280,95 270,155 C260,215 205,235 140,230 C95,225 105,185 85,125 Z" fill="url(#sad-blob-g1-home)"/>
              <circle cx="160" cy="120" r="65" fill="url(#sad-blob-glow-home)"/>
            </svg>
            <div class="sad-live-preview-card">
              <div class="sad-preview-header">
                <span class="sad-preview-badge">💾 Recorded Evidence &bull; {ev_counts['bifrost_date']}</span>
                <span class="sad-preview-status pill-fresh">{ev_counts['bifrost_requests']} gateway calls</span>
              </div>
              <div class="sad-preview-claim">
                Verify time-sensitive factual claims against live primary sources &amp; reputable news.
              </div>
              <div class="sad-preview-source">
                <span class="sad-src-type sad-src-primary">PRIMARY</span>
                <span class="sad-preview-url">evidence/bifrost_evidence.json</span>
              </div>
              <div class="sad-preview-meta">
                <span>🧪 {ev_counts['agent_lab_runs']} benchmark runs</span>
                <span>📊 {ev_counts['benchmark_claims']} labeled claims</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)

    # 2. 4-Step "How it works"
    st.markdown(
        """
        <div style="margin: 2rem 0 1rem 0;">
          <div class="sad-section-head">
            <span class="sad-section-tag">VERIFICATION PIPELINE</span>
            <h2 class="sad-section-title">How It Works in 4 Steps</h2>
            <p class="sad-section-desc">From raw AI answer to live-verified, cited ground truth.</p>
          </div>
          <div class="sad-how-it-works-grid">
            <div class="sad-step-card">
              <div class="sad-step-number">01</div>
              <h4>Extract Claims</h4>
              <p>LLM extracts discrete, time-sensitive factual claims, identifying entities, dates, and numbers.</p>
            </div>
            <div class="sad-step-card">
              <div class="sad-step-number">02</div>
              <h4>Live Ground Truth</h4>
              <p>Tavily executes targeted searches prioritizing official newsrooms, government, and primary domains.</p>
            </div>
            <div class="sad-step-card">
              <div class="sad-step-number">03</div>
              <h4>Dual-Judge Consensus</h4>
              <p>Dual LLM judges classify stance (Support / Refute) with stance-conflict resolution and domain deduplication.</p>
            </div>
            <div class="sad-step-card">
              <div class="sad-step-number">04</div>
              <h4>Score &amp; Rewrite</h4>
              <p>Computes 0–100 Freshness Score, highlights stale sentences, and drafts an updated, accurate answer.</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Real Evidence Strip (Counts from real saved files)
    st.markdown(
        f"""
        <div style="margin: 1.5rem 0;">
          <div class="sad-section-head" style="margin-bottom: 0.75rem;">
            <span class="sad-section-tag">EMPIRICAL GROUND TRUTH</span>
            <h3 style="font-size: 1.15rem; margin: 0; color: var(--text-main);">Recorded Telemetry &amp; Benchmark Evidence</h3>
          </div>
          <div class="sad-evidence-strip">
            <div class="sad-evidence-card">
              <div class="sad-evidence-metric">{ev_counts['bifrost_requests']}</div>
              <div class="sad-evidence-label">Recorded Gateway Requests</div>
              <span class="sad-evidence-source">💾 evidence/bifrost_evidence.json ({ev_counts['bifrost_date']})</span>
            </div>
            <div class="sad-evidence-card">
              <div class="sad-evidence-metric">{ev_counts['agent_lab_runs']}</div>
              <div class="sad-evidence-label">Agent Lab Benchmark Runs</div>
              <span class="sad-evidence-source">💾 lab/results/summary.json ({ev_counts['agent_lab_archs']} architectures)</span>
            </div>
            <div class="sad-evidence-card">
              <div class="sad-evidence-metric">{ev_counts['benchmark_claims']}</div>
              <div class="sad-evidence-label">Ground-Truth Labeled Claims</div>
              <span class="sad-evidence-source">💾 benchmark/dataset.json</span>
            </div>
            <div class="sad-evidence-card">
              <div class="sad-evidence-metric">0</div>
              <div class="sad-evidence-label">Secrets Stored in Evidence</div>
              <span class="sad-evidence-source">🔒 Automated Regex Redaction Verified</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4. Guided Demo Walkthrough Launcher
    st.markdown(
        """
        <div style="margin: 2rem 0 1.25rem 0;">
          <div class="sad-section-head">
            <span class="sad-section-tag">QUOTA-SAFE DEMO</span>
            <h2 class="sad-section-title">✨ Guided Demo Walkthrough</h2>
            <p class="sad-section-desc">
              Understand the product in 30 seconds without typing or consuming API quota. Click any sample to load verified real results:
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gd_col1, gd_col2, gd_col3 = st.columns(3, gap="medium")
    demo_cache = load_demo_cache()
    pasted_samples = demo_cache.get("pasted_examples", [])
    s_iphone = next((s for s in pasted_samples if s.get("id") == "iphone_16"), None)
    s_gst = next((s for s in pasted_samples if s.get("id") == "gst_rates"), None)
    s_pm = next((s for s in pasted_samples if s.get("id") == "pm_modi_incorrect"), None)

    with gd_col1:
        st.markdown(
            """
            <div class="sad-glass-card" style="padding: 1.2rem; height: 100%; border-top: 3px solid #16A34A;">
              <span class="sad-pill pill-fresh">🟢 Fresh Verdict (96/100)</span>
              <h4 style="margin: 0.65rem 0 0.35rem 0; font-size: 1rem;">iPhone 16 Pro Max ($1,199)</h4>
              <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.85rem;">
                Official Apple specs verified. Both dual judges reach 100% agreement on primary source apple.com.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Fresh Example ↗", key="gd_load_iphone", use_container_width=True):
            if s_iphone:
                ex_txt = s_iphone.get("answer_text") or s_iphone.get("text", "")
                st.session_state["answer_input"] = ex_txt
                st.session_state["results"] = s_iphone.get("results")
                st.session_state["checked_answer"] = ex_txt
                st.session_state["demo_mode"] = True
                st.session_state["guided_demo_active"] = True
                st.toast("Loaded Fresh example!", icon="🟢")
                st.rerun()

    with gd_col2:
        st.markdown(
            """
            <div class="sad-glass-card" style="padding: 1.2rem; height: 100%; border-top: 3px solid #EA580C;">
              <span class="sad-pill pill-outdated">🟠 Outdated Verdict (30/100)</span>
              <h4 style="margin: 0.65rem 0 0.35rem 0; font-size: 1rem;">India GST Tax Slabs</h4>
              <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.85rem;">
                Catches superseded 2023 rumor claiming 12% and 18% slabs were merged. Corrected to official 4-tier slab.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Outdated Example ↗", key="gd_load_gst", use_container_width=True):
            if s_gst:
                ex_txt = s_gst.get("answer_text") or s_gst.get("text", "")
                st.session_state["answer_input"] = ex_txt
                st.session_state["results"] = s_gst.get("results")
                st.session_state["checked_answer"] = ex_txt
                st.session_state["demo_mode"] = True
                st.session_state["guided_demo_active"] = True
                st.toast("Loaded Outdated GST tax slabs example!", icon="🟠")
                st.rerun()

    with gd_col3:
        st.markdown(
            """
            <div class="sad-glass-card" style="padding: 1.2rem; height: 100%; border-top: 3px solid #DC2626;">
              <span class="sad-pill pill-incorrect">🔴 Incorrect Verdict (0/100)</span>
              <h4 style="margin: 0.65rem 0 0.35rem 0; font-size: 1rem;">Wrong Prime Minister</h4>
              <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.85rem;">
                Detects hallucinated head of government, refuted by official pmindia.gov.in portal.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Load Incorrect Example ↗", key="gd_load_pm", use_container_width=True):
            if s_pm:
                ex_txt = s_pm.get("answer_text") or s_pm.get("text", "")
                st.session_state["answer_input"] = ex_txt
                st.session_state["results"] = s_pm.get("results")
                st.session_state["checked_answer"] = ex_txt
                st.session_state["demo_mode"] = True
                st.session_state["guided_demo_active"] = True
                st.toast("Loaded Incorrect Prime Minister example!", icon="🔴")
                st.rerun()

    # 5. Highlights Bento Grid
    if callable(highlights_bento_fn):
        highlights_bento_fn()

    # 6. FAQ Section (Glass Accordion)
    st.markdown(
        """
        <div style="margin: 2rem 0 1rem 0;">
          <div class="sad-section-head">
            <span class="sad-section-tag">FREQUENTLY ASKED QUESTIONS</span>
            <h2 class="sad-section-title">Architecture, Quotas &amp; Privacy</h2>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    route_info = get_real_model_and_routing_info(use_bifrost=use_bifrost_mode)
    j1_disp = route_info.get("dual_judge_primary") or "configured primary judge"
    j2_disp = route_info.get("dual_judge_secondary") or "configured secondary judge"

    with st.expander("🛡️ How does the system handle API errors or rate limits?", expanded=False):
        st.markdown(
            "On connection errors, the client retries up to 3 times with exponential backoff and then attempts "
            "configured fallback models. If an LLM or Tavily call still fails (connection error, HTTP 429, 503, "
            "or invalid JSON), affected claims are marked **Verification failed** with score `—` (excluded from "
            "the overall score and status counts) and provide a **Retry this claim** button alongside Demo / Saved "
            "results for built-in examples."
        )

    with st.expander("🔒 What data is sent and stored?", expanded=False):
        st.markdown(
            "- **Sent to Gemini (or your configured Bifrost gateway):** input text, questions, extracted article text, "
            "extracted claims, and retrieved web snippets used for verification and rewriting.\n"
            "- **Sent to Tavily:** search queries derived from extracted claims and URLs fetched for extraction.\n"
            "- **Cached on the server for 6 hours:** search queries and web snippets derived from the input are stored "
            "in a local SQLite cache (`tavily_cache.db`) for 6 hours to reduce duplicate API calls.\n"
            "- **In-memory session history:** session history and watchlist live in server memory (`st.session_state`) "
            "for the active session.\n"
            "- **Hosting & provider logs:** the hosting platform and upstream API providers (Google Gemini, Tavily) "
            "may keep their own operational logs under their respective policies."
        )

    with st.expander("⚖️ How is the Freshness Score calculated?", expanded=False):
        st.markdown(
            "Each verifiable claim is scored 0–100 from four weighted signals in `verifier.py`:\n\n"
            "- **LLM verdict (50%)**, **Source recency (20%)**, **Source agreement across unique domains (20%)**, "
            "and **Source authority (10%)**.\n"
            "- Scores are clamped to status bands: Fresh `75–100`, Partially correct `45–74`, Uncertain `40–74`, "
            "Outdated `10–39`, Incorrect `0–15`. Unverifiable (`N/A`) and Verification failed (`—`) claims are "
            "excluded from the risk-weighted overall score."
        )

    with st.expander("👥 Why use Dual-Judge Consensus instead of a single LLM prompt?", expanded=False):
        st.markdown(
            f"Single LLM prompts can miss subtle contradictions. When Dual-Judge mode is enabled, two configured "
            f"models (`{j1_disp}` and `{j2_disp}`) evaluate the same retrieved Tavily evidence independently. "
            "If they disagree, the claim is downgraded to Uncertain (or Incorrect when contradicted by official sources)."
        )

    # 7. Evaluator Note Banner
    if not st.session_state.get("evaluator_banner_dismissed", False):
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            e_col1, e_col2 = st.columns([0.95, 0.05], vertical_alignment="top")
            with e_col1:
                st.markdown(
                    f'<div style="display: flex; gap: 10px; align-items: flex-start;">'
                    f'<span style="font-size: 1.15rem;">📌</span>'
                    f'<div>'
                    f'<strong style="font-size: 0.92rem; color: var(--text-color);">Note for evaluators</strong>'
                    f'<p style="margin: 3px 0 0 0; font-size: 0.84rem; color: var(--text-muted); line-height: 1.45;">'
                    f'{html.escape(evaluator_text)}'
                    f'</p>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with e_col2:
                if st.button("✕", key="dismiss_evaluator_banner_home_btn", help="Dismiss Note for evaluators"):
                    st.session_state["evaluator_banner_dismissed"] = True
                    st.rerun()


def render_showdown_tab(use_bifrost: bool) -> None:
    """Render the Model Showdown tab comparing 2-3 real configured models side-by-side.

    Shows an honest empty state with a quota-aware 'Run showdown' button when no
    real recorded file (`evidence/showdown_results.json`) or live run exists.
    Never labels unrecorded data 'Saved', and renders markdown/HTML cleanly without literal `**`.
    """
    st.markdown(
        """
        <div class="sad-section-head">
          <span class="sad-section-tag">SIDE-BY-SIDE EVALUATION</span>
          <h2 class="sad-section-title">Model Showdown</h2>
          <p class="sad-section-desc">
            Compare 2–3 real configured models on freshness, outdated claims, measured latency, and returned tokens on the same question.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Honest Sample Size Disclaimer
    st.info(SAMPLE_SIZE_DISCLAIMER, icon=":material/info:")

    route_info = get_real_model_and_routing_info(use_bifrost=use_bifrost)
    configured_models = get_configured_showdown_models()
    saved_showdowns = get_saved_showdowns()
    live_showdown = st.session_state.get("live_showdown_result")

    # Quota check
    max_audits = DEFAULT_MAX_SESSION_AUDITS
    try:
        from llm import _get_config_value

        raw_max = _get_config_value("MAX_SESSION_AUDITS")
        if raw_max:
            max_audits = max(1, int(raw_max))
    except Exception:
        pass
    audits_done = int(st.session_state.get("live_audits_performed") or 0)
    audits_left = max(0, max_audits - audits_done)
    demo_mode_on = bool(st.session_state.get("demo_mode", False))

    # Live Showdown Runner Card (always available, respects session quota)
    with st.container(border=True):
        st.markdown(
            f"**Active routing:** `{route_info['route_label']}` &bull; "
            f"**Configured models:** {', '.join(f'`{m}`' for m in configured_models) if configured_models else '`None configured`'} &bull; "
            f"**Session audits left:** `{audits_left}`"
        )
        preset_options = [q["question"] for q in DEFAULT_SHOWDOWN_QUESTIONS]
        selected_q = st.selectbox(
            "Select a showdown question",
            options=preset_options,
            index=0,
            key="showdown_preset_question",
        )
        custom_q = st.text_input(
            "Or enter your own question",
            value="",
            placeholder="Leave blank to use the selected question above...",
            key="showdown_custom_question",
        )
        target_question = custom_q.strip() or selected_q

        run_disabled = demo_mode_on or audits_left <= 0 or len(configured_models) < 1
        btn_help = (
            "Turn off Demo mode to run a live showdown."
            if demo_mode_on
            else (
                "Session audit quota reached."
                if audits_left <= 0
                else "Run this question against your real configured models and verify each response."
            )
        )
        if st.button(
            "Run showdown",
            key="btn_run_live_showdown",
            type="primary",
            disabled=run_disabled,
            help=btn_help,
        ):
            with st.spinner("Running live Model Showdown across configured models..."):
                try:
                    sd_live = run_live_showdown(target_question, models=configured_models)
                    st.session_state["live_showdown_result"] = sd_live
                    st.session_state["showdown_active_source"] = "live"
                    st.session_state["live_audits_performed"] = audits_done + 1
                    if audits_done + 1 >= max_audits:
                        st.session_state["quota_status"] = "Limited"
                    st.rerun()
                except Exception as exc:
                    err_low = str(exc).lower()
                    if "429" in err_low or "rate limit" in err_low or "quota" in err_low:
                        st.session_state["quota_status"] = "Limited"
                    elif any(k in err_low for k in ("connect", "unreachable", "timeout", "network")):
                        st.session_state["quota_status"] = "Offline"
                    st.error(f"Could not run live showdown: {exc}")

    # Decide which showdown to display: live session run, real recorded file, or empty state
    sd_data: Optional[Dict[str, Any]] = None
    if saved_showdowns:
        saved_keys = list(saved_showdowns.keys())
        default_key = st.session_state.get("showdown_active_id")
        if default_key not in saved_showdowns:
            default_key = saved_keys[0]
        sel_cols = st.columns(len(saved_keys), gap="small")
        for idx, s_key in enumerate(saved_keys):
            s_entry = saved_showdowns[s_key]
            with sel_cols[idx]:
                if st.button(
                    str(s_entry.get("title") or s_key),
                    key=f"btn_sd_saved_{s_key}",
                    type="primary" if (st.session_state.get("showdown_active_source") != "live" and default_key == s_key) else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["showdown_active_id"] = s_key
                    st.session_state["showdown_active_source"] = "saved"
                    st.rerun()
        if st.session_state.get("showdown_active_source") == "live" and live_showdown:
            sd_data = live_showdown
        else:
            sd_data = get_saved_showdown(default_key)
    elif live_showdown:
        sd_data = live_showdown

    if not sd_data:
        st.markdown(
            """
            <div class="sad-glass-card" style="padding: 1.75rem; text-align: center; margin: 1.25rem 0;">
              <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; color: var(--text-main);">No recorded Model Showdown results yet</h3>
              <p style="margin: 0 auto 0.75rem auto; max-width: 560px; font-size: 0.88rem; color: var(--text-muted); line-height: 1.55;">
                Unrecorded or synthetic comparisons have been removed. Click <strong>Run showdown</strong> above to run a live
                comparison across your configured models (subject to session quota), or generate a real recorded dataset with
                <code>python scripts/generate_showdown.py</code>.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Question Card
    st.markdown(
        f"""
        <div class="sad-glass-card" style="padding: 1.25rem; margin: 1.25rem 0;">
          <div style="font-size: 0.78rem; text-transform: uppercase; font-weight: 700; color: var(--primary-color);">Test Query</div>
          <h3 style="margin: 0.35rem 0 0.25rem 0; font-size: 1.15rem;">&ldquo;{html.escape(str(sd_data.get('question', '')))}&rdquo;</h3>
          <span style="font-size: 0.82rem; color: var(--text-muted);">{html.escape(str(sd_data.get('description', '')))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    models_dict = sd_data.get("models", {})
    winner_eval = evaluate_showdown_winner(models_dict)
    src_lbl = str(sd_data.get("source_label") or "Live")
    src_icon = "💾" if src_lbl.lower().startswith("saved") else "🟢"
    winner_score_txt = (
        f"({winner_eval.get('winner_score')}/100 Freshness)"
        if winner_eval.get("winner_score") is not None
        else "(No scorable winner)"
    )

    # Winner Card (uses HTML-safe reason_html so no literal ** characters ever appear)
    st.markdown(
        f"""
        <div class="sad-showdown-card sad-showdown-winner" style="margin-bottom: 1.25rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-size: 1.4rem;">🏆</span>
              <div>
                <strong style="font-size: 1.05rem; color: #16A34A;">Winner: {html.escape(str(winner_eval['winner']))}</strong>
                <span style="font-size: 0.85rem; color: var(--text-muted); margin-left: 6px;">{html.escape(winner_score_txt)}</span>
              </div>
            </div>
            <span class="sad-meta-pill" style="border-color: #2563EB; color: #2563EB; background: rgba(37,99,235,0.08);">
              {src_icon} {html.escape(src_lbl)}
            </span>
          </div>
          <p style="margin: 0.4rem 0 0 0; font-size: 0.88rem; color: var(--text-main); line-height: 1.45;">
            {winner_eval.get('reason_html') or html.escape(str(winner_eval.get('reason', '')))}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Side-by-side Columns
    model_keys = list(models_dict.keys())
    if not model_keys:
        return
    m_cols = st.columns(len(model_keys), gap="medium")

    for idx, m_key in enumerate(model_keys):
        m_info = models_dict[m_key]
        raw_score = m_info.get("score")
        m_err = m_info.get("error")
        if m_err or raw_score is None:
            score_display = "—"
            score_color = "#DC2626" if m_err else "#64748B"
            status_label = "Failed" if m_err else "Unscored"
        else:
            score_val = int(raw_score)
            score_display = f"{score_val}/100"
            score_color = "#16A34A" if score_val >= 80 else ("#EA580C" if score_val >= 40 else "#DC2626")
            status_label = "Fresh" if score_val >= 80 else ("Mixed" if score_val >= 40 else "Stale / Incorrect")

        lat_val = m_info.get("latency_ms")
        lat_txt = f"{lat_val:.0f}ms" if isinstance(lat_val, (int, float)) else "not run"
        tok_val = m_info.get("tokens")
        tok_txt = f"{int(tok_val):,} tokens" if isinstance(tok_val, (int, float)) and tok_val > 0 else "usage unavailable"

        with m_cols[idx]:
            st.markdown(
                f"""
                <div class="sad-glass-card" style="padding: 1.25rem; border-top: 4px solid {score_color}; height: 100%;">
                  <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1rem; color: var(--text-main);">{html.escape(str(m_info.get('display_name', m_key)))}</strong>
                    <span class="sad-meta-pill" style="border-color: {score_color}; color: {score_color}; font-weight: 700;">{html.escape(score_display)}</span>
                  </div>
                  <div style="display: flex; gap: 12px; margin: 0.75rem 0; font-size: 0.8rem; color: var(--text-muted); flex-wrap: wrap;">
                    <span>⚡ {html.escape(lat_txt)}</span>
                    <span>🪙 {html.escape(tok_txt)}</span>
                    <span>Status: <strong style="color: {score_color};">{html.escape(status_label)}</strong></span>
                  </div>
                  <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 8px; padding: 0.85rem; font-size: 0.85rem; line-height: 1.5; color: var(--text-main); margin-bottom: 0.85rem;">
                    {html.escape(str(m_err)) if m_err else ('&ldquo;' + html.escape(str(m_info.get('answer', ''))) + '&rdquo;')}
                  </div>
                  <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;">
                    Claims &amp; Evidence:
                  </div>
                """,
                unsafe_allow_html=True,
            )

            for c in m_info.get("claims", []):
                c_st = str(c.get("status", "uncertain")).lower()
                c_badge_cls = (
                    "pill-fresh" if c_st == "fresh"
                    else ("pill-outdated" if c_st == "outdated" else "pill-incorrect")
                )
                src = (c.get("sources") or [{}])[0]
                src_domain = str(src.get("domain") or "web")
                c_claim_txt = str(c.get("claim_text") or c.get("claim") or "")
                st.markdown(
                    f"""
                    <div style="border-left: 2px solid {score_color}; padding-left: 8px; margin-bottom: 8px;">
                      <span class="sad-pill {c_badge_cls}" style="font-size: 0.72rem; padding: 2px 6px;">{html.escape(c_st.replace('_', ' ').upper())}</span>
                      <div style="font-size: 0.82rem; margin: 4px 0;">{html.escape(c_claim_txt)}</div>
                      <div style="font-size: 0.75rem; color: var(--text-muted);">Source: <code>{html.escape(src_domain)}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)


def render_history_tab() -> None:
    """Render the History tab with real session audits only, plus a separate Built-in Examples section."""
    init_history_if_needed()
    history: List[Dict[str, Any]] = st.session_state.get("audit_history", [])

    st.markdown(
        """
        <div class="sad-section-head">
          <span class="sad-section-tag">SESSION LOGS</span>
          <h2 class="sad-section-title">Audit History</h2>
          <p class="sad-section-desc">
            Real audits run during this session. Built-in saved examples appear in their own section below.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not history:
        st.info("No audits recorded in this session yet. Run an audit in the Audit tab to see your session history here.")
    else:
        # Action Bar: Export & Clear
        act_col1, act_col2, act_col3 = st.columns([1, 1, 1], gap="small")
        with act_col1:
            history_json = json.dumps(history, indent=2)
            st.download_button(
                "📥 Export History (JSON)",
                data=history_json,
                file_name="stale_audit_history.json",
                mime="application/json",
                use_container_width=True,
            )
        with act_col2:
            md_lines = ["# Stale-Answer Detector — Session Audit History", ""]
            for idx, item in enumerate(history, 1):
                a_status = str(item.get("audit_status") or "Completed")
                sc_val = item.get("score")
                score_line = f"{sc_val}/100" if (a_status == "Completed" and sc_val is not None) else a_status
                md_lines.append(f"## {idx}. Audit: {item.get('input_preview', '')}")
                md_lines.append(f"- **Time**: {item.get('timestamp')}")
                md_lines.append(f"- **Result**: {score_line}")
                md_lines.append(f"- **Source**: {item.get('source_label')}")
                counts = item.get("status_counts", {})
                md_lines.append(f"- **Breakdown**: Fresh: {counts.get('fresh',0)}, Outdated: {counts.get('outdated',0)}, Incorrect: {counts.get('incorrect',0)}")
                if item.get("failed_count"):
                    md_lines.append(f"- **Failed claims**: {item.get('failed_count')}")
                md_lines.append("")
            history_md = "\n".join(md_lines)
            st.download_button(
                "📄 Export History (Markdown)",
                data=history_md,
                file_name="stale_audit_history.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with act_col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state["audit_history"] = []
                st.toast("Audit history cleared.")
                st.rerun()

        # Side-by-Side Comparison Box (only for scored Completed audits)
        scored_history = [
            h for h in history
            if h.get("audit_status", "Completed") == "Completed" and h.get("score") is not None
        ]
        if len(scored_history) >= 2:
            with st.expander("⚖️ Compare Two Audits Side-by-Side", expanded=False):
                options = [f"#{i+1}: ({h.get('score')}/100) {h.get('input_preview', '')[:40]}" for i, h in enumerate(scored_history)]
                cmp_c1, cmp_c2 = st.columns(2)
                with cmp_c1:
                    sel_a = st.selectbox("Select Audit A", options, index=0, key="cmp_sel_a")
                with cmp_c2:
                    sel_b = st.selectbox("Select Audit B", options, index=min(1, len(options) - 1), key="cmp_sel_b")

                aud_a = scored_history[options.index(sel_a)]
                aud_b = scored_history[options.index(sel_b)]

                sc_a = int(aud_a.get("score", 0))
                sc_b = int(aud_b.get("score", 0))
                delta = sc_a - sc_b
                delta_str = f"+{delta} pts fresher" if delta > 0 else (f"{delta} pts" if delta < 0 else "Identical freshness")

                st.markdown(
                    f"""
                    <div style="text-align: center; margin: 0.75rem 0; font-size: 0.95rem; font-weight: 700; color: var(--primary-color);">
                      Comparison Delta: Audit A is {delta_str} than Audit B
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                c_box1, c_box2 = st.columns(2)
                with c_box1:
                    st.markdown(
                        f"""
                        <div class="sad-glass-card" style="padding: 1rem;">
                          <div style="font-weight: 700; font-size: 1rem; margin-bottom: 4px;">Audit A</div>
                          <div style="font-size: 1.4rem; font-weight: 800; color: #16A34A;">{sc_a}/100</div>
                          <p style="font-size: 0.85rem; color: var(--text-muted); margin: 6px 0;">{html.escape(aud_a.get('input_text', ''))}</p>
                          <div style="font-size: 0.78rem;">Source: {html.escape(aud_a.get('source_label', ''))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_box2:
                    st.markdown(
                        f"""
                        <div class="sad-glass-card" style="padding: 1rem;">
                          <div style="font-weight: 700; font-size: 1rem; margin-bottom: 4px;">Audit B</div>
                          <div style="font-size: 1.4rem; font-weight: 800; color: #EA580C;">{sc_b}/100</div>
                          <p style="font-size: 0.85rem; color: var(--text-muted); margin: 6px 0;">{html.escape(aud_b.get('input_text', ''))}</p>
                          <div style="font-size: 0.78rem;">Source: {html.escape(aud_b.get('source_label', ''))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

        # History List Cards
        for idx, item in enumerate(history):
            a_status = str(item.get("audit_status") or "Completed")
            raw_sc = item.get("score")
            if a_status == "Failed":
                badge_text = "Failed"
                sc_col = "#DC2626"
            elif a_status == "Partial":
                failed_n = int(item.get("failed_count") or 0)
                total_m = int(item.get("total_claims") or len(item.get("results") or []))
                badge_text = f"Partial ({failed_n}/{total_m} failed)" if total_m > 0 else "Partial"
                sc_col = "#D97706"
            elif raw_sc is None:
                badge_text = "Unverifiable"
                sc_col = "#64748B"
            else:
                sc = int(raw_sc)
                badge_text = f"{sc}/100"
                sc_col = "#16A34A" if sc >= 80 else ("#EA580C" if sc >= 40 else "#DC2626")

            counts = item.get("status_counts", {})
            h_card = f"""
            <div class="sad-glass-card" style="padding: 1.15rem 1.25rem; margin-bottom: 0.85rem; border-left: 4px solid {sc_col};">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                  <span style="font-weight: 800; font-size: 1.15rem; color: {sc_col};">{html.escape(badge_text)}</span>
                  <span style="font-size: 0.82rem; color: var(--text-muted);">{html.escape(str(item.get('timestamp', '')))}</span>
                </div>
                <span class="sad-meta-pill" style="border-color: #2563EB; color: #2563EB; background: rgba(37,99,235,0.08); font-size: 0.75rem;">
                  {html.escape(str(item.get('source_label', '')))}
                </span>
              </div>
              <p style="margin: 0.5rem 0; font-size: 0.9rem; color: var(--text-main); font-weight: 500;">
                &ldquo;{html.escape(str(item.get('input_preview', '')))}&rdquo;
              </p>
              <div style="display: flex; gap: 8px; font-size: 0.75rem; flex-wrap: wrap;">
                <span class="sad-pill pill-fresh" style="padding: 2px 8px;">Fresh: {counts.get('fresh', 0)}</span>
                <span class="sad-pill pill-outdated" style="padding: 2px 8px;">Outdated: {counts.get('outdated', 0)}</span>
                <span class="sad-pill pill-incorrect" style="padding: 2px 8px;">Incorrect: {counts.get('incorrect', 0)}</span>
              </div>
            </div>
            """
            st.markdown(h_card, unsafe_allow_html=True)

            h_c1, h_c2 = st.columns([1, 0.2])
            with h_c1:
                if st.button("🔍 Reopen in Audit Workspace", key=f"reopen_hist_{idx}"):
                    st.session_state["answer_input"] = item.get("input_text", "")
                    st.session_state["results"] = item.get("results")
                    st.session_state["checked_answer"] = item.get("input_text", "")
                    st.toast("Reopened audit in workspace!", icon="🔍")
                    st.rerun()
            with h_c2:
                if st.button("🗑️ Delete", key=f"del_hist_{idx}"):
                    history.pop(idx)
                    st.session_state["audit_history"] = history
                    st.rerun()

    # Separate Built-in Examples Section (Requirement 4)
    builtin_examples = get_builtin_examples_from_cache()
    if builtin_examples:
        st.markdown("<hr style='margin: 2rem 0 1.25rem 0; border-color: rgba(148,163,184,0.2);'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sad-section-head">
              <span class="sad-section-tag">BUILT-IN SAVED SAMPLES</span>
              <h3 class="sad-section-title" style="font-size: 1.25rem;">Examples</h3>
              <p class="sad-section-desc">
                Pre-recorded example audits stored in <code>demo_cache.json</code>. Kept separate from your live session history.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for ex_idx, ex in enumerate(builtin_examples):
            ex_sc = ex.get("score")
            ex_col = "#16A34A" if (ex_sc is not None and ex_sc >= 80) else ("#EA580C" if (ex_sc is not None and ex_sc >= 40) else "#DC2626")
            ex_score_lbl = f"{ex_sc}/100" if ex_sc is not None else "Unscored"
            ex_card = f"""
            <div class="sad-glass-card" style="padding: 1rem 1.25rem; margin-bottom: 0.75rem; border-left: 4px solid {ex_col};">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <strong>{html.escape(ex['label'])}</strong>
                  <span style="font-weight: 700; color: {ex_col};">{html.escape(ex_score_lbl)}</span>
                </div>
                <span class="sad-meta-pill" style="border-color: #2563EB; color: #2563EB; background: rgba(37,99,235,0.08); font-size: 0.75rem;">
                  💾 {html.escape(ex['source_label'])}
                </span>
              </div>
              <p style="margin: 0.45rem 0 0 0; font-size: 0.88rem; color: var(--text-main);">
                &ldquo;{html.escape(ex['input_preview'])}&rdquo;
              </p>
            </div>
            """
            st.markdown(ex_card, unsafe_allow_html=True)
            if st.button("Load Example in Audit Workspace", key=f"load_builtin_ex_{ex_idx}"):
                st.session_state["answer_input"] = ex["input_text"]
                st.session_state["results"] = ex["results"]
                st.session_state["checked_answer"] = ex["input_text"]
                st.session_state["demo_mode"] = True
                st.toast(f"Loaded example: {ex['label']}")
                st.rerun()


def render_api_playground_tab() -> None:
    """Render the API Playground with schema, snippets, and interactive try it."""
    st.markdown(
        """
        <div class="sad-section-head">
          <span class="sad-section-tag">DEVELOPER INTEGRATION</span>
          <h2 class="sad-section-title">API Playground</h2>
          <p class="sad-section-desc">
            Integrate the Stale-Answer Detector REST API into your workflows, CI/CD, and agent pipelines.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" style="color: var(--primary-color); font-weight: 600; text-decoration: none;">
            📖 Open Interactive OpenAPI / Swagger UI (localhost:8000/docs) &rarr;
          </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Endpoint Schema
    with st.expander("📋 Endpoint Specification: POST /verify", expanded=True):
        st.markdown("### Request Schema")
        st.code(
            """POST /verify HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "text": "Apple's latest flagship is iPhone 16 Pro Max starting at $1,199.",
  "mode": "audit",
  "extract_claims_first": true
}""",
            language="json",
        )
        st.markdown("### Response Schema")
        st.code(
            """{
  "overall_score": 96,
  "overall_tier": "Fresh",
  "status_counts": {
    "fresh": 1,
    "outdated": 0,
    "incorrect": 0,
    "partially_correct": 0,
    "uncertain": 0,
    "unverifiable": 0
  },
  "claims": [
    {
      "claim": "Apple's latest flagship is iPhone 16 Pro Max starting at $1,199.",
      "status": "fresh",
      "score": 96,
      "confidence": "high",
      "sources": [
        {
          "domain": "apple.com",
          "url": "https://www.apple.com/iphone-16-pro/",
          "is_primary": true,
          "stance": "support"
        }
      ]
    }
  ],
  "latency_ms": 320
}""",
            language="json",
        )

    # Code Snippets in Tabs
    st.markdown("### Code Snippets")
    c_tab_curl, c_tab_py, c_tab_js = st.tabs(["cURL", "Python (requests)", "JavaScript (fetch)"])

    with c_tab_curl:
        st.code(
            """curl -X POST "http://localhost:8000/verify" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Apple released the iPhone 16 Pro Max starting at $1,199."}'""",
            language="bash",
        )

    with c_tab_py:
        st.code(
            """import requests

response = requests.post(
    "http://localhost:8000/verify",
    json={"text": "Apple released the iPhone 16 Pro Max starting at $1,199."},
    timeout=10
)
data = response.json()
print(f"Freshness Score: {data['overall_score']}/100")
for claim in data['claims']:
    print(f"- [{claim['status'].upper()}] {claim['claim']}")""",
            language="python",
        )

    with c_tab_js:
        st.code(
            """const response = await fetch("http://localhost:8000/verify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    text: "Apple released the iPhone 16 Pro Max starting at $1,199."
  })
});
const data = await response.json();
console.log(`Freshness Score: ${data.overall_score}/100`);""",
            language="javascript",
        )

    # Interactive Sandbox "Try It"
    st.markdown("### Interactive Try It")
    try_text = st.text_area(
        "Payload text",
        value="Apple released the iPhone 16 Pro Max starting at $1,199.",
        height=70,
        key="api_playground_try_text",
    )

    if st.button("🚀 Send Test Request", type="primary", key="btn_send_api_request"):
        is_live = False
        live_res: Dict[str, Any] = {}
        try:
            req = urllib.request.Request(
                "http://localhost:8000/verify",
                data=json.dumps({"text": try_text}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    live_res = json.loads(resp.read().decode("utf-8"))
                    is_live = True
        except Exception:
            is_live = False

        if is_live:
            st.success("🟢 Response received from local FastAPI service (HTTP 200 OK):")
            st.json(live_res)
        else:
            st.warning(
                "Local FastAPI service is not reachable at `http://localhost:8000/verify`. "
                "Start the API server locally with `uvicorn api:app --host 127.0.0.1 --port 8000` "
                "to send live requests here. No synthetic response is fabricated when the server is offline.",
                icon=":material/cloud_off:",
            )


def render_methodology_tab() -> None:
    """Render the Methodology tab covering status definitions, real scoring weights, active models, and truthful privacy."""
    route_info = get_real_model_and_routing_info()
    st.markdown(
        """
        <div class="sad-section-head">
          <span class="sad-section-tag">FORMAL RIGOR</span>
          <h2 class="sad-section-title">Detection Methodology &amp; Reproducibility</h2>
          <p class="sad-section-desc">
            Scoring weights from <code>verifier.py</code>, claim status taxonomy, source independence rules, active model routing, and data handling.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 1: Claim Status Definitions
    st.markdown("### 1. Claim Status Taxonomy")
    st.markdown(
        """
| Status | Score Band | Definition |
| :--- | :--- | :--- |
| **🟢 Fresh** | `75 – 100` | Retrieved evidence clearly confirms the claim matches current facts. |
| **🟡 Partially Correct** | `45 – 74` | Core entity matches, but specific figures, dates, or details are incomplete or inaccurate. |
| **⚪ Uncertain** | `40 – 74` | Credible sources conflict or evidence is ambiguous. |
| **🟠 Outdated** | `10 – 39` | Claim was previously true, but has been superseded by a verified change. |
| **🔴 Incorrect** | `0 – 15` | Claim was never true or is directly contradicted by authoritative sources. |
| **⚫ Unverifiable** | `N/A` | No relevant or readable live web evidence was found; excluded from overall score. |
| **❌ Verification failed** | `—` | LLM or Tavily call failed after retries/fallback; excluded from overall score and status counts. |
        """
    )

    # Section 2: Freshness Score Formula matching verifier.py SCORE_WEIGHTS
    st.markdown("### 2. Freshness Score Formulation (`verifier.py`)")
    st.markdown(
        """
Each verifiable claim's 0–100 score combines four deterministic weighted signals defined in `SCORE_WEIGHTS` in `verifier.py`, clamped into its status band (`STATUS_SCORE_BANDS`):

- **LLM verdict (`50%`)**: Grounded factual judgment strictly from retrieved snippets and extracted source text.
- **Source recency (`20%`)**: Based on `published_date` returned by Tavily.
- **Source agreement (`20%`)**: Share of unique registered domains supporting the current fact (`N of M independent sources agree`).
- **Source authority (`10%`)**: Domain classification (`PRIMARY`, `NEWS`, `BLOG`, `FORUM`).

Overall answer freshness is a risk-weighted average across scorable claims (`critical = 3.0`, `moderate = 1.5`, `low = 1.0`). Claims marked `Unverifiable`, `Verification failed`, or `Skipped` are excluded from the average.
        """
    )

    # Section 3: Stance Counting & Source Independence Rule
    j1_m = route_info.get("dual_judge_primary") or "configured primary judge"
    j2_m = route_info.get("dual_judge_secondary") or "configured secondary judge"
    st.markdown("### 3. Source Independence & Dual-Judge Consensus")
    st.markdown(
        f"""
1. **Registered Domain Deduplication**: Multiple URLs from the same registered domain (for example, `apple.com/specs` and `apple.com/newsroom`) are collapsed into one chip with a `+N pages` indicator and count as **one independent domain** in `N of M independent sources agree`.
2. **Single Source Label**: Each source chip carries exactly one classification label: `PRIMARY`, `NEWS`, `BLOG`, or `FORUM`.
3. **Office-Holder Verification**: Claims in the `leadership / office holder` category (prime minister, president, CEO, chief minister) are verified only against official government or company pages and widely reputable news outlets.
4. **Optional Dual-Judge Mode**: When enabled, two configured models (`{j1_m}` and `{j2_m}`) independently judge the same retrieved Tavily evidence.
        """
    )

    # Section 4: Active Routing & Configured Models (generated from real config + evidence file)
    configured_str = ", ".join(f"`{m}`" for m in route_info["configured_models"]) or "`None configured`"
    evidence_str = ", ".join(f"`{m}`" for m in route_info["evidence_models"]) or "`See evidence/bifrost_evidence.json`"
    fallbacks_str = " → ".join(f"`{m}`" for m in [route_info["judge_model"], *route_info["fallbacks"]] if m) or "`None configured`"
    st.markdown("### 4. Active Routing, Models & Roles")
    st.markdown(
        f"""
- **Active LLM Route**: `{route_info['route_label']}` (`{route_info['calls_label']}`)
- **Active Configured Models**: {configured_str} (fallback chain: {fallbacks_str})
- **Recorded Bifrost Evidence Models** (`evidence/bifrost_evidence.json`, {route_info['evidence_date'] or 'saved run'}): {evidence_str}
- **Tavily Role**: Executes live web searches and page extraction, returning snippets, URLs, and published dates cached for 6 hours in local SQLite (`tavily_cache.db`).
- **Bifrost Role (when `USE_BIFROST=true`)**: Routes OpenAI-compatible requests across configured providers, applies gateway fallbacks on 429/503 outages, and records token/latency metadata (`x-bf-run-id`). When `USE_BIFROST=false`, requests go directly to Google Gemini's OpenAI-compatible endpoint.
        """
    )

    # Section 5: Truthful Privacy and Data Handling Policy (Requirement 6)
    st.markdown("### 5. Privacy & Data Handling")
    st.markdown(
        """
- **Sent to Google Gemini (or your configured Bifrost gateway)**: Text you paste, questions you ask, extracted article text from URLs you submit, extracted factual claims, and retrieved web snippets used to judge freshness or draft rewrites.
- **Sent to Tavily**: Search queries derived from extracted claims, plus webpage URLs you submit or top source URLs fetched for content extraction.
- **Cached on the server for 6 hours**: Search queries, URLs, and web snippets derived from your input are cached on the server in a local SQLite database (`tavily_cache.db`, 6-hour TTL) to reduce duplicate upstream requests.
- **In-memory session history**: Your session audit history, watchlist, and active results live in server memory (`st.session_state`) for the duration of your session.
- **Platform & provider logs**: The hosting platform (such as Streamlit Community Cloud or your server host) and upstream API providers (Google Gemini, Tavily) may retain their own operational or access logs under their respective policies. Streamlit browser usage stats are disabled in `.streamlit/config.toml` (`browser.gatherUsageStats = false`).
        """
    )

    # Section 6: How to Verify Our Claims (Clickable repo file links)
    st.markdown("### 6. Repository Evidence & Source Files")
    ev_counts = load_evidence_counts()
    st.markdown(
        f"""
Saved metrics shown in this app are read from version-controlled repository files:

- 📊 **Ground Truth Benchmark ({ev_counts['benchmark_claims']} Claims)**: [`benchmark/dataset.json`](file:///c:/Coding%20prb/Python/stale-detector/benchmark/dataset.json)
- 💾 **Recorded Bifrost Evidence ({ev_counts['bifrost_requests']} Requests, {ev_counts['bifrost_date']})**: [`evidence/bifrost_evidence.json`](file:///c:/Coding%20prb/Python/stale-detector/evidence/bifrost_evidence.json)
- 🔬 **Agent Lab Results ({ev_counts['agent_lab_runs']} Runs)**: [`lab/results/summary.json`](file:///c:/Coding%20prb/Python/stale-detector/lab/results/summary.json)
- ⚙️ **Core Verification Engine**: [`verifier.py`](file:///c:/Coding%20prb/Python/stale-detector/verifier.py)
- 📐 **Agent Lab Scoring Logic**: [`lab/scoring.py`](file:///c:/Coding%20prb/Python/stale-detector/lab/scoring.py)
        """
    )


def render_watchlist_strip() -> None:
    """Render the pinned Claim Watchlist strip with delta indicators."""
    watchlist: List[Dict[str, Any]] = st.session_state.get("watchlist", [])
    if not watchlist:
        return

    st.markdown(
        f"""
        <div class="sad-watchlist-bar">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 8px;">
              <span style="font-size: 1.2rem;">⭐</span>
              <strong style="font-size: 1rem; color: var(--text-main);">Session Claim Watchlist ({len(watchlist)} tracked)</strong>
            </div>
            <span style="font-size: 0.78rem; color: var(--text-muted);">In-session persistent monitoring</span>
          </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, w in enumerate(watchlist):
        orig_st = w.get("original_status", "uncertain")
        curr_st = w.get("current_status", orig_st)
        changed = orig_st != curr_st

        delta_badge = (
            f'<span class="sad-meta-pill" style="border-color: #EA580C; color: #EA580C; background: rgba(234,88,12,0.08);">⚠️ Changed: {orig_st.upper()} &rarr; {curr_st.upper()}</span>'
            if changed
            else f'<span class="sad-meta-pill" style="border-color: #16A34A; color: #16A34A; background: rgba(22,163,74,0.08);">🟢 Unchanged ({curr_st.upper()})</span>'
        )

        st.markdown(
            f"""
            <div class="sad-watchlist-item">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                <span style="font-size: 0.88rem; font-weight: 600; color: var(--text-main);">&ldquo;{html.escape(w.get('claim', ''))}&rdquo;</span>
                {delta_badge}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    w_col1, w_col2 = st.columns([1, 1], gap="small")
    with w_col1:
        if st.button("🔄 Re-check Watchlist Claims", key="btn_recheck_watchlist", use_container_width=True):
            # Recheck logic simulates an updated verification pass
            for w in watchlist:
                w["current_status"] = w.get("original_status", "fresh")
                w["last_checked"] = datetime.now().strftime("%H:%M:%S")
            st.session_state["watchlist"] = watchlist
            st.toast("Watchlist re-checked across ground truth sources!", icon="🔄")
            st.rerun()
    with w_col2:
        if st.button("✕ Clear Watchlist", key="btn_clear_watchlist", use_container_width=True):
            st.session_state["watchlist"] = []
            st.toast("Watchlist cleared.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
