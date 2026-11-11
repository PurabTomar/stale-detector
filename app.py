"""
app.py - Modern dark-themed Streamlit UI for the Stale-Answer Detector.

Uses a two-column workspace (input on the left, live verification results on
the right), a pure HTML/CSS circular freshness score ring, 4 KPI metric cards,
side-by-side Claimed vs Actually (today) cards with clickable domain chips,
live st.status progress steps, and a sidebar with 4 demo examples, a live
Bifrost Gateway panel (health, outage toggle, latency/tokens/fallback telemetry,
dashboard link), and session Tavily credit tracking.
"""

from datetime import datetime
import html
import os
import re
from urllib.parse import urlparse

import streamlit as st

from extractor import extract_claims, get_last_gateway_metrics as get_extract_metrics
from llm import (
    _get_config_value,
    ask_question,
    check_gateway_health,
    get_gateway_dashboard_url,
    get_llm_config,
    is_use_bifrost,
    set_simulate_outage,
)
from reporting import (
    build_change_log,
    build_report_payload,
    build_side_by_side_word_diff,
    format_report_json,
    format_report_markdown,
)
from verifier import (
    clear_tavily_cache,
    generate_corrected_answer,
    get_last_gateway_metrics as get_rewrite_metrics,
    verify_claims,
)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stale-Answer Detector",
    page_icon=":material/fact_check:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar demo examples (iPhone, GPT version, GST, company CEO claim)
# ---------------------------------------------------------------------------
DEMO_EXAMPLES = [
    {
        "label": "iPhone release & pricing",
        "icon": ":material/smartphone:",
        "text": (
            "Apple's latest flagship smartphone is the iPhone 15 Pro Max, "
            "starting at $1,199 in the United States. It is powered by the A17 "
            "Pro chip and ships with iOS 17 out of the box."
        ),
    },
    {
        "label": "GPT version & API pricing",
        "icon": ":material/smart_toy:",
        "text": (
            "OpenAI's flagship API model is GPT-4 Turbo, which offers a 128K "
            "context window and is priced at $10 per million input tokens and "
            "$30 per million output tokens."
        ),
    },
    {
        "label": "India GST slabs & threshold",
        "icon": ":material/receipt_long:",
        "text": (
            "Under India's Goods and Services Tax (GST) regime, goods and "
            "services fall under four main tax slabs—5%, 12%, 18%, and 28%—and "
            "businesses with annual turnover above ₹20 lakh must register for GST."
        ),
    },
    {
        "label": "Company CEO & valuation",
        "icon": ":material/corporate_fare:",
        "text": (
            "Linda Yaccarino is the CEO of X (formerly Twitter), while Sam "
            "Altman serves as CEO of OpenAI, which reached a valuation of "
            "$86 billion in early 2024."
        ),
    },
]

# Status styling tokens used for badges, score bars, and ring colors.
STATUS_META = {
    "fresh": {
        "label": "Fresh",
        "pill_class": "pill-fresh",
        "color": "#22C55E",
        "badge_color": "green",
    },
    "uncertain": {
        "label": "Uncertain",
        "pill_class": "pill-uncertain",
        "color": "#F59E0B",
        "badge_color": "yellow",
    },
    "outdated": {
        "label": "Outdated",
        "pill_class": "pill-outdated",
        "color": "#EF4444",
        "badge_color": "red",
    },
}

# Clean inline SVG icons for claim categories so cards look sharp on any OS.
CATEGORY_ICONS = {
    "price": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/>'
        '<line x1="7" y1="7" x2="7.01" y2="7"/></svg>'
    ),
    "version": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>'
    ),
    "regulation": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 3v18"/><path d="M5 7h14"/><path d="M3 14l2-7 2 7a2 2 0 0 1-4 0z"/>'
        '<path d="M17 14l2-7 2 7a2 2 0 0 1-4 0z"/></svg>'
    ),
    "company": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/>'
        '<path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M8 10h.01"/><path d="M16 10h.01"/></svg>'
    ),
    "statistic": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/></svg>'
    ),
    "other": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
    ),
}

# ---------------------------------------------------------------------------
# Custom CSS (responsive dark UI with a single indigo accent: #6366F1)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.sad-hero {
    background: linear-gradient(135deg, #131A2A 0%, #171E35 60%, #1C2042 100%);
    border: 1px solid #243047;
    border-radius: 14px;
    padding: 1.35rem 1.6rem;
    margin-bottom: 1.25rem;
}
.sad-hero-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
}
.sad-hero-title {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    line-height: 1.2;
}
.sad-hero-tagline {
    margin: 0.45rem 0 0 0;
    font-size: 0.96rem;
    color: #94A3B8;
}
.sad-hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.sad-tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.76rem;
    font-weight: 600;
    color: #C7D2FE;
    background: rgba(99, 102, 241, 0.14);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    white-space: nowrap;
}
.sad-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6366F1;
    display: inline-block;
}

/* Overall score ring card */
.sad-score-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.sad-ring-wrap {
    flex-shrink: 0;
    width: 136px;
    height: 136px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 24px rgba(0, 0, 0, 0.35);
}
.sad-ring-inner {
    width: 106px;
    height: 106px;
    border-radius: 50%;
    background: #0F1524;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.sad-ring-num {
    font-size: 2rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1;
}
.sad-ring-denom {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 2px;
}
.sad-ring-tier {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}
.sad-score-meta h3 {
    margin: 0 0 0.3rem 0;
    font-size: 1.15rem;
    font-weight: 600;
    color: #F8FAFC;
}
.sad-score-meta p {
    margin: 0 0 0.5rem 0;
    font-size: 0.88rem;
    color: #94A3B8;
    line-height: 1.45;
}
.sad-model-tag {
    display: inline-flex;
    align-items: center;
    font-size: 0.78rem;
    color: #CBD5E1;
    background: #1A2236;
    border: 1px solid #243047;
    border-radius: 6px;
    padding: 0.22rem 0.55rem;
}

/* Claim cards */
.sad-claim-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 12px;
    padding: 1.05rem 1.15rem;
    margin-bottom: 0.85rem;
}
.sad-claim-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.85rem;
}
.sad-claim-badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
}
.sad-pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
}
.pill-fresh {
    color: #4ADE80;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.pill-uncertain {
    color: #FBBF24;
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.pill-outdated {
    color: #F87171;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.sad-cat-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.76rem;
    font-weight: 500;
    color: #CBD5E1;
    background: #1A2236;
    border: 1px solid #2A3752;
    border-radius: 6px;
    padding: 0.18rem 0.55rem;
    text-transform: capitalize;
}
.sad-meta-pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.72rem;
    font-weight: 600;
    border-radius: 6px;
    padding: 0.16rem 0.5rem;
    background: #1A2236;
    border: 1px solid #2A3752;
    color: #94A3B8;
}
.risk-critical {
    color: #FCA5A5;
    background: rgba(239, 68, 68, 0.12);
    border-color: rgba(239, 68, 68, 0.32);
}
.risk-moderate {
    color: #FCD34D;
    background: rgba(245, 158, 11, 0.12);
    border-color: rgba(245, 158, 11, 0.32);
}
.risk-low {
    color: #94A3B8;
    background: rgba(148, 163, 184, 0.1);
    border-color: rgba(148, 163, 184, 0.25);
}
.sad-score-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    min-width: 140px;
}
.sad-score-track {
    width: 84px;
    height: 7px;
    border-radius: 999px;
    background: #243047;
    overflow: hidden;
}
.sad-score-fill {
    height: 100%;
    border-radius: 999px;
}
.sad-score-text {
    font-size: 0.8rem;
    font-weight: 600;
    color: #E2E8F0;
    font-variant-numeric: tabular-nums;
}

/* Claimed vs Actually (today) side-by-side grid */
.sad-compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
}
.sad-compare-box {
    background: #0E1422;
    border: 1px solid #202B40;
    border-radius: 9px;
    padding: 0.7rem 0.85rem;
}
.sad-compare-actual {
    border-color: rgba(99, 102, 241, 0.38);
    background: rgba(99, 102, 241, 0.06);
}
.sad-compare-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.3rem;
}
.sad-compare-actual .sad-compare-label {
    color: #A5B4FC;
}
.sad-compare-body {
    font-size: 0.88rem;
    color: #E2E8F0;
    line-height: 1.45;
}
.sad-reason {
    font-size: 0.86rem;
    color: #CBD5E1;
    line-height: 1.45;
    margin-bottom: 0.65rem;
}
.sad-reason strong {
    color: #F1F5F9;
}
.sad-error-note {
    font-size: 0.82rem;
    color: #FCD34D;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 0.45rem 0.7rem;
    margin-bottom: 0.65rem;
}
.sad-chips-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.45rem;
}
.sad-source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.76rem;
    font-weight: 500;
    color: #A5B4FC !important;
    text-decoration: none !important;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 999px;
    padding: 0.2rem 0.62rem;
    transition: background 0.15s ease, border-color 0.15s ease;
}
.sad-source-chip:hover {
    background: rgba(99, 102, 241, 0.24);
    border-color: #6366F1;
}
.sad-card-footer {
    margin-top: 0.55rem;
    font-size: 0.75rem;
    color: #64748B;
}

/* Empty state card on the right column */
.sad-empty-state {
    background: #131A2A;
    border: 1px dashed #2A3752;
    border-radius: 14px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: #94A3B8;
}
.sad-empty-state h4 {
    margin: 0 0 0.4rem 0;
    color: #E2E8F0;
    font-weight: 600;
}
.sad-empty-state p {
    margin: 0;
    font-size: 0.9rem;
}

/* Sidebar Tavily session credits card */
.sad-credits-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 10px;
    padding: 0.7rem 0.85rem;
    margin-top: 0.35rem;
}
.sad-credits-label {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}
.sad-credits-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
}
.sad-credits-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F8FAFC;
    font-variant-numeric: tabular-nums;
}
.sad-credits-pill {
    font-size: 0.7rem;
    font-weight: 600;
    color: #C7D2FE;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 999px;
    padding: 0.12rem 0.5rem;
}

/* Sidebar Bifrost Gateway panel */
.sad-gw-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 10px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.55rem;
}
.sad-gw-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.sad-health-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.72rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 0.16rem 0.55rem;
}
.sad-health-green {
    color: #4ADE80;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.sad-health-red {
    color: #F87171;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.sad-health-blue {
    color: #A5B4FC;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.4);
}
.sad-health-dot-green {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22C55E;
    display: inline-block;
}
.sad-health-dot-red {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #EF4444;
    display: inline-block;
}
.sad-health-dot-blue {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #6366F1;
    display: inline-block;
}
.sad-gw-link {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #A5B4FC !important;
    text-decoration: none !important;
    margin-top: 0.35rem;
}
.sad-gw-link:hover {
    text-decoration: underline !important;
    color: #C7D2FE !important;
}
.sad-gw-sub {
    font-size: 0.72rem;
    color: #94A3B8;
    margin-top: 0.22rem;
    word-break: break-word;
}
.sad-gw-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
    margin-top: 0.5rem;
}
.sad-gw-stat {
    background: #0E1422;
    border: 1px solid #202B40;
    border-radius: 8px;
    padding: 0.42rem 0.52rem;
}
.sad-gw-stat-full {
    grid-column: 1 / -1;
}
.sad-gw-stat-label {
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94A3B8;
}
.sad-gw-stat-value {
    font-size: 0.84rem;
    font-weight: 600;
    color: #F8FAFC;
    margin-top: 0.14rem;
    word-break: break-word;
    font-variant-numeric: tabular-nums;
}
.sad-fallback-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.73rem;
    font-weight: 700;
    color: #FBBF24;
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 999px;
    padding: 0.18rem 0.58rem;
    margin-top: 0.4rem;
    word-break: break-word;
}
.sad-primary-pill {
    display: inline-flex;
    align-items: center;
    font-size: 0.71rem;
    font-weight: 600;
    color: #4ADE80;
    background: rgba(34, 197, 94, 0.12);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 999px;
    padding: 0.15rem 0.52rem;
    margin-top: 0.4rem;
}
.sad-gw-log {
    margin-top: 0.55rem;
    max-height: 185px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}
.sad-gw-log-row {
    background: #0E1422;
    border: 1px solid #202B40;
    border-radius: 7px;
    padding: 0.38rem 0.52rem;
    font-size: 0.72rem;
    color: #CBD5E1;
    line-height: 1.38;
}
.sad-gw-log-head {
    display: flex;
    justify-content: space-between;
    gap: 0.4rem;
    font-weight: 600;
    color: #E2E8F0;
}

/* Ask & Audit inline claim highlights */
.sad-ask-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
}
.sad-ask-question {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 0.6rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #202B40;
}
.sad-ask-body {
    font-size: 0.95rem;
    line-height: 1.72;
    color: #F1F5F9;
    word-break: break-word;
}
.sad-hl-outdated {
    background: rgba(239, 68, 68, 0.16);
    text-decoration: underline;
    text-decoration-color: #EF4444;
    text-decoration-thickness: 2.5px;
    text-underline-offset: 4px;
    border-radius: 4px;
    padding: 0.05rem 0.22rem;
    color: #FECACA;
}
.sad-hl-fresh {
    background: rgba(34, 197, 94, 0.14);
    text-decoration: underline;
    text-decoration-color: #22C55E;
    text-decoration-thickness: 2px;
    text-underline-offset: 4px;
    border-radius: 4px;
    padding: 0.05rem 0.22rem;
    color: #DCFCE7;
}
.sad-hl-uncertain {
    background: rgba(245, 158, 11, 0.12);
    text-decoration: underline dashed;
    text-decoration-color: #F59E0B;
    text-decoration-thickness: 2px;
    text-underline-offset: 4px;
    border-radius: 4px;
    padding: 0.05rem 0.22rem;
    color: #FEF3C7;
}
.sad-hl-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.85rem;
    margin-top: 0.85rem;
    padding-top: 0.65rem;
    border-top: 1px solid #202B40;
    font-size: 0.76rem;
    color: #94A3B8;
}

/* Corrected answer side-by-side diff & changelog */
.sad-diff-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    height: 100%;
}
.sad-diff-card-orig {
    border-top: 3px solid rgba(239, 68, 68, 0.7);
}
.sad-diff-card-new {
    border-top: 3px solid rgba(34, 197, 94, 0.75);
}
.sad-diff-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.65rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #202B40;
}
.sad-diff-title {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #CBD5E1;
}
.sad-diff-tag {
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 0.14rem 0.55rem;
}
.sad-diff-tag-del {
    color: #FCA5A5;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.sad-diff-tag-ins {
    color: #86EFAC;
    background: rgba(34, 197, 94, 0.14);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.sad-diff-body {
    font-size: 0.93rem;
    line-height: 1.72;
    color: #E2E8F0;
    word-break: break-word;
}
.sad-diff-del {
    background: rgba(239, 68, 68, 0.22);
    color: #FCA5A5;
    text-decoration: line-through;
    text-decoration-color: #EF4444;
    text-decoration-thickness: 2px;
    border-radius: 4px;
    padding: 0.03rem 0.24rem;
}
.sad-diff-ins {
    background: rgba(34, 197, 94, 0.18);
    color: #86EFAC;
    text-decoration: none;
    border-bottom: 2px solid rgba(34, 197, 94, 0.75);
    border-radius: 4px;
    padding: 0.03rem 0.24rem;
    font-weight: 500;
}
.sad-change-card {
    background: #131A2A;
    border: 1px solid #243047;
    border-radius: 11px;
    padding: 0.9rem 1.05rem;
    margin-bottom: 0.7rem;
}
.sad-change-top {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.6rem;
}
.sad-change-pill-replaced {
    color: #F87171;
    background: rgba(239, 68, 68, 0.14);
    border: 1px solid rgba(239, 68, 68, 0.35);
}
.sad-change-pill-qualified {
    color: #FBBF24;
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.sad-change-diff-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
    margin-bottom: 0.55rem;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .sad-score-card {
        flex-direction: column;
        text-align: center;
    }
    .sad-compare-grid,
    .sad-change-diff-row {
        grid-template-columns: 1fr;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def escape_markdown(text: str) -> str:
    """Escape unescaped '$' characters so Streamlit does not trigger KaTeX math."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


def escape_html(text: str) -> str:
    """Escape HTML special characters and '$' for safe HTML rendering."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def extract_domain(url: str) -> str:
    """Return a clean domain name (without 'www.') from a URL."""
    parsed = urlparse(url.strip())
    domain = parsed.netloc or url
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def get_overall_tier(score: int) -> tuple[str, str, str]:
    """
    Map an overall freshness score (0-100) to (label, ring_color, description).

    Thresholds:
    - Fresh: 80-100
    - Mixed: 40-79
    - Stale: 0-39
    """
    if score >= 80:
        return "Fresh", "#22C55E", "Claims align closely with current live web sources."
    if score >= 40:
        return "Mixed", "#F59E0B", "Some details are outdated or need verification."
    return "Stale", "#EF4444", "Key facts are outdated compared to today's sources."


def _sanitize_error_snippet(raw_text: str, max_len: int = 160) -> str:
    """Remove Python traceback frames/paths and keep at most one concise line."""
    clean_lines: list[str] = []
    for line in str(raw_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            stripped.startswith("Traceback ")
            or stripped.startswith('File "')
            or "Traceback (most recent call last)" in stripped
        ):
            continue
        clean_lines.append(stripped)
    snippet = clean_lines[0] if clean_lines else ""
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 3] + "..."
    return snippet


def format_friendly_ui_error(exc: Exception | str, *, action: str = "checking claims") -> str:
    """
    Translate raw exceptions into calm, actionable messages for Bifrost down,
    invalid API keys, Tavily rate/credit limits, and invalid JSON from the LLM,
    never leaking raw Python stack traces to the Streamlit UI.
    """
    raw = str(exc or "").strip()
    lower = raw.lower()
    use_bifrost = is_use_bifrost()

    # 1. Invalid JSON from the LLM (extraction or verification schema parsing)
    if any(
        tok in lower
        for tok in (
            "valid json",
            "json list",
            "json object",
            "parse json",
            "jsondecodeerror",
            "non-json formatting",
            "invalid json",
        )
    ):
        return (
            "The LLM returned invalid JSON instead of the expected claim structure. "
            "Please try running the check again or switch models in LLM_MODEL."
        )

    # 2. Tavily rate / credit / quota limit
    if any(
        tok in lower
        for tok in (
            "tavily rate or credit limit",
            "tavily credit or plan limit",
            "tavily rate limit",
            "usage limit",
            "usagelimitexceedederror",
            "account/billing limit",
            "billing limit exceeded",
            "432",
            "433",
            "434",
        )
    ) or ("tavily" in lower and ("429" in lower or "rate limit" in lower or "quota" in lower or "credit" in lower)):
        return (
            "Tavily search rate or credit limit reached. Wait a short moment or check "
            "your remaining Tavily quota in your Tavily dashboard."
        )

    # 3. Invalid or missing API keys (Tavily, Bifrost virtual key, Gemini, or LLM)
    if "tavily_api_key" in lower or (
        "tavily" in lower
        and any(tok in lower for tok in ("401", "403", "invalid", "unauthorized", "not set", "missing", "forbidden"))
    ):
        return (
            "Invalid or missing Tavily API key. Please check TAVILY_API_KEY in your "
            ".env file or Streamlit Secrets."
        )

    if "llm_virtual_key" in lower or "x-bf-vk" in lower or "virtual key" in lower:
        return (
            "Invalid Bifrost virtual key (LLM_VIRTUAL_KEY / x-bf-vk). Please verify "
            "your virtual key in .env and try again."
        )

    is_auth_failure = (
        "llm_api_key" in lower
        or "gemini_api_key" in lower
        or any(
            tok in lower
            for tok in (
                "invalid api key",
                "invalid gemini api key",
                "invalid llm api key",
                "incorrect api key",
                "api_key_invalid",
                "api key not valid",
                "authentication",
                "unauthorized",
                "401",
                "403",
            )
        )
    )
    if is_auth_failure:
        if not use_bifrost or "gemini" in lower:
            return (
                "Invalid or missing Gemini API key. Please check LLM_API_KEY "
                "(or GEMINI_API_KEY) in your .env file or Streamlit Secrets."
            )
        return (
            "Invalid or missing LLM API key. Please check LLM_API_KEY and your "
            "provider keys in Bifrost before retrying."
        )

    # 4. Bifrost gateway down / unreachable
    if ("bifrost" in lower and any(tok in lower for tok in ("down", "unreachable", "offline", "refused", "timeout", "timed out"))) or (
        use_bifrost
        and any(
            tok in lower
            for tok in (
                "connection refused",
                "connecterror",
                "unreachable",
                "winerror 10061",
                "apiconnectionerror",
                "failed to establish a new connection",
                "502",
                "503",
                "504",
            )
        )
    ):
        return (
            "Bifrost gateway is down or unreachable. Start Bifrost locally on port 8080, "
            "or set USE_BIFROST=false in .env to call Gemini directly."
        )

    # 5. Direct connection / timeout errors
    if any(tok in lower for tok in ("apiconnectionerror", "connecterror", "timed out", "timeout", "connection error", "connection refused")):
        return (
            f"Could not reach the API service while {action}. Check your internet connection "
            "and endpoint settings, then try again."
        )

    # 6. LLM rate limits / missing model
    if "429" in lower or "rate limit" in lower or "quota" in lower or "resource_exhausted" in lower:
        return (
            "The LLM provider rate or quota limit was reached (429). Wait a moment, "
            "or add fallback models in LLM_MODEL_FALLBACKS."
        )

    if "404" in lower or "not found" in lower:
        return (
            "The configured LLM model was not found (404). Please verify LLM_MODEL in your "
            ".env file or Streamlit Secrets (e.g. 'gemini-2.5-flash' for Direct Gemini)."
        )

    # 7. Safe catch-all: never display raw stack traces in UI
    if "traceback" in lower or 'file "' in lower:
        return f"Something went wrong while {action}. Check your API configuration and try again."

    snippet = _sanitize_error_snippet(raw)
    if snippet:
        return f"Could not complete {action}: {snippet}"
    return f"Something went wrong while {action}. Please check your settings and try again."


def load_demo_example(example_text: str) -> None:
    """Sidebar button callback: fills the text area and clears prior results."""
    st.session_state["answer_input"] = example_text
    st.session_state["results"] = None
    st.session_state["checked_answer"] = ""
    st.session_state["extract_model"] = None
    st.session_state["extract_meta"] = None
    st.session_state["corrected_answer"] = None
    st.session_state["corrected_model"] = None
    st.session_state["corrected_meta"] = None
    st.session_state["pasted_audit_at"] = None
    st.session_state["status_steps"] = []


def clear_workspace() -> None:
    """Clear the text area and previous verification results in Tab 1."""
    load_demo_example("")


def clear_ask_workspace() -> None:
    """Clear the question, answer, and audit results in Tab 2 ('Ask & Audit')."""
    st.session_state["ask_question_input"] = ""
    st.session_state["ask_question_asked"] = ""
    st.session_state["ask_answer"] = ""
    st.session_state["ask_model"] = None
    st.session_state["ask_meta"] = None
    st.session_state["ask_results"] = None
    st.session_state["ask_extract_model"] = None
    st.session_state["ask_extract_meta"] = None
    st.session_state["ask_corrected_answer"] = None
    st.session_state["ask_corrected_model"] = None
    st.session_state["ask_corrected_meta"] = None
    st.session_state["ask_audit_at"] = None
    st.session_state["ask_status_steps"] = []


def init_session_state() -> None:
    """Initialize required keys in Streamlit session_state."""
    defaults = {
        "answer_input": "",
        "results": None,
        "checked_answer": "",
        "extract_model": None,
        "extract_meta": None,
        "corrected_answer": None,
        "corrected_model": None,
        "corrected_meta": None,
        "pasted_audit_at": None,
        "status_steps": [],
        "ask_question_input": "",
        "ask_question_asked": "",
        "ask_answer": "",
        "ask_model": None,
        "ask_meta": None,
        "ask_results": None,
        "ask_extract_model": None,
        "ask_extract_meta": None,
        "ask_corrected_answer": None,
        "ask_corrected_model": None,
        "ask_corrected_meta": None,
        "ask_audit_at": None,
        "ask_status_steps": [],
        "show_search_queries": False,
        "keep_status_expanded": False,
        "simulate_outage": False,
        "tavily_credits_used": 0.0,
        "gateway_requests": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_MATCH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have", "has", "had",
    "was", "were", "are", "been", "being", "its", "their", "into", "about",
    "which", "who", "whom", "what", "when", "where", "under", "over", "per",
    "out", "box", "also", "very", "such", "more", "most", "can", "will", "it",
}


def _extract_match_tokens(text: str) -> dict[str, float]:
    """Extract weighted keywords and normalized numbers/versions for claim-span matching."""
    tokens: dict[str, float] = {}
    for raw_num in re.findall(r"[$€£₹]?\d[\d,]*(?:\.\d+)*%?", text):
        normalized = raw_num.lstrip("$€£₹").replace(",", "").strip().lower()
        if normalized:
            tokens[normalized] = max(tokens.get(normalized, 0.0), 2.5)

    no_comma_nums = re.sub(r"(\d),(\d)", r"\1\2", text)
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", no_comma_nums):
        low = word.lower().strip(".")
        if len(low) < 2 and not low.isdigit():
            continue
        if low in _MATCH_STOPWORDS:
            continue
        weight = 1.6 if any(ch.isdigit() for ch in low) else 1.0
        tokens[low] = max(tokens.get(low, 0.0), weight)
    return tokens


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """Trim leading/trailing whitespace and sentence-ending punctuation from a span."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in ".!?;,"):
        end -= 1
    return (start, end) if end > start else None


def _candidate_answer_spans(answer_text: str) -> list[tuple[int, int]]:
    """Build sentence, sub-clause, and adjacent-clause candidate spans with exact offsets."""
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_span(s: int, e: int) -> None:
        trimmed = _trim_span(answer_text, s, e)
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            spans.append(trimmed)

    # Split into sentence-like segments without breaking decimal numbers or "U.S." initials.
    sent_bounds = [0]
    for m in re.finditer(r"(?:(?<!\b[A-Z])(?<=[.!?])\s+|\n+)", answer_text):
        sent_bounds.append(m.start())
        sent_bounds.append(m.end())
    sent_bounds.append(len(answer_text))

    sentence_ranges: list[tuple[int, int]] = []
    for i in range(0, len(sent_bounds) - 1, 2):
        trimmed_sent = _trim_span(answer_text, sent_bounds[i], sent_bounds[i + 1])
        if trimmed_sent:
            sentence_ranges.append(trimmed_sent)
            add_span(*trimmed_sent)

    # Split each sentence into tighter sub-clauses (preserving commas inside numbers like $1,199).
    clause_sep = re.compile(
        r"(?:\s*;\s*|,\s+(?!\d)|\s+[—–-]\s+|\s+(?:and|while|whereas)\s+)",
        re.IGNORECASE,
    )
    for sent_s, sent_e in sentence_ranges:
        sent_slice = answer_text[sent_s:sent_e]
        parts: list[tuple[int, int]] = []
        cursor = 0
        for sep in clause_sep.finditer(sent_slice):
            c_trimmed = _trim_span(answer_text, sent_s + cursor, sent_s + sep.start())
            if c_trimmed:
                parts.append(c_trimmed)
                add_span(*c_trimmed)
            cursor = sep.end()
        tail_trimmed = _trim_span(answer_text, sent_s + cursor, sent_e)
        if tail_trimmed:
            parts.append(tail_trimmed)
            add_span(*tail_trimmed)

        # Include adjacent 2-clause pairs in case one claim bridges two clauses.
        for j in range(len(parts) - 1):
            add_span(parts[j][0], parts[j + 1][1])

    return spans


def find_claim_span(answer_text: str, claim_text: str) -> tuple[int, int, float] | None:
    """
    Locate the character range `(start, end, confidence)` in `answer_text` that
    corresponds to `claim_text`. Tries exact substring matching first, then
    falls back to weighted token overlap across clauses and sentences.
    """
    if not answer_text or not claim_text:
        return None

    clean_claim = claim_text.strip().rstrip(".!?")
    if len(clean_claim) >= 4:
        words = clean_claim.split()
        exact_pattern = r"\s+".join(re.escape(w) for w in words)
        exact_match = re.search(exact_pattern, answer_text, re.IGNORECASE)
        if exact_match:
            return exact_match.start(), exact_match.end(), 1.0

    claim_tokens = _extract_match_tokens(claim_text)
    if not claim_tokens:
        return None
    total_claim_weight = sum(claim_tokens.values())

    best_span: tuple[int, int, float] | None = None
    best_len = 10**9

    for s, e in _candidate_answer_spans(answer_text):
        span_tokens = _extract_match_tokens(answer_text[s:e])
        if not span_tokens:
            continue
        matched_keys = [k for k in claim_tokens if k in span_tokens]
        if not matched_keys:
            continue

        has_numeric = any(k[0].isdigit() for k in matched_keys)
        if len(matched_keys) < 2 and not has_numeric:
            continue

        matched_weight = sum(claim_tokens[k] for k in matched_keys)
        recall = matched_weight / total_claim_weight
        if recall < 0.42:
            continue

        span_total_weight = sum(span_tokens.values()) or 1.0
        span_matched_weight = sum(span_tokens[k] for k in matched_keys)
        precision = span_matched_weight / span_total_weight
        combined = round(0.72 * recall + 0.28 * precision, 4)

        span_len = e - s
        if (
            best_span is None
            or combined > best_span[2]
            or (combined == best_span[2] and span_len < best_len)
        ):
            best_span = (s, e, combined)
            best_len = span_len

    return best_span


def match_claims_to_answer_spans(answer_text: str, verification_results: list[dict]) -> list[dict]:
    """
    Match verified claim results to non-overlapping character spans in `answer_text`.
    When a compound sentence matches a wider span while another claim cleanly matches
    one sub-clause, tighten the compound match to its remaining sub-clause. Otherwise,
    when two spans genuinely overlap, outdated claims take precedence.
    """
    raw_matches: list[dict] = []
    status_priority = {"outdated": 0, "fresh": 1, "uncertain": 2}
    all_candidate_spans = _candidate_answer_spans(answer_text)

    for item in verification_results or []:
        claim_text = str(item.get("claim", ""))
        span = find_claim_span(answer_text, claim_text)
        if span is None:
            continue
        s, e, match_score = span
        status = str(item.get("status", "uncertain")).lower()
        if status not in status_priority:
            status = "uncertain"
        raw_matches.append(
            {
                "claim": claim_text,
                "start": s,
                "end": e,
                "status": status,
                "score": int(item.get("score", 50)),
                "current_info": str(item.get("current_info", "")).strip(),
                "match_score": match_score,
                "priority": status_priority[status],
            }
        )

    # Tighten wide compound-sentence matches when another narrower claim already claims
    # one of the sub-clauses inside that same sentence (e.g., green clause + red clause).
    for idx, cand in enumerate(raw_matches):
        enclosing_others = [
            other
            for j, other in enumerate(raw_matches)
            if j != idx
            and other["start"] >= cand["start"]
            and other["end"] <= cand["end"]
            and (other["end"] - other["start"]) < (cand["end"] - cand["start"])
        ]
        if not enclosing_others:
            continue

        claim_tokens = _extract_match_tokens(cand["claim"])
        total_weight = sum(claim_tokens.values()) or 1.0
        best_tight: tuple[int, int, float] | None = None
        for sub_s, sub_e in all_candidate_spans:
            if sub_s < cand["start"] or sub_e > cand["end"]:
                continue
            if (sub_e - sub_s) >= (cand["end"] - cand["start"]):
                continue
            if any(sub_s < o["end"] and sub_e > o["start"] for o in enclosing_others):
                continue
            sub_tokens = _extract_match_tokens(answer_text[sub_s:sub_e])
            matched = [k for k in claim_tokens if k in sub_tokens]
            if not matched:
                continue
            if len(matched) < 2 and not any(k[0].isdigit() for k in matched):
                continue
            recall = sum(claim_tokens[k] for k in matched) / total_weight
            if recall < 0.35:
                continue
            if best_tight is None or recall > best_tight[2]:
                best_tight = (sub_s, sub_e, round(recall, 4))

        if best_tight is not None:
            cand["start"], cand["end"], cand["match_score"] = best_tight

    raw_matches.sort(key=lambda c: (c["priority"], -c["match_score"], c["end"] - c["start"]))

    accepted: list[dict] = []
    for cand in raw_matches:
        overlaps = any(
            cand["start"] < prev["end"] and cand["end"] > prev["start"]
            for prev in accepted
        )
        if not overlaps:
            accepted.append(cand)

    accepted.sort(key=lambda c: c["start"])
    return accepted


def render_highlighted_answer_html(
    answer_text: str,
    verification_results: list[dict] | None,
) -> tuple[str, int]:
    """
    Render `answer_text` as safe HTML with inline underlines highlighting verified claims:
    red underline for outdated, green underline for fresh, dashed amber for uncertain.
    """
    if not answer_text:
        return "", 0

    spans = match_claims_to_answer_spans(answer_text, verification_results or [])
    if not spans:
        return escape_html(answer_text).replace("\n", "<br>"), 0

    cls_map = {
        "outdated": ("sad-hl-outdated", "Outdated"),
        "fresh": ("sad-hl-fresh", "Fresh"),
        "uncertain": ("sad-hl-uncertain", "Uncertain"),
    }

    parts: list[str] = []
    cursor = 0
    for span in spans:
        if span["start"] > cursor:
            parts.append(escape_html(answer_text[cursor : span["start"]]).replace("\n", "<br>"))
        hl_cls, status_label = cls_map.get(span["status"], ("sad-hl-uncertain", "Uncertain"))
        tip_text = f"{status_label} ({span['score']}/100)"
        if span["current_info"]:
            tip_text += f" — {span['current_info']}"
        segment = escape_html(answer_text[span["start"] : span["end"]]).replace("\n", "<br>")
        parts.append(f'<span class="{hl_cls}" title="{escape_html(tip_text)}">{segment}</span>')
        cursor = span["end"]

    if cursor < len(answer_text):
        parts.append(escape_html(answer_text[cursor:]).replace("\n", "<br>"))

    return "".join(parts), len(spans)


def render_ask_answer_card(
    placeholder,
    answer_text: str,
    verification_results: list[dict] | None = None,
    question_text: str | None = None,
    answer_model: str | None = None,
    answer_meta: dict | None = None,
    audit_error: bool = False,
) -> None:
    """Render the LLM answer card (upgrading with red/green underlines once audited)."""
    body_html, hl_count = render_highlighted_answer_html(answer_text, verification_results)

    question_block = (
        f'<div class="sad-ask-question"><strong>Question:</strong> {escape_html(question_text)}</div>'
        if question_text
        else ""
    )

    if verification_results is not None:
        if hl_count > 0:
            legend_html = (
                '<div class="sad-hl-legend">'
                '<span><span class="sad-hl-outdated">Red underline</span> = Outdated</span>'
                '<span><span class="sad-hl-fresh">Green underline</span> = Fresh</span>'
                '<span><span class="sad-hl-uncertain">Dashed amber</span> = Uncertain</span>'
                '</div>'
            )
        elif len(verification_results) == 0:
            legend_html = (
                '<div class="sad-hl-legend">'
                '<span>Audit complete &bull; No time-sensitive factual claims were flagged.</span>'
                '</div>'
            )
        else:
            legend_html = (
                '<div class="sad-hl-legend">'
                '<span>Audit complete &bull; See claim cards on the right for full claim-by-claim details.</span>'
                '</div>'
            )
    elif audit_error:
        legend_html = (
            '<div class="sad-hl-legend">'
            '<span>Audit stopped due to an error &bull; See details on the right.</span>'
            '</div>'
        )
    else:
        legend_html = (
            '<div class="sad-hl-legend">'
            '<span>Auditing answer against live web sources...</span>'
            '</div>'
        )

    meta_parts: list[str] = []
    if answer_model:
        meta_parts.append(f"Answered by: {escape_html(answer_model)}")
    if isinstance(answer_meta, dict) and answer_meta.get("fallback_used"):
        fb_route = format_fallback_route(
            answer_meta.get("primary_model"),
            answer_meta.get("model_used") or answer_model,
        )
        meta_parts.append(f'<span style="color:#FBBF24;">Fallback used: {escape_html(fb_route)}</span>')
    if isinstance(answer_meta, dict) and answer_meta.get("latency_ms"):
        meta_parts.append(f"{float(answer_meta['latency_ms']):.0f} ms")
    footer_html = (
        f'<div class="sad-card-footer">{" &bull; ".join(meta_parts)}</div>'
        if meta_parts
        else ""
    )

    placeholder.markdown(
        f'<div class="sad-ask-card">'
        f'{question_block}'
        f'<div class="sad-ask-body">{body_html}</div>'
        f'{legend_html}'
        f'{footer_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def format_fallback_route(primary_model: str | None, answered_model: str | None) -> str:
    """Format a fallback route string as 'A -> B' for 'Fallback used: A -> B' badges."""
    source = (primary_model or "").strip() or (os.getenv("LLM_MODEL", "").strip() or "primary")
    target = (answered_model or "").strip() or "fallback"
    return f"{source} -> {target}"


def build_request_telemetry(
    operation: str,
    metrics: dict | None,
    model_used: str | None = None,
    fallback_used: bool = False,
    primary_model: str | None = None,
) -> dict | None:
    """Normalize per-request Bifrost telemetry from backend metrics or result dicts."""
    if isinstance(metrics, dict):
        record = dict(metrics)
        record["operation"] = operation
        record["model_used"] = str(record.get("model_used") or model_used or "Unknown")
        record["primary_model"] = str(
            record.get("primary_model")
            or primary_model
            or os.getenv("LLM_MODEL", "").strip()
            or record["model_used"]
        )
        record["latency_ms"] = round(float(record.get("latency_ms") or 0.0), 1)
        record["prompt_tokens"] = int(record.get("prompt_tokens") or 0)
        record["completion_tokens"] = int(record.get("completion_tokens") or 0)
        record["total_tokens"] = int(
            record.get("total_tokens")
            or (record["prompt_tokens"] + record["completion_tokens"])
        )
        record["fallback_used"] = bool(record.get("fallback_used") or fallback_used)
        return record

    if not model_used:
        return None

    resolved_primary = (primary_model or os.getenv("LLM_MODEL", "").strip() or model_used)
    return {
        "operation": operation,
        "primary_model": resolved_primary,
        "model_used": str(model_used),
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "fallback_used": bool(fallback_used),
        "simulated_outage": False,
        "timestamp": "",
    }


def append_gateway_request(record: dict | None) -> None:
    """Append a normalized request record to session telemetry history."""
    if not record:
        return
    history = list(st.session_state.get("gateway_requests") or [])
    history.append(record)
    st.session_state["gateway_requests"] = history


def collect_fallback_routes(
    extract_meta: dict | None,
    results: list[dict] | None,
) -> list[str]:
    """Return unique 'A -> B' strings for any request where Bifrost triggered fallback."""
    routes: list[str] = []
    if isinstance(extract_meta, dict) and extract_meta.get("fallback_used"):
        route = format_fallback_route(
            extract_meta.get("primary_model"),
            extract_meta.get("model_used"),
        )
        if route not in routes:
            routes.append(route)

    for item in results or []:
        gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else {}
        used_fallback = bool(item.get("fallback_used") or gw.get("fallback_used"))
        if used_fallback:
            route = format_fallback_route(
                item.get("primary_model") or gw.get("primary_model"),
                item.get("model") or gw.get("model_used"),
            )
            if route not in routes:
                routes.append(route)
    return routes


def render_gateway_health_card(health: dict, dashboard_url: str, use_bifrost: bool = True) -> None:
    """Render the green/red Bifrost health badge OR Direct Gemini cloud badge."""
    mode = str(health.get("mode") or ("bifrost" if use_bifrost else "direct_gemini"))
    if mode == "direct_gemini" or not use_bifrost:
        endpoint_host = extract_domain(str(health.get("url") or "https://generativelanguage.googleapis.com/v1beta/openai/"))
        card_html = (
            '<div class="sad-gw-card">'
            '<div class="sad-gw-top">'
            '<span class="sad-health-pill sad-health-blue">'
            '<span class="sad-health-dot-blue"></span>Direct Gemini mode'
            '</span>'
            '</div>'
            '<div class="sad-gw-sub">USE_BIFROST=false &bull; Cloud-ready direct endpoint</div>'
            f'<div class="sad-gw-sub">{escape_html(endpoint_host)}/v1beta/openai/</div>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
        return

    is_healthy = bool(health.get("healthy"))
    pill_cls = "sad-health-green" if is_healthy else "sad-health-red"
    dot_cls = "sad-health-dot-green" if is_healthy else "sad-health-dot-red"
    status_label = "Gateway healthy" if is_healthy else "Gateway offline"
    version_text = f"v{health['version']}" if health.get("version") else health.get("status", "")
    detail_line = (
        version_text
        if is_healthy
        else "Start Bifrost on port 8080 or set USE_BIFROST=false"
    )

    safe_url = html.escape(dashboard_url, quote=True)
    display_url = escape_html(dashboard_url)

    card_html = (
        '<div class="sad-gw-card">'
        '<div class="sad-gw-top">'
        f'<span class="sad-health-pill {pill_cls}">'
        f'<span class="{dot_cls}"></span>{escape_html(status_label)}'
        '</span>'
        '</div>'
        f'<div class="sad-gw-sub">{escape_html(detail_line)}</div>'
        f'<a class="sad-gw-link" href="{safe_url}" target="_blank" rel="noopener noreferrer">'
        f'Bifrost dashboard logs ({display_url}) &#8599;'
        '</a>'
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_gateway_panel(placeholder, requests_log: list[dict]) -> None:
    """Render live per-request metrics and session token totals in the sidebar Gateway panel."""
    total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in requests_log)
    total_completion = sum(int(r.get("completion_tokens") or 0) for r in requests_log)
    total_session_tokens = sum(int(r.get("total_tokens") or 0) for r in requests_log)
    req_count = len(requests_log)
    fallback_count = sum(1 for r in requests_log if r.get("fallback_used"))

    if requests_log:
        latest = requests_log[-1]
        lat_text = f"{float(latest.get('latency_ms') or 0.0):.0f} ms"
        latest_model = escape_html(latest.get("model_used") or "Unknown")
        latest_op = escape_html(latest.get("operation") or "Request")
        p_tok = int(latest.get("prompt_tokens") or 0)
        c_tok = int(latest.get("completion_tokens") or 0)

        if latest.get("fallback_used"):
            fb_label = format_fallback_route(latest.get("primary_model"), latest.get("model_used"))
            fb_badge = f'<span class="sad-fallback-pill">Fallback used: {escape_html(fb_label)}</span>'
        else:
            fb_badge = '<span class="sad-primary-pill">Fallback used: No (primary)</span>'

        latest_html = (
            '<div class="sad-gw-grid">'
            '<div class="sad-gw-stat sad-gw-stat-full">'
            f'<div class="sad-gw-stat-label">Last answered ({latest_op})</div>'
            f'<div class="sad-gw-stat-value">{latest_model}</div>'
            '</div>'
            '<div class="sad-gw-stat">'
            '<div class="sad-gw-stat-label">Latency</div>'
            f'<div class="sad-gw-stat-value">{escape_html(lat_text)}</div>'
            '</div>'
            '<div class="sad-gw-stat">'
            '<div class="sad-gw-stat-label">Prompt / Completion</div>'
            f'<div class="sad-gw-stat-value">{p_tok:,} / {c_tok:,}</div>'
            '</div>'
            '</div>'
            f'{fb_badge}'
        )

        # Compact per-request log (most recent first so the newest call stays visible).
        rows_html: list[str] = []
        for idx_rev, req in enumerate(reversed(requests_log), start=1):
            req_num = req_count - idx_rev + 1
            op = escape_html(req.get("operation") or f"Request #{req_num}")
            m_used = escape_html(req.get("model_used") or "Unknown")
            r_lat = f"{float(req.get('latency_ms') or 0.0):.0f}ms"
            rp = int(req.get("prompt_tokens") or 0)
            rc = int(req.get("completion_tokens") or 0)
            if req.get("fallback_used"):
                route_str = escape_html(
                    format_fallback_route(req.get("primary_model"), req.get("model_used"))
                )
                r_fb = f'<span style="color:#FBBF24;">Fallback used: {route_str}</span>'
            else:
                r_fb = '<span style="color:#4ADE80;">Primary</span>'
            rows_html.append(
                '<div class="sad-gw-log-row">'
                '<div class="sad-gw-log-head">'
                f'<span>#{req_num} {op}</span>'
                f'<span>{escape_html(r_lat)}</span>'
                '</div>'
                f'<div>{m_used} &bull; {rp:,} in / {rc:,} out &bull; {r_fb}</div>'
                '</div>'
            )
        log_section = '<div class="sad-gw-log">' + "".join(rows_html) + "</div>"
    else:
        empty_note = (
            "No LLM requests yet. Run Check Freshness to inspect live Bifrost telemetry."
            if is_use_bifrost()
            else "No LLM requests yet. Run Check Freshness to inspect Gemini token and latency telemetry."
        )
        latest_html = (
            f'<div class="sad-gw-sub" style="margin-top:0.5rem;">{escape_html(empty_note)}</div>'
        )
        log_section = ""

    panel_html = (
        '<div class="sad-gw-card">'
        '<div class="sad-gw-grid" style="margin-top:0;">'
        '<div class="sad-gw-stat sad-gw-stat-full">'
        '<div class="sad-gw-stat-label">Session total tokens</div>'
        f'<div class="sad-gw-stat-value">{total_session_tokens:,} '
        f'<span style="font-size:0.72rem;font-weight:400;color:#94A3B8;">'
        f'({total_prompt:,} in / {total_completion:,} out &bull; {req_count} reqs &bull; '
        f'{fallback_count} fallbacks)</span></div>'
        '</div>'
        '</div>'
        f'{latest_html}'
        f'{log_section}'
        '</div>'
    )
    placeholder.markdown(panel_html, unsafe_allow_html=True)


def format_credits(credits: float) -> str:
    """Format Tavily credit count cleanly (e.g. '3' instead of '3.0', '1.2' if fractional)."""
    rounded = round(float(credits), 2)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:g}"


def render_sidebar_credits(placeholder, credits_used: float) -> None:
    """Render or update the 'Tavily credits used this session' counter in the sidebar."""
    formatted = escape_html(format_credits(credits_used))
    placeholder.markdown(
        '<div class="sad-credits-card">'
        '<div class="sad-credits-label">Tavily credits used this session</div>'
        '<div class="sad-credits-row">'
        f'<span class="sad-credits-value">{formatted}</span>'
        '<span class="sad-credits-pill">6h cache active</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_score_ring_html(
    overall_score: int,
    models_used: list[str],
    fallback_routes: list[str] | None = None,
) -> str:
    """Build the pure HTML/CSS circular score ring and summary banner."""
    tier_label, ring_color, tier_desc = get_overall_tier(overall_score)
    clamped = max(0, min(100, overall_score))
    model_str = escape_html(", ".join(models_used)) if models_used else "Unknown"
    fallback_pills = "".join(
        f' <span class="sad-fallback-pill">Fallback used: {escape_html(route)}</span>'
        for route in (fallback_routes or [])
    )

    return (
        f'<div class="sad-score-card">'
        f'<div class="sad-ring-wrap" style="background: conic-gradient({ring_color} {clamped}%, #243047 0);">'
        f'<div class="sad-ring-inner">'
        f'<span class="sad-ring-num">{clamped}</span>'
        f'<span class="sad-ring-denom">of 100</span>'
        f'<span class="sad-ring-tier" style="color: {ring_color};">{tier_label}</span>'
        f'</div></div>'
        f'<div class="sad-score-meta">'
        f'<h3>Overall freshness: {tier_label}</h3>'
        f'<p>{escape_html(tier_desc)}</p>'
        f'<span class="sad-model-tag">Answered by: {model_str}</span>'
        f'{fallback_pills}'
        f'</div></div>'
    )


def render_claim_card(
    index: int,
    item: dict,
    show_query: bool = False,
    key_prefix: str = "audit",
) -> None:
    """Render a polished per-claim card with status pill, score bar, comparison, and chips."""
    status = item.get("status", "uncertain")
    meta = STATUS_META.get(status, STATUS_META["uncertain"])
    score = max(0, min(100, int(item.get("score", 50))))
    category = str(item.get("category", "other")).lower()
    cat_icon = CATEGORY_ICONS.get(category, CATEGORY_ICONS["other"])

    claimed_html = escape_html(item.get("claim", ""))
    actual_html = escape_html(item.get("current_info", ""))
    raw_reason = str(item.get("reason", ""))
    safe_reason = (
        format_friendly_ui_error(raw_reason, action="verifying this claim")
        if ("traceback" in raw_reason.lower() or 'file "' in raw_reason.lower())
        else raw_reason
    )
    reason_html = escape_html(safe_reason)

    error_block = ""
    if item.get("error"):
        safe_item_err = format_friendly_ui_error(item["error"], action="verifying this claim")
        error_block = (
            f'<div class="sad-error-note">'
            f'Verification issue: {escape_html(safe_item_err)}'
            f'</div>'
        )

    query_block = ""
    if show_query and item.get("search_query"):
        query_block = (
            f'<div class="sad-reason">'
            f'<strong>Search query:</strong> <code>{escape_html(item["search_query"])}</code>'
            f'</div>'
        )

    chips_parts: list[str] = []
    for src in item.get("sources", []):
        url = str(src.get("url", "")).strip()
        if not url:
            continue
        domain = escape_html(extract_domain(url))
        title_attr = escape_html(src.get("title") or url)
        safe_url = html.escape(url, quote=True)
        chips_parts.append(
            f'<a class="sad-source-chip" href="{safe_url}" target="_blank" '
            f'rel="noopener noreferrer" title="{title_attr}">'
            f'{domain} &#8599;</a>'
        )

    sources_block = (
        '<div class="sad-chips-row">' + "".join(chips_parts) + "</div>"
        if chips_parts
        else '<div class="sad-card-footer">No source links available.</div>'
    )

    model_name = item.get("model")
    gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else {}
    used_fallback = bool(item.get("fallback_used") or gw.get("fallback_used"))
    footer_parts = [f"Claim #{index}"]
    if model_name:
        footer_parts.append(f"Answered by: {escape_html(model_name)}")
    if used_fallback:
        fb_route = format_fallback_route(
            item.get("primary_model") or gw.get("primary_model"),
            model_name or gw.get("model_used"),
        )
        footer_parts.append(f'<span style="color:#FBBF24;">Fallback used: {escape_html(fb_route)}</span>')
    if gw.get("latency_ms"):
        footer_parts.append(f"{float(gw['latency_ms']):.0f} ms")
    footer_block = f'<div class="sad-card-footer">{" &bull; ".join(footer_parts)}</div>'

    confidence = str(item.get("confidence", "medium")).strip().lower()
    risk_level = str(item.get("risk_level", "moderate")).strip().lower()
    risk_cls = f"risk-{risk_level}" if risk_level in ("critical", "moderate", "low") else "risk-moderate"

    card_html = (
        f'<div class="sad-claim-card">'
        f'<div class="sad-claim-header">'
        f'<div class="sad-claim-badges">'
        f'<span class="sad-pill {meta["pill_class"]}">{meta["label"]}</span>'
        f'<span class="sad-cat-badge">{cat_icon}{escape_html(category)}</span>'
        f'<span class="sad-meta-pill">{escape_html(confidence.title())} confidence</span>'
        f'<span class="sad-meta-pill {risk_cls}">{escape_html(risk_level.title())} risk</span>'
        f'</div>'
        f'<div class="sad-score-bar-wrap">'
        f'<div class="sad-score-track">'
        f'<div class="sad-score-fill" style="width: {score}%; background: {meta["color"]};"></div>'
        f'</div>'
        f'<span class="sad-score-text">{score}/100</span>'
        f'</div></div>'
        f'<div class="sad-compare-grid">'
        f'<div class="sad-compare-box">'
        f'<div class="sad-compare-label">Claimed</div>'
        f'<div class="sad-compare-body">{claimed_html}</div>'
        f'</div>'
        f'<div class="sad-compare-box sad-compare-actual">'
        f'<div class="sad-compare-label">Actually (today)</div>'
        f'<div class="sad-compare-body">{actual_html}</div>'
        f'</div></div>'
        f'{error_block}'
        f'<div class="sad-reason"><strong>Reason:</strong> {reason_html}</div>'
        f'{query_block}'
        f'{sources_block}'
        f'{footer_block}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Explainable score breakdown inside a collapsible "Why this score?" expander.
    breakdown = item.get("score_breakdown")
    why_this_score = item.get("why_this_score", "")
    if breakdown or why_this_score:
        with st.expander("Why this score?", key=f"why_score_{key_prefix}_{index}", icon=":material/analytics:"):
            if why_this_score:
                st.markdown(escape_markdown(why_this_score))

            if isinstance(breakdown, dict):
                components = [
                    ("LLM verdict", "llm_verdict", "50%"),
                    ("Source recency", "source_recency", "20%"),
                    ("Source agreement", "source_agreement", "20%"),
                    ("Source authority", "source_authority", "10%"),
                ]
                for label, key, weight_label in components:
                    part = breakdown.get(key)
                    if not isinstance(part, dict):
                        continue
                    part_score = max(0, min(100, int(part.get("score", 0))))
                    weighted_pts = part.get("weighted_points", 0)
                    detail = str(part.get("detail", ""))
                    c_left, c_right = st.columns([2, 1])
                    with c_left:
                        st.markdown(f"**{label} ({weight_label})**")
                        if detail:
                            st.caption(escape_markdown(detail))
                    with c_right:
                        st.markdown(f"`{part_score}/100` → **+{weighted_pts} pts**")
                    st.progress(part_score / 100)

                st.caption(
                    f"Confidence: **{confidence.title()}**  •  "
                    f"Risk level: **{risk_level.title()}**  •  "
                    f"Final score: **{score}/100**"
                )

                # Show Tavily strategy, cache status, and Extract URL metadata when available.
                meta_parts: list[str] = []
                if item.get("search_strategy"):
                    meta_parts.append(f"Search: `{item['search_strategy']}`")
                if item.get("cache_hit"):
                    meta_parts.append("Cache hit (`0` credits)")
                elif item.get("tavily_credits_used") is not None:
                    meta_parts.append(f"Tavily credits: `{format_credits(item['tavily_credits_used'])}`")
                if item.get("extracted_url"):
                    meta_parts.append(f"Extracted: `{extract_domain(item['extracted_url'])}`")
                if meta_parts:
                    st.caption("  •  ".join(meta_parts))
                if item.get("search_warning"):
                    st.caption(f"Partial search notice: {escape_markdown(item['search_warning'])}")


def run_stale_audit(
    answer_text: str,
    status_widget,
    steps_log: list[str],
    credits_placeholder,
    gateway_placeholder,
) -> tuple[str | None, dict | None, list[dict]]:
    """
    Reuse the shared extractor + verifier pipeline on `answer_text`, streaming live
    progress into `status_widget`, sidebar Tavily credits, and the Bifrost Gateway panel.
    """
    st.write("Extracting claims...")
    claims, extract_model = extract_claims(answer_text)
    extract_record = build_request_telemetry(
        "Extract claims",
        get_extract_metrics(),
        model_used=extract_model,
    )
    append_gateway_request(extract_record)
    render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

    extract_msg = f"Extracted {len(claims)} claim(s) ({extract_model})"
    if extract_record and extract_record.get("fallback_used"):
        fb_route = format_fallback_route(
            extract_record.get("primary_model"),
            extract_record.get("model_used"),
        )
        extract_msg += f" — Fallback used: {fb_route}"
    steps_log.append(extract_msg)
    st.write(f":material/check: {extract_msg}")

    if not claims:
        status_widget.update(
            label="Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )
        return extract_model, extract_record, []

    total = len(claims)
    status_widget.update(label=f"Searching Tavily (1/{total})", state="running")
    live_step = st.empty()
    live_step.write(f"Searching Tavily (1/{total})...")

    progress_calls = [0]

    def on_progress(idx: int, total_count: int, result: dict) -> None:
        progress_calls[0] += 1

        claim_credits = float(result.get("tavily_credits_used", 0.0) or 0.0)
        if claim_credits > 0:
            st.session_state["tavily_credits_used"] = round(
                float(st.session_state.get("tavily_credits_used", 0.0)) + claim_credits,
                2,
            )
            render_sidebar_credits(
                credits_placeholder,
                st.session_state["tavily_credits_used"],
            )

        claim_record = build_request_telemetry(
            f"Verify claim #{idx}",
            result.get("gateway_metrics"),
            model_used=result.get("model"),
            fallback_used=bool(result.get("fallback_used", False)),
            primary_model=result.get("primary_model"),
        )
        if claim_record:
            append_gateway_request(claim_record)
            render_gateway_panel(
                gateway_placeholder,
                st.session_state["gateway_requests"],
            )

        if idx < total_count:
            status_widget.update(
                label=f"Searching Tavily ({idx + 1}/{total_count})",
                state="running",
            )
            live_step.write(f"Searching Tavily ({idx + 1}/{total_count})...")
        else:
            status_widget.update(label="Judging freshness", state="running")
            live_step.write("Judging freshness...")

        step_text = (
            f"Claim {idx}/{total_count}: {result.get('status', 'uncertain').upper()} "
            f"({result.get('score', 50)}/100)"
        )
        steps_log.append(step_text)

    verified_results = verify_claims(
        claims,
        progress_callback=on_progress,
    )
    if progress_calls[0] == 0:
        batch_credits = sum(
            float(r.get("tavily_credits_used", 0.0) or 0.0)
            for r in verified_results
        )
        if batch_credits > 0:
            st.session_state["tavily_credits_used"] = round(
                float(st.session_state.get("tavily_credits_used", 0.0)) + batch_credits,
                2,
            )
            render_sidebar_credits(
                credits_placeholder,
                st.session_state["tavily_credits_used"],
            )
        batch_added = False
        for b_idx, r in enumerate(verified_results, start=1):
            rec = build_request_telemetry(
                f"Verify claim #{b_idx}",
                r.get("gateway_metrics"),
                model_used=r.get("model"),
                fallback_used=bool(r.get("fallback_used", False)),
                primary_model=r.get("primary_model"),
            )
            if rec:
                append_gateway_request(rec)
                batch_added = True
        if batch_added:
            render_gateway_panel(
                gateway_placeholder,
                st.session_state["gateway_requests"],
            )

    live_step.write(":material/check: Judging freshness complete")
    steps_log.append("Done")
    status_widget.update(
        label="Done",
        state="complete",
        expanded=st.session_state["keep_status_expanded"],
    )
    return extract_model, extract_record, verified_results


def render_audit_results(
    results: list[dict] | None,
    extract_model: str | None,
    extract_meta: dict | None,
    key_prefix: str,
    just_clicked: bool = False,
    empty_title: str = "Ready to verify",
    empty_body: str = "Paste an LLM answer on the left or pick a demo in the sidebar, then click <strong>Check Freshness</strong>.",
    extra_models: list[str | None] | None = None,
    extra_metas: list[dict | None] | None = None,
) -> None:
    """Render shared freshness score ring, KPI grid, failover banners, and per-claim cards."""
    if results is None:
        if not just_clicked:
            st.markdown(
                f'<div class="sad-empty-state">'
                f'<h4>{escape_html(empty_title)}</h4>'
                f'<p>{empty_body}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        return

    if len(results) == 0:
        st.info(
            "No potentially outdated or time-sensitive factual claims were found in this answer.",
            icon=":material/info:",
        )
        if extract_model:
            extract_caption = f"Answered by: {extract_model}"
            if isinstance(extract_meta, dict) and extract_meta.get("fallback_used"):
                fb_route = format_fallback_route(
                    extract_meta.get("primary_model"),
                    extract_meta.get("model_used"),
                )
                extract_caption += f"  •  Fallback used: {fb_route}"
            st.caption(escape_markdown(extract_caption))
        return

    overall_score = round(sum(r["score"] for r in results) / len(results))
    fresh_count = sum(1 for r in results if r["status"] == "fresh")
    uncertain_count = sum(1 for r in results if r["status"] == "uncertain")
    outdated_count = sum(1 for r in results if r["status"] == "outdated")
    error_count = sum(1 for r in results if r.get("error"))

    models_used: list[str] = []
    for model_name in [
        *(extra_models or []),
        extract_model,
        *(r.get("model") for r in results),
    ]:
        if model_name and model_name not in models_used:
            models_used.append(model_name)

    fallback_routes = collect_fallback_routes(extract_meta, results)
    for meta in extra_metas or []:
        if isinstance(meta, dict) and meta.get("fallback_used"):
            route = format_fallback_route(meta.get("primary_model"), meta.get("model_used"))
            if route not in fallback_routes:
                fallback_routes.append(route)

    st.markdown(
        render_score_ring_html(overall_score, models_used, fallback_routes=fallback_routes),
        unsafe_allow_html=True,
    )
    if models_used:
        caption_parts = [f"Answered by: {', '.join(models_used)}"]
        for route in fallback_routes:
            caption_parts.append(f"Fallback used: {route}")
        st.caption(escape_markdown("  •  ".join(caption_parts)))

    if fallback_routes:
        failover_prefix = (
            "Bifrost gateway automatically handled provider failover ("
            if is_use_bifrost()
            else "Model fallback chain automatically recovered the request ("
        )
        st.info(
            escape_markdown(
                failover_prefix
                + ", ".join(f"Fallback used: {r}" for r in fallback_routes)
                + ")."
            ),
            icon=":material/swap_horiz:",
        )

    with st.grid(4, min_column_width=110, gap="small"):
        st.metric("Claims checked", len(results), border=True, height="stretch")
        st.metric("Fresh", fresh_count, border=True, height="stretch")
        st.metric("Uncertain", uncertain_count, border=True, height="stretch")
        st.metric("Outdated", outdated_count, border=True, height="stretch")

    if error_count > 0:
        friendly_reasons: list[str] = []
        for err_item in results:
            if err_item.get("error"):
                friendly_msg = format_friendly_ui_error(err_item["error"], action="verifying claims")
                if friendly_msg not in friendly_reasons:
                    friendly_reasons.append(friendly_msg)
        summary_msg = f"{error_count} claim(s) encountered a verification error and were marked as uncertain."
        if friendly_reasons:
            summary_msg += " " + " ".join(friendly_reasons)
        st.warning(summary_msg, icon=":material/warning:")

    st.space("small")
    show_queries = st.session_state.get("show_search_queries", False)
    for idx, item in enumerate(results, start=1):
        render_claim_card(idx, item, show_query=show_queries, key_prefix=key_prefix)


def render_corrected_and_export_section(
    mode: str,
    key_prefix: str,
    original_answer: str,
    results: list[dict] | None,
    corrected_answer: str | None,
    corrected_model: str | None,
    corrected_meta: dict | None,
    question: str | None = None,
    ask_meta: dict | None = None,
    extract_model: str | None = None,
    extract_meta: dict | None = None,
    audit_at: datetime | None = None,
) -> None:
    """
    Render the full-width corrected-answer side-by-side word diff, 'What changed & why'
    source-backed changelog, and Markdown/JSON audit report download buttons.
    """
    if results is None:
        return

    if corrected_answer:
        st.divider()
        st.subheader("Original vs. corrected answer", icon=":material/compare_arrows:")

        diff_result = build_side_by_side_word_diff(original_answer or "", corrected_answer)
        diff_stats = diff_result["stats"]

        meta_notes: list[str] = [
            f"-{diff_stats['removed_words']} removed / +{diff_stats['added_words']} added word(s)"
        ]
        if corrected_model:
            meta_notes.append(f"Answered by: {corrected_model}")
        if isinstance(corrected_meta, dict) and corrected_meta.get("fallback_used"):
            fb_route = format_fallback_route(
                corrected_meta.get("primary_model"),
                corrected_meta.get("model_used") or corrected_model,
            )
            meta_notes.append(f"Fallback used: {fb_route}")
        st.caption(escape_markdown("  •  ".join(meta_notes)))

        orig_col, corr_col = st.columns(2, gap="medium")
        with orig_col:
            st.markdown(
                '<div class="sad-diff-card sad-diff-card-orig">'
                '<div class="sad-diff-header">'
                '<span class="sad-diff-title">Original answer</span>'
                '<span class="sad-diff-tag sad-diff-tag-del">Removed in red</span>'
                '</div>'
                f'<div class="sad-diff-body">{diff_result["original_html"]}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with corr_col:
            st.markdown(
                '<div class="sad-diff-card sad-diff-card-new">'
                '<div class="sad-diff-header">'
                '<span class="sad-diff-title">Corrected answer</span>'
                '<span class="sad-diff-tag sad-diff-tag-ins">New text in green</span>'
                '</div>'
                f'<div class="sad-diff-body">{diff_result["corrected_html"]}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        with st.expander(
            "Copy plain corrected text",
            key=f"copy_plain_{key_prefix}",
            icon=":material/content_copy:",
        ):
            st.code(corrected_answer, language=None, wrap_lines=True)

        st.subheader("What changed & why", icon=":material/difference:")
        changes = build_change_log(results)
        if changes:
            for ch in changes:
                pill_cls = (
                    "sad-change-pill-replaced"
                    if ch["status"] == "outdated"
                    else "sad-change-pill-qualified"
                )
                cat_icon = CATEGORY_ICONS.get(ch["category"], CATEGORY_ICONS["other"])
                chips_html: list[str] = []
                for src in ch.get("sources") or []:
                    src_url = str(src.get("url", "")).strip()
                    if not src_url:
                        continue
                    safe_url = html.escape(src_url, quote=True)
                    domain = escape_html(extract_domain(src_url))
                    title_attr = escape_html(src.get("title") or src_url)
                    chips_html.append(
                        f'<a class="sad-source-chip" href="{safe_url}" target="_blank" '
                        f'rel="noopener noreferrer" title="{title_attr}">'
                        f'{domain} &#8599;</a>'
                    )
                sources_block = (
                    '<div class="sad-chips-row">' + "".join(chips_html) + "</div>"
                    if chips_html
                    else '<div class="sad-card-footer">No source links available.</div>'
                )

                st.markdown(
                    '<div class="sad-change-card">'
                    '<div class="sad-change-top">'
                    f'<span class="sad-pill {pill_cls}">{escape_html(ch["action"])}</span>'
                    f'<span class="sad-cat-badge">{cat_icon}{escape_html(ch["category"])}</span>'
                    f'<span class="sad-meta-pill">Claim #{ch["claim_index"]} &bull; {ch["score"]}/100</span>'
                    '</div>'
                    '<div class="sad-change-diff-row">'
                    '<div class="sad-compare-box">'
                    '<div class="sad-compare-label">Before (original claim)</div>'
                    f'<div class="sad-compare-body">{escape_html(ch["original_claim"])}</div>'
                    '</div>'
                    '<div class="sad-compare-box sad-compare-actual">'
                    '<div class="sad-compare-label">After (verified today)</div>'
                    f'<div class="sad-compare-body">{escape_html(ch["updated_to"])}</div>'
                    '</div>'
                    '</div>'
                    f'<div class="sad-reason"><strong>Why:</strong> {escape_html(ch["why"])}</div>'
                    f'{sources_block}'
                    '</div>',
                    unsafe_allow_html=True,
                )

            fresh_kept = sum(1 for r in results if r.get("status") == "fresh")
            if fresh_kept > 0:
                st.caption(f"Kept {fresh_kept} verified fresh claim(s) unchanged.")
        else:
            st.info(
                "No outdated or uncertain claims needed factual updates — all checked claims stayed fresh.",
                icon=":material/check_circle:",
            )

    # Report export buttons (available as soon as an audit completes, and automatically
    # enriched with the diff/changelog/rewrite telemetry once a corrected answer is generated).
    report_time = audit_at or datetime.now().astimezone()
    report_payload = build_report_payload(
        mode=mode,
        original_answer=original_answer or "",
        verification_results=results,
        question=question,
        corrected_answer=corrected_answer,
        ask_meta=ask_meta,
        extract_model=extract_model,
        extract_meta=extract_meta,
        corrected_model=corrected_model,
        corrected_meta=corrected_meta,
        generated_at=report_time,
    )
    md_report = format_report_markdown(report_payload)
    json_report = format_report_json(report_payload)
    ts_slug = report_time.strftime("%Y%m%d-%H%M%S")

    st.space("small")
    st.subheader("Export audit report", icon=":material/download:")
    models_label = ", ".join(report_payload["models_used"]) or "N/A"
    tokens_total = report_payload["tokens_used"]["total_tokens"]
    st.caption(
        escape_markdown(
            f"Includes {report_payload['summary']['claims_checked']} claim(s), scores, sources, "
            f"timestamps ({report_payload['generated_at']}), models ({models_label}), "
            f"and {tokens_total:,} tokens."
        )
    )

    dl_col1, dl_col2 = st.columns(2, gap="small")
    with dl_col1:
        st.download_button(
            "Download report (Markdown)",
            data=md_report,
            file_name=f"stale-answer-report-{ts_slug}.md",
            mime="text/markdown",
            key=f"dl_md_{key_prefix}",
            icon=":material/download:",
            on_click="ignore",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            "Download JSON",
            data=json_report,
            file_name=f"stale-answer-report-{ts_slug}.json",
            mime="application/json",
            key=f"dl_json_{key_prefix}",
            icon=":material/data_object:",
            on_click="ignore",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# App setup & custom CSS injection
# ---------------------------------------------------------------------------
init_session_state()
use_bifrost_mode = is_use_bifrost()
set_simulate_outage(bool(st.session_state.get("simulate_outage", False)) and use_bifrost_mode)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: 4 demo examples, Bifrost/Gemini Routing panel, Tavily credits, How it works, Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Demo examples", icon=":material/science:")
    st.caption("Click an example to populate the input box:")

    for demo_idx, demo in enumerate(DEMO_EXAMPLES):
        st.button(
            demo["label"],
            key=f"demo_example_{demo_idx}",
            icon=demo["icon"],
            on_click=load_demo_example,
            args=(demo["text"],),
            width="stretch",
        )

    st.subheader("Gateway panel", icon=":material/hub:")
    gateway_health = check_gateway_health(timeout=0.5)
    dashboard_url = get_gateway_dashboard_url()
    render_gateway_health_card(gateway_health, dashboard_url, use_bifrost=use_bifrost_mode)

    if use_bifrost_mode and not gateway_health.get("healthy"):
        st.caption(
            "Bifrost looks unreachable on port 8080. Start Bifrost locally or set "
            "`USE_BIFROST=false` to use Gemini directly."
        )

    outage_enabled = st.toggle(
        "Simulate provider outage",
        key="simulate_outage",
        disabled=(not use_bifrost_mode),
        help=(
            "Forces the primary model to fail so Bifrost switches to the next fallback model."
            if use_bifrost_mode
            else "Available only when USE_BIFROST=true (disabled in Direct Gemini mode)."
        ),
    )
    set_simulate_outage(bool(outage_enabled) and use_bifrost_mode)

    try:
        _, resolved_base_url, resolved_chain = get_llm_config()
    except ValueError:
        resolved_chain = []
        resolved_base_url = None

    configured_fallbacks = (
        resolved_chain[1:]
        if resolved_chain
        else [m.strip() for m in _get_config_value("LLM_MODEL_FALLBACKS").split(",") if m.strip()]
    )
    if use_bifrost_mode and outage_enabled:
        primary_cfg = resolved_chain[0] if resolved_chain else (_get_config_value("LLM_MODEL") or "gpt-4o-mini")
        if configured_fallbacks:
            st.caption(
                f"Simulating primary failure on `{primary_cfg}` → fallback: `{configured_fallbacks[0]}`"
            )
        else:
            st.warning(
                "Set LLM_MODEL_FALLBACKS in .env so Bifrost has a fallback model to switch to.",
                icon=":material/warning:",
            )

    gateway_placeholder = st.empty()
    render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

    credits_placeholder = st.empty()
    render_sidebar_credits(credits_placeholder, st.session_state["tavily_credits_used"])

    st.subheader("How it works", icon=":material/route:")
    st.markdown(
        "1. **Extract claims** — Identifies time-sensitive facts (prices, versions, laws, CEOs, stats).\n"
        "2. **Verify via Tavily** — Uses category-smart search + page extraction (6h cache).\n"
        "3. **Judge freshness** — Scores accuracy (`0–100`) against today's date.\n"
        "4. **Rewrite answer** — Updates stale facts with verified current info."
    )

    with st.expander("Settings", icon=":material/settings:"):
        routing_label = (
            "Bifrost gateway (`USE_BIFROST=true`)"
            if use_bifrost_mode
            else "Direct Gemini (`USE_BIFROST=false`)"
        )
        primary_model = (
            resolved_chain[0]
            if resolved_chain
            else (_get_config_value("LLM_MODEL") or ("gpt-4o-mini" if use_bifrost_mode else "gemini-2.5-flash"))
        )
        fallbacks_label = ", ".join(resolved_chain[1:]) if len(resolved_chain) > 1 else (_get_config_value("LLM_MODEL_FALLBACKS") or "None")
        endpoint_display = (
            resolved_base_url
            or _get_config_value("LLM_BASE_URL")
            or ("Default OpenAI" if use_bifrost_mode else "https://generativelanguage.googleapis.com/v1beta/openai/")
        )
        llm_key_ok = bool(
            _get_config_value("LLM_API_KEY")
            or (not use_bifrost_mode and _get_config_value("GEMINI_API_KEY"))
        )
        tavily_ok = bool(_get_config_value("TAVILY_API_KEY"))
        vk_ok = bool(_get_config_value("LLM_VIRTUAL_KEY")) if use_bifrost_mode else False
        vk_status = (
            ("Configured" if vk_ok else "Not set")
            if use_bifrost_mode
            else "Bypassed (Direct Gemini)"
        )

        st.caption(
            f"**Routing mode:** {routing_label}  \n"
            f"**Primary model:** `{primary_model}`  \n"
            f"**Fallbacks:** `{fallbacks_label}`  \n"
            f"**Endpoint:** `{endpoint_display}`  \n"
            f"**LLM / Gemini key:** {'Configured' if llm_key_ok else 'Missing'}  \n"
            f"**Virtual key (`x-bf-vk`):** {vk_status}  \n"
            f"**Tavily key:** {'Configured' if tavily_ok else 'Missing'}  \n"
            "**Tavily cache:** `6h on-disk SQLite TTL`"
        )
        st.toggle("Show search queries on cards", key="show_search_queries")
        st.toggle("Keep status log expanded", key="keep_status_expanded")
        if st.button("Clear Tavily cache", key="clear_tavily_cache_btn", width="stretch"):
            removed = clear_tavily_cache()
            st.caption(f"Cleared {removed} cached Tavily entr{'y' if removed == 1 else 'ies'}.")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
routing_badge_text = "Routed via Bifrost" if use_bifrost_mode else "Direct Gemini API"
st.markdown(
    '<div class="sad-hero">'
    '<div class="sad-hero-top">'
    '<div>'
    '<h1 class="sad-hero-title">Stale-Answer Detector</h1>'
    '<p class="sad-hero-tagline">'
    'Verify time-sensitive AI claims against live web sources and rewrite outdated answers with today&#39;s facts.'
    '</p>'
    '</div>'
    '<div class="sad-hero-badges">'
    '<span class="sad-tech-badge"><span class="sad-dot"></span>Powered by Tavily</span>'
    f'<span class="sad-tech-badge"><span class="sad-dot"></span>{routing_badge_text}</span>'
    '</div>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Two-tab main workspace:
# 1. "Audit pasted answer" (existing two-column flow)
# 2. "Ask & Audit" (ask question via Bifrost -> show answer -> auto-audit + highlight)
# ---------------------------------------------------------------------------
tab_pasted, tab_ask = st.tabs(["Audit pasted answer", "Ask & Audit"])

with tab_pasted:
    left_col, right_col = st.columns([1, 1.35], gap="large")

    with left_col:
        st.subheader("Input answer", icon=":material/edit_note:")
        st.text_area(
            "LLM-generated answer",
            key="answer_input",
            height=230,
            label_visibility="collapsed",
            placeholder=(
                "Paste an LLM-generated answer here (or choose one of the 4 demo "
                "examples in the sidebar)..."
            ),
        )

        btn_col1, btn_col2 = st.columns([3, 1], gap="small")
        with btn_col1:
            check_clicked = st.button(
                "Check Freshness",
                key="check_freshness_btn",
                type="primary",
                icon=":material/verified:",
                width="stretch",
            )
        with btn_col2:
            st.button(
                "Clear",
                key="clear_workspace_btn",
                icon=":material/delete_outline:",
                on_click=clear_workspace,
                width="stretch",
            )

        # Show the "Generate corrected answer" action once verified claims exist.
        stored_results = st.session_state.get("results")
        if stored_results:
            st.space("small")
            if st.button(
                "Generate corrected answer",
                key="generate_corrected_btn",
                icon=":material/auto_fix_high:",
                width="stretch",
            ):
                try:
                    with st.spinner("Rewriting answer with verified current info..."):
                        corrected, corrected_model = generate_corrected_answer(
                            st.session_state["checked_answer"],
                            stored_results,
                        )
                        st.session_state["corrected_answer"] = corrected
                        st.session_state["corrected_model"] = corrected_model
                        rewrite_record = build_request_telemetry(
                            "Rewrite answer",
                            get_rewrite_metrics(),
                            model_used=corrected_model,
                        )
                        st.session_state["corrected_meta"] = rewrite_record
                        st.session_state["pasted_audit_at"] = datetime.now().astimezone()
                        append_gateway_request(rewrite_record)
                        render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])
                except Exception as exc:
                    st.error(
                        format_friendly_ui_error(exc, action="generating the corrected answer"),
                        icon=":material/error:",
                    )

    with right_col:
        st.subheader("Verification results", icon=":material/fact_check:")

        if check_clicked:
            answer_text = st.session_state["answer_input"].strip()
            if not answer_text:
                st.warning(
                    "Please paste an LLM-generated answer (or pick a sidebar demo) first.",
                    icon=":material/warning:",
                )
            else:
                st.session_state["results"] = None
                st.session_state["extract_model"] = None
                st.session_state["extract_meta"] = None
                st.session_state["corrected_answer"] = None
                st.session_state["corrected_model"] = None
                st.session_state["corrected_meta"] = None
                st.session_state["pasted_audit_at"] = None
                st.session_state["status_steps"] = []

                status = None
                try:
                    with st.status("Extracting claims", expanded=True) as status:
                        steps_log: list[str] = []
                        extract_model, extract_record, verified_results = run_stale_audit(
                            answer_text=answer_text,
                            status_widget=status,
                            steps_log=steps_log,
                            credits_placeholder=credits_placeholder,
                            gateway_placeholder=gateway_placeholder,
                        )
                        st.session_state["extract_model"] = extract_model
                        st.session_state["extract_meta"] = extract_record
                        st.session_state["results"] = verified_results
                        st.session_state["checked_answer"] = answer_text
                        st.session_state["pasted_audit_at"] = datetime.now().astimezone()
                        st.session_state["status_steps"] = steps_log

                except Exception as exc:
                    if status is not None:
                        status.update(label="Check failed", state="error", expanded=False)
                    st.error(
                        format_friendly_ui_error(exc, action="verifying your answer"),
                        icon=":material/error:",
                    )

        # Render collapsed status trail on subsequent reruns (e.g., after rewriting answer).
        elif st.session_state.get("status_steps"):
            with st.status(
                "Done",
                state="complete",
                expanded=st.session_state["keep_status_expanded"],
            ):
                for step_line in st.session_state["status_steps"]:
                    st.write(step_line)

        # Render stored verification results (score ring, 4 metric cards, claim cards).
        render_audit_results(
            results=st.session_state.get("results"),
            extract_model=st.session_state.get("extract_model"),
            extract_meta=st.session_state.get("extract_meta"),
            key_prefix="pasted",
            just_clicked=check_clicked,
        )

    render_corrected_and_export_section(
        mode="Audit pasted answer",
        key_prefix="pasted",
        original_answer=st.session_state.get("checked_answer") or st.session_state.get("answer_input", ""),
        results=st.session_state.get("results"),
        corrected_answer=st.session_state.get("corrected_answer"),
        corrected_model=st.session_state.get("corrected_model"),
        corrected_meta=st.session_state.get("corrected_meta"),
        extract_model=st.session_state.get("extract_model"),
        extract_meta=st.session_state.get("extract_meta"),
        audit_at=st.session_state.get("pasted_audit_at"),
    )

route_target = "Bifrost" if use_bifrost_mode else "Gemini"
with tab_ask:
    st.caption(
        f"Type a question below. The app fetches an answer via {route_target}, shows it, "
        "runs the live stale-claim audit automatically, and highlights outdated vs. fresh claims inline."
    )

    q_col, ask_btn_col, ask_clr_col = st.columns(
        [4, 1.3, 0.75],
        vertical_alignment="bottom",
        gap="small",
    )
    with q_col:
        st.text_input(
            "Ask a question",
            key="ask_question_input",
            label_visibility="collapsed",
            placeholder="e.g. What is the latest iPhone model, chip, and US starting price?",
        )
    with ask_btn_col:
        ask_clicked = st.button(
            "Ask & Audit",
            key="ask_and_audit_btn",
            type="primary",
            icon=":material/send:",
            width="stretch",
        )
    with ask_clr_col:
        st.button(
            "Clear",
            key="ask_clear_btn",
            icon=":material/delete_outline:",
            on_click=clear_ask_workspace,
            width="stretch",
        )

    ask_left, ask_right = st.columns([1, 1.35], gap="large")

    with ask_left:
        st.subheader("LLM answer", icon=":material/chat_bubble_outline:")
        ask_answer_slot = st.empty()
        ask_rewrite_container = st.container()
        question_ready = bool(st.session_state.get("ask_question_input", "").strip())

        if not (ask_clicked and question_ready):
            saved_ask_answer = st.session_state.get("ask_answer", "")
            if saved_ask_answer:
                render_ask_answer_card(
                    placeholder=ask_answer_slot,
                    answer_text=saved_ask_answer,
                    verification_results=st.session_state.get("ask_results"),
                    question_text=st.session_state.get("ask_question_asked"),
                    answer_model=st.session_state.get("ask_model"),
                    answer_meta=st.session_state.get("ask_meta"),
                )
            else:
                ask_answer_slot.markdown(
                    '<div class="sad-empty-state">'
                    '<h4>Ask any question</h4>'
                    f'<p>Enter a question above and click <strong>Ask &amp; Audit</strong> '
                    f'to inspect live {route_target} output with inline claim highlights.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

    with ask_right:
        st.subheader("Verification audit", icon=":material/fact_check:")

        if ask_clicked:
            question_text = st.session_state["ask_question_input"].strip()
            if not question_text:
                st.warning(
                    "Please enter a question to ask and audit.",
                    icon=":material/warning:",
                )
            else:
                st.session_state["ask_question_asked"] = question_text
                st.session_state["ask_answer"] = ""
                st.session_state["ask_model"] = None
                st.session_state["ask_meta"] = None
                st.session_state["ask_results"] = None
                st.session_state["ask_extract_model"] = None
                st.session_state["ask_extract_meta"] = None
                st.session_state["ask_corrected_answer"] = None
                st.session_state["ask_corrected_model"] = None
                st.session_state["ask_corrected_meta"] = None
                st.session_state["ask_audit_at"] = None
                st.session_state["ask_status_steps"] = []

                ask_status = None
                try:
                    with st.status(f"Asking LLM via {route_target}", expanded=True) as ask_status:
                        ask_steps: list[str] = []
                        st.write(f"Getting answer from LLM via {route_target}...")

                        llm_answer, raw_ask_meta = ask_question(question_text)
                        ask_record = build_request_telemetry(
                            "Ask & Audit Q&A",
                            raw_ask_meta,
                            model_used=raw_ask_meta.get("model_used") if isinstance(raw_ask_meta, dict) else None,
                        )
                        ask_model = ask_record.get("model_used") if ask_record else None

                        st.session_state["ask_answer"] = llm_answer
                        st.session_state["ask_model"] = ask_model
                        st.session_state["ask_meta"] = ask_record
                        append_gateway_request(ask_record)
                        render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

                        ask_msg = f"Answered via {route_target} ({ask_model})"
                        if ask_record and ask_record.get("fallback_used"):
                            fb_route = format_fallback_route(
                                ask_record.get("primary_model"),
                                ask_record.get("model_used"),
                            )
                            ask_msg += f" — Fallback used: {fb_route}"
                        ask_steps.append(ask_msg)
                        st.write(f":material/check: {ask_msg}")

                        # Show the raw LLM answer immediately before the Tavily audit runs.
                        render_ask_answer_card(
                            placeholder=ask_answer_slot,
                            answer_text=llm_answer,
                            verification_results=None,
                            question_text=question_text,
                            answer_model=ask_model,
                            answer_meta=ask_record,
                        )

                        # Automatically run the stale-claim audit using existing extractor + verifier.
                        ask_status.update(label="Extracting claims", state="running")
                        ext_model, ext_record, ask_verified = run_stale_audit(
                            answer_text=llm_answer,
                            status_widget=ask_status,
                            steps_log=ask_steps,
                            credits_placeholder=credits_placeholder,
                            gateway_placeholder=gateway_placeholder,
                        )

                        st.session_state["ask_extract_model"] = ext_model
                        st.session_state["ask_extract_meta"] = ext_record
                        st.session_state["ask_results"] = ask_verified
                        st.session_state["ask_audit_at"] = datetime.now().astimezone()
                        st.session_state["ask_status_steps"] = ask_steps

                        # Re-render the answer card with inline red/green claim highlights!
                        render_ask_answer_card(
                            placeholder=ask_answer_slot,
                            answer_text=llm_answer,
                            verification_results=ask_verified,
                            question_text=question_text,
                            answer_model=ask_model,
                            answer_meta=ask_record,
                        )

                except Exception as exc:
                    if ask_status is not None:
                        ask_status.update(label="Ask & Audit failed", state="error", expanded=False)
                    if st.session_state.get("ask_answer"):
                        render_ask_answer_card(
                            placeholder=ask_answer_slot,
                            answer_text=st.session_state["ask_answer"],
                            verification_results=None,
                            question_text=st.session_state.get("ask_question_asked"),
                            answer_model=st.session_state.get("ask_model"),
                            answer_meta=st.session_state.get("ask_meta"),
                            audit_error=True,
                        )
                    st.error(
                        format_friendly_ui_error(exc, action="running Ask & Audit"),
                        icon=":material/error:",
                    )

        elif st.session_state.get("ask_status_steps"):
            with st.status(
                "Done",
                state="complete",
                expanded=st.session_state["keep_status_expanded"],
            ):
                for step_line in st.session_state["ask_status_steps"]:
                    st.write(step_line)

        render_audit_results(
            results=st.session_state.get("ask_results"),
            extract_model=st.session_state.get("ask_extract_model"),
            extract_meta=st.session_state.get("ask_extract_meta"),
            key_prefix="ask",
            just_clicked=ask_clicked,
            empty_title="Ready to ask & audit",
            empty_body=(
                f"Type any question above and click <strong>Ask &amp; Audit</strong> to fetch "
                f"an LLM response via {route_target} and highlight stale facts automatically."
            ),
            extra_models=[st.session_state.get("ask_model")],
            extra_metas=[st.session_state.get("ask_meta")],
        )

    # Render the optional rewrite button inside ask_left immediately once audited.
    with ask_rewrite_container:
        ask_stored_results = st.session_state.get("ask_results")
        if ask_stored_results and st.session_state.get("ask_answer"):
            st.space("small")
            if st.button(
                "Rewrite answer with verified facts",
                key="ask_generate_corrected_btn",
                icon=":material/auto_fix_high:",
                width="stretch",
            ):
                try:
                    with st.spinner("Rewriting answer with verified current info..."):
                        corrected, corrected_model = generate_corrected_answer(
                            st.session_state["ask_answer"],
                            ask_stored_results,
                        )
                        st.session_state["ask_corrected_answer"] = corrected
                        st.session_state["ask_corrected_model"] = corrected_model
                        rewrite_record = build_request_telemetry(
                            "Rewrite Q&A answer",
                            get_rewrite_metrics(),
                            model_used=corrected_model,
                        )
                        st.session_state["ask_corrected_meta"] = rewrite_record
                        st.session_state["ask_audit_at"] = datetime.now().astimezone()
                        append_gateway_request(rewrite_record)
                        render_gateway_panel(
                            gateway_placeholder,
                            st.session_state["gateway_requests"],
                        )
                except Exception as exc:
                    st.error(
                        format_friendly_ui_error(exc, action="generating the corrected answer"),
                        icon=":material/error:",
                    )

    render_corrected_and_export_section(
        mode="Ask & Audit",
        key_prefix="ask",
        original_answer=st.session_state.get("ask_answer", ""),
        results=st.session_state.get("ask_results"),
        corrected_answer=st.session_state.get("ask_corrected_answer"),
        corrected_model=st.session_state.get("ask_corrected_model"),
        corrected_meta=st.session_state.get("ask_corrected_meta"),
        question=st.session_state.get("ask_question_asked"),
        ask_meta=st.session_state.get("ask_meta"),
        extract_model=st.session_state.get("ask_extract_model"),
        extract_meta=st.session_state.get("ask_extract_meta"),
        audit_at=st.session_state.get("ask_audit_at"),
    )
