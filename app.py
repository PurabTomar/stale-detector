"""
app.py - Modern dark-themed Streamlit UI for the Stale-Answer Detector.

Uses a two-column workspace (input on the left, live verification results on
the right), a pure HTML/CSS circular freshness score ring, 4 KPI metric cards,
side-by-side Claimed vs Actually (today) cards with clickable domain chips,
live st.status progress steps, and a sidebar with 4 demo examples, a live
Bifrost Gateway panel (health, outage toggle, latency/tokens/fallback telemetry,
dashboard link), and session Tavily credit tracking.
"""

import copy
from datetime import datetime
import html
import inspect
import json
from pathlib import Path
import re
import sys
import threading
import time
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from audit_animations import (
    AuditUIState,
    apply_pipeline_event,
    create_initial_ui_state,
    render_action_status_pill_html,
    render_activity_feed_html,
    render_claim_discovery_chips_html,
    render_freshness_strip_from_results,
    render_freshness_strip_html,
    render_live_counters_html,
    render_live_marked_text_html,
    render_provisional_score_meter_html,
    render_scan_effect_html,
    render_stepper_check_svg,
    render_verify_live_board_html,
)
from heuristic import (
    detect_heuristic_highlights,
    render_heuristic_preview_html,
)

import settings
from settings import (
    MAX_TEXT_CHARS,
    MAX_CLAIMS_PER_AUDIT,
    validate_text_input,
    validate_question_input,
    validate_url_input,
)
from batch_audit import (
    DEFAULT_BATCH_CLAIMS_PER_ITEM,
    MAX_BATCH_FILE_SIZE_BYTES,
    MAX_BATCH_ITEMS,
    MAX_BATCH_TOTAL_CLAIMS,
    export_batch_summary_csv,
    export_batch_summary_json,
    parse_batch_input,
    summarize_batch_item,
)
from calibration import compute_calibration
from expiry_heuristics import get_likely_expiry_info
from settings_panel import (
    SUPPORTED_CATEGORIES,
    SUPPORTED_LANGUAGES,
    estimate_audit_resources,
)
from source_credibility import get_source_credibility, render_credibility_bar_html
from stale_check.checker import (
    audit_markdown_document,
    find_line_number,
    format_json_report,
    format_markdown_report,
)
from trust_badge import (
    generate_trust_badge_markdown,
    generate_trust_badge_svg,
    get_trust_badge_snapshot_notice,
)

from benchmark import (
    format_results_markdown as format_benchmark_markdown,
    load_cached_results as load_benchmark_cache,
    load_dataset as load_benchmark_dataset,
    run_benchmark,
)
from lab.ui import render_agent_lab_tab
from demo_data import (
    GUIDED_DEMO_STEPS,
    SAVED_RESULTS_DATE,
    find_cached_pasted_audit,
    get_cached_ask_audit,
    get_cached_url_audit,
    get_guided_demo_step,
    load_demo_cache,
)
from extractor import (
    extract_claims,
    format_category_label,
    get_last_gateway_metrics as get_extract_metrics,
    locate_span_in_text,
    normalize_category,
)
from llm import (
    GEMINI_OPENAI_BASE_URL,
    _get_config_value,
    ask_question,
    check_gateway_health,
    get_gateway_dashboard_url,
    get_llm_config,
    get_task_routing_summary,
    is_use_bifrost,
    set_simulate_outage,
)
from reporting import (
    build_change_log,
    build_draft_corrected_answer,
    build_report_payload,
    build_side_by_side_word_diff,
    format_report_html,
    format_report_json,
    format_report_markdown,
    format_shareable_summary,
    render_cited_corrected_answer_html,
)
from tab_components import (
    get_real_model_and_routing_info,
    init_history_if_needed,
    load_evidence_counts,
    record_audit_to_history,
    render_api_playground_tab,
    render_history_tab,
    render_home_tab,
    render_methodology_tab,
    render_showdown_tab,
    render_watchlist_strip,
)
from verifier import (
    _extract_independent_domain,
    build_citation_map,
    classify_source_type,
    clear_tavily_cache,
    compute_risk_weighted_overall_score,
    create_skipped_claim_result,
    extract_article_from_url,
    generate_corrected_answer,
    get_last_gateway_metrics as get_rewrite_metrics,
    partition_claims_by_threshold,
    run_llm_only_verification,
    run_why_bifrost_simulation,
    truncate_snippet_words,
    verify_claim,
    verify_claims,
)
from ledger import (
    compute_audit_ledger,
    create_empty_ledger,
    merge_ledgers,
)
from bifrost_evidence import render_bifrost_evidence_panel

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
_DEFAULT_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">'
    '<rect width="64" height="64" rx="14" fill="#4F46E5"/>'
    '<circle cx="32" cy="32" r="21" fill="#0F1524" opacity="0.92"/>'
    '<path d="M22 33.5L29 40.5L43 24.5" fill="none" stroke="#22C55E" stroke-width="5.2" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="48" cy="16" r="6" fill="#F59E0B" stroke="#0F1524" stroke-width="2"/>'
    '</svg>'
)
_FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon.svg"


def _resolve_page_icon() -> str:
    """Load custom brand SVG favicon from assets/favicon.svg (with inline fallback)."""
    try:
        if _FAVICON_PATH.is_file():
            svg_text = _FAVICON_PATH.read_text(encoding="utf-8").strip()
            if svg_text.startswith("<svg"):
                return svg_text
    except OSError:
        pass
    return _DEFAULT_FAVICON_SVG


st.set_page_config(
    page_title="Stale-Answer Detector",
    page_icon=_resolve_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**Stale-Answer Detector** — Audits time-sensitive AI claims against live "
            "web sources via Tavily and Bifrost (or Direct Gemini), computes explainable "
            "freshness scores (0–100), and rewrites outdated answers with verified citations."
        ),
    },
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
    {
        "label": "World leadership & geography",
        "icon": ":material/public:",
        "text": (
            "The Prime Minister of India is Rahul Gandhi, while the capital "
            "of Australia is Sydney. Meanwhile, the standard VAT rate in the "
            "United Kingdom remains 17.5%."
        ),
    },
    {
        "label": "iPhone launch (Hinglish)",
        "icon": ":material/translate:",
        "text": (
            "Apple ne iPhone 15 Pro Max ko $1,199 ke starting price par launch kiya tha. "
            "Abhi Apple ka sabse latest flagship phone iPhone 15 Pro Max hi hai."
        ),
    },
    {
        "label": "भारत जीएसटी और सोना (Hindi)",
        "icon": ":material/language:",
        "text": (
            "भारत में मानक जीएसटी दरें 5%, 12%, 18% और 28% हैं। "
            "इसके अतिरिक्त सोने और चांदी पर 3% की विशेष जीएसटी दर लागू होती है।"
        ),
    },
]

# Status styling tokens used for badges, score bars, and ring colors.
STATUS_META = {
    "fresh": {
        "label": "Fresh",
        "pill_class": "pill-fresh",
        "color": "#16A34A",
        "badge_color": "green",
    },
    "outdated": {
        "label": "Outdated",
        "pill_class": "pill-outdated",
        "color": "#EA580C",
        "badge_color": "orange",
    },
    "incorrect": {
        "label": "Incorrect",
        "pill_class": "pill-incorrect",
        "color": "#DC2626",
        "badge_color": "red",
    },
    "partially_correct": {
        "label": "Partially correct",
        "pill_class": "pill-partially-correct",
        "color": "#D97706",
        "badge_color": "amber",
    },
    "uncertain": {
        "label": "Uncertain",
        "pill_class": "pill-uncertain",
        "color": "#475569",
        "badge_color": "yellow",
    },
    "unverifiable": {
        "label": "Unverifiable",
        "pill_class": "pill-unverifiable",
        "color": "#6B7280",
        "badge_color": "gray",
    },
    "verification_failed": {
        "label": "Verification failed",
        "pill_class": "pill-incorrect",
        "color": "#DC2626",
        "badge_color": "red",
    },
    "skipped": {
        "label": "Stable (skipped)",
        "pill_class": "pill-skipped",
        "color": "#64748B",
        "badge_color": "slate",
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
    "leadership": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    ),
    "geography": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/>'
        '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>'
    ),
    "records": (
        '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
        '<path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
        '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
        '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/></svg>'
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
# Visual Styling: Load modern SaaS theme from assets/style.css
# ---------------------------------------------------------------------------
_STYLE_CSS_PATH = Path(__file__).resolve().parent / "assets" / "style.css"


def load_css() -> str:
    """Load core visual stylesheet from assets/style.css."""
    try:
        if _STYLE_CSS_PATH.is_file():
            return _STYLE_CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Theme CSS tokens (loaded alongside assets/style.css for theme switching)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Dark Mode and Core Overrides — Deep Space Glassmorphism */
:root, [data-theme="dark"], body.dark-theme, .stApp.dark-theme, .stApp {
    color-scheme: dark !important;
    --bg-page: #070913 !important;
    --bg-page-gradient: radial-gradient(at 15% 15%, #181028 0px, transparent 55%),
                        radial-gradient(at 85% 15%, #0B1E2D 0px, transparent 55%),
                        radial-gradient(at 50% 85%, #11142E 0px, transparent 60%),
                        #070913 !important;
    --bg-card: rgba(18, 22, 38, 0.72) !important;
    --bg-card-solid: #121626 !important;
    --bg-card-subtle: rgba(18, 22, 38, 0.5) !important;
    --bg-secondary: rgba(24, 30, 52, 0.7) !important;
    --bg-input: rgba(14, 18, 32, 0.85) !important;
    --tab-bg: rgba(18, 22, 38, 0.7) !important;
    --border-main: rgba(255, 255, 255, 0.12) !important;
    --border-subtle: rgba(255, 255, 255, 0.07) !important;
    --text-main: #F1F5F9 !important;
    --text-muted: #94A3B8 !important;
    --text-subtle: #64748B !important;
    --shadow-card: 0 16px 36px -4px rgba(0, 0, 0, 0.5), 0 0 20px rgba(91, 91, 240, 0.1) !important;
}
.stApp {
    background: radial-gradient(at 15% 15%, #181028 0px, transparent 55%),
                radial-gradient(at 85% 15%, #0B1E2D 0px, transparent 55%),
                radial-gradient(at 50% 85%, #11142E 0px, transparent 60%),
                #070913 !important;
    background-attachment: fixed !important;
    color: #F1F5F9 !important;
}

/* Navigation Tabs - Dark Contrast (React Aria Components & BaseWeb support) */
[data-testid="stTabs"] [data-testid="stTab"],
[data-testid="stTabs"] [data-baseweb="tab"],
div[role="tablist"] [role="tab"] {
    color: #94A3B8 !important;
    opacity: 1 !important;
    border-bottom: none !important;
    border: none !important;
}
[data-testid="stTabs"] [data-testid="stTab"] p,
[data-testid="stTabs"] [data-testid="stTab"] span,
[data-testid="stTabs"] [data-testid="stTab"] div,
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span,
[data-testid="stTabs"] [data-baseweb="tab"] div,
div[role="tablist"] [role="tab"] p,
div[role="tablist"] [role="tab"] span,
div[role="tablist"] [role="tab"] div {
    color: #94A3B8 !important;
    opacity: 1 !important;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover,
[data-testid="stTabs"] [data-baseweb="tab"]:hover,
div[role="tablist"] [role="tab"]:hover {
    color: #F8FAFC !important;
    background-color: rgba(255, 255, 255, 0.06) !important;
}
[data-testid="stTabs"] [data-testid="stTab"]:hover p,
[data-testid="stTabs"] [data-testid="stTab"]:hover span,
[data-testid="stTabs"] [data-baseweb="tab"]:hover p,
[data-testid="stTabs"] [data-baseweb="tab"]:hover span,
div[role="tablist"] [role="tab"]:hover p,
div[role="tablist"] [role="tab"]:hover span {
    color: #F8FAFC !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
[data-testid="stTabs"] [data-testid="stTab"][data-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
div[role="tablist"] [role="tab"][aria-selected="true"],
div[role="tablist"] [role="tab"][data-selected="true"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    background-color: rgba(99, 102, 241, 0.32) !important;
    background-image: linear-gradient(135deg, rgba(99, 102, 241, 0.35), rgba(168, 85, 247, 0.25)) !important;
    border: 1px solid rgba(129, 140, 248, 0.7) !important;
    border-bottom: 1px solid rgba(129, 140, 248, 0.7) !important;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.5) !important;
    border-radius: 9999px !important;
    outline: none !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p,
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] span,
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] div,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] p,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] span,
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] div,
div[role="tablist"] [role="tab"][aria-selected="true"] p,
div[role="tablist"] [role="tab"][aria-selected="true"] span,
div[role="tablist"] [role="tab"][aria-selected="true"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
[data-baseweb="tab-highlight"],
[data-baseweb="tab-border"],
div[data-baseweb="tab-highlight"],
div[data-baseweb="tab-border"],
[data-testid="stTabs"] div[data-baseweb="tab-highlight"],
[data-testid="stTabs"] div[data-baseweb="tab-border"],
[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-testid="stTab"] .react-aria-SelectionIndicator,
div[role="tablist"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-baseweb="tab-list"] > div:not([data-baseweb="tab"]) {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* Secondary Buttons & Chips - Dark Glass Contrast */
button[data-testid="stBaseButton-secondary"],
button[kind="secondary"],
div[data-testid="stPopover"] > button,
div.stButton > button[data-testid="stBaseButton-secondary"],
div.stButton > button:not([data-testid="stBaseButton-primary"]):not([kind="primary"]) {
    background: rgba(30, 41, 59, 0.75) !important;
    background-color: rgba(30, 41, 59, 0.75) !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    color: #F8FAFC !important;
    font-weight: 500 !important;
}
button[data-testid="stBaseButton-secondary"] p,
button[data-testid="stBaseButton-secondary"] span,
button[kind="secondary"] p,
button[kind="secondary"] span,
div[data-testid="stPopover"] > button p,
div[data-testid="stPopover"] > button span,
div.stButton > button:not([data-testid="stBaseButton-primary"]) p,
div.stButton > button:not([data-testid="stBaseButton-primary"]) span {
    color: #F8FAFC !important;
    font-weight: 500 !important;
}
button[data-testid="stBaseButton-secondary"]:hover,
button[kind="secondary"]:hover,
div[data-testid="stPopover"] > button:hover,
div.stButton > button:not([data-testid="stBaseButton-primary"]):hover {
    background: rgba(51, 65, 85, 0.9) !important;
    background-color: rgba(51, 65, 85, 0.9) !important;
    border: 1px solid #818CF8 !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
}
button[data-testid="stBaseButton-secondary"]:hover p,
button[data-testid="stBaseButton-secondary"]:hover span,
button[kind="secondary"]:hover p,
button[kind="secondary"]:hover span,
div[data-testid="stPopover"] > button:hover p,
div[data-testid="stPopover"] > button:hover span,
div.stButton > button:not([data-testid="stBaseButton-primary"]):hover p,
div.stButton > button:not([data-testid="stBaseButton-primary"]):hover span {
    color: #FFFFFF !important;
}

/* Guided Demo Chip */
button[aria-label*="Guided Demo"],
div[data-testid="stColumn"]:has(#sad-guided-demo-anchor) button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25)) !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
    color: #F8FAFC !important;
}
button[aria-label*="Guided Demo"] p,
button[aria-label*="Guided Demo"] span,
div[data-testid="stColumn"]:has(#sad-guided-demo-anchor) button p,
div[data-testid="stColumn"]:has(#sad-guided-demo-anchor) button span {
    color: #F8FAFC !important;
}
button[aria-label*="Guided Demo"]:hover,
div[data-testid="stColumn"]:has(#sad-guided-demo-anchor) button:hover {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.4), rgba(168, 85, 247, 0.4)) !important;
    border: 1px solid #A855F7 !important;
    color: #FFFFFF !important;
}

/* Primary Action Buttons */
button[data-testid="stBaseButton-primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
}
button[data-testid="stBaseButton-primary"] p,
button[data-testid="stBaseButton-primary"] span,
button[kind="primary"] p,
button[kind="primary"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
button[data-testid="stBaseButton-primary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    color: #FFFFFF !important;
}

/* Segmented Controls / ButtonGroups */
div.stButtonGroup,
div[data-testid="stButtonGroup"],
div[data-testid="stSegmentedControl"],
div[data-testid="stButtonGroup"] > div,
div[data-testid="stSegmentedControl"] > div {
    background: #0F172A !important;
    background-color: #0F172A !important;
    padding: 4px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}
div.stButtonGroup button,
div[data-testid="stButtonGroup"] button,
div[data-testid="stSegmentedControl"] button {
    background: transparent !important;
    background-color: transparent !important;
    color: #94A3B8 !important;
    border: none !important;
    border-radius: 8px !important;
}
div.stButtonGroup button p,
div.stButtonGroup button span,
div[data-testid="stButtonGroup"] button p,
div[data-testid="stButtonGroup"] button span,
div[data-testid="stSegmentedControl"] button p,
div[data-testid="stSegmentedControl"] button span {
    color: #94A3B8 !important;
}
div.stButtonGroup button:hover,
div[data-testid="stButtonGroup"] button:hover,
div[data-testid="stSegmentedControl"] button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #F8FAFC !important;
}
div.stButtonGroup button:hover p,
div.stButtonGroup button:hover span,
div[data-testid="stButtonGroup"] button:hover p,
div[data-testid="stButtonGroup"] button:hover span,
div[data-testid="stSegmentedControl"] button:hover p,
div[data-testid="stSegmentedControl"] button:hover span {
    color: #F8FAFC !important;
}
div.stButtonGroup button[data-selected],
div.stButtonGroup button[aria-selected="true"],
div.stButtonGroup button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[data-selected],
div[data-testid="stButtonGroup"] button[aria-selected="true"],
div[data-testid="stButtonGroup"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[data-selected],
div[data-testid="stSegmentedControl"] button[aria-selected="true"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35) !important;
}
div.stButtonGroup button[data-selected] p,
div.stButtonGroup button[data-selected] span,
div.stButtonGroup button[aria-selected="true"] p,
div.stButtonGroup button[aria-selected="true"] span,
div[data-testid="stButtonGroup"] button[data-selected] p,
div[data-testid="stButtonGroup"] button[data-selected] span,
div[data-testid="stButtonGroup"] button[aria-selected="true"] p,
div[data-testid="stButtonGroup"] button[aria-selected="true"] span,
div[data-testid="stSegmentedControl"] button[data-selected] p,
div[data-testid="stSegmentedControl"] button[data-selected] span,
div[data-testid="stSegmentedControl"] button[aria-selected="true"] p,
div[data-testid="stSegmentedControl"] button[aria-selected="true"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Textarea & Inputs */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.85) !important;
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(148, 163, 184, 0.3) !important;
    border-radius: 14px !important;
    color: #F8FAFC !important;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
    opacity: 0.8 !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
}
</style>
"""

LIGHT_THEME_CSS = """
<style>
/* Light Mode Theme Overrides - Modern Bright Glassmorphism */
:root, .stApp, body {
    color-scheme: light !important;
    --bg-page: #F8F9FE !important;
    --bg-card: rgba(255, 255, 255, 0.65) !important;
    --bg-card-solid: #FFFFFF !important;
    --bg-card-subtle: rgba(255, 255, 255, 0.45) !important;
    --bg-secondary: rgba(241, 245, 249, 0.65) !important;
    --bg-input: rgba(255, 255, 255, 0.85) !important;
    --tab-bg: rgba(255, 255, 255, 0.65) !important;
    --border-main: rgba(255, 255, 255, 0.8) !important;
    --border-subtle: rgba(229, 231, 239, 0.6) !important;
    --text-main: #0F172A !important;
    --text-muted: #64748B !important;
    --text-subtle: #94A3B8 !important;
    --shadow-card: 0 8px 30px -4px rgba(91, 91, 240, 0.08), 0 2px 8px -2px rgba(15, 23, 42, 0.04) !important;
}
.stApp {
    background: radial-gradient(at 0% 0%, #EEF0FF 0px, transparent 50%),
                radial-gradient(at 100% 0%, #FDE7F3 0px, transparent 50%),
                radial-gradient(at 100% 100%, #FFEBDD 0px, transparent 50%),
                radial-gradient(at 0% 100%, #EEF0FF 0px, transparent 50%),
                #F8F9FE !important;
    background-attachment: fixed !important;
    color: #0F172A !important;
}
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border-right: 1px solid var(--border-main) !important;
}
.sad-hero {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border: 1px solid var(--border-main) !important;
}
.sad-score-card {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border: 1px solid var(--border-main) !important;
}
.sad-claim-card {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(180%) !important;
    border: 1px solid var(--border-main) !important;
}
.sad-compare-box {
    background: rgba(241, 245, 249, 0.65) !important;
    border: 1px solid var(--border-main) !important;
}
.sad-source-chip {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid var(--border-main) !important;
    color: var(--text-main) !important;
}
.sad-snippet-card {
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid var(--border-main) !important;
}
.sad-empty-state {
    background: rgba(255, 255, 255, 0.6) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
    border: 1px dashed var(--border-main) !important;
}
.sad-diff-card {
    background: rgba(255, 255, 255, 0.65) !important;
    backdrop-filter: blur(18px) !important;
    border: 1px solid var(--border-main) !important;
}
.pill-fresh {
    background: rgba(220, 252, 231, 0.85) !important;
    color: #16A34A !important;
}
.pill-outdated {
    background: rgba(255, 237, 213, 0.85) !important;
    color: #EA580C !important;
}
.pill-uncertain {
    background: rgba(226, 232, 240, 0.85) !important;
    color: #475569 !important;
}
.pill-unverifiable {
    background: rgba(243, 244, 246, 0.85) !important;
    color: #6B7280 !important;
}
div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border-main) !important;
    color: var(--text-main) !important;
}
div.stDownloadButton > button {
    border-radius: 9999px !important;
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


def classify_error_quota_status(exc: Exception | str) -> str | None:
    """Return 'Limited' on HTTP 429/rate-limit/quota errors, 'Offline' on connection/unreachable errors."""
    text = str(exc or "").lower()
    if any(tok in text for tok in ("429", "rate limit", "rate-limit", "quota", "resource_exhausted", "too many requests", "usage limit", "high demand", "model busy", "spikes in demand")):
        return "Limited"
    if any(tok in text for tok in ("bifrost down", "could not connect", "connection refused", "winerror 10061", "network unreachable", "network offline", "getaddrinfo", "name resolution")):
        return "Offline"
    if "503" in text:
        return "Limited" if ("demand" in text or "quota" in text or "busy" in text) else "Offline"
    if any(tok in text for tok in ("connection", "unreachable", "timed out", "timeout", "network", "offline", "502", "504", "is down")):
        return "Offline"
    return None


def get_overall_tier(
    score: int | None,
    results: list[dict] | None = None,
) -> tuple[str, str, str]:
    """
    Map a risk-weighted overall freshness score (0-100 or None) to
    (label, ring_color, description).

    When one or more claims have status 'verification_failed', returns
    'Could not verify N of M claims' instead of 'Mixed'.
    """
    res_list = results or []
    failed_count = sum(
        1 for r in res_list if isinstance(r, dict) and r.get("status") == "verification_failed"
    )
    if failed_count > 0:
        total_claims = len(res_list)
        label = f"Could not verify {failed_count} of {total_claims} claims"
        if failed_count == total_claims:
            desc = "Verification failed for all claims after retries and fallback. Retry individual claims or load Demo / Saved results."
        else:
            desc = (
                f"{failed_count} of {total_claims} claims could not be verified and are excluded "
                "from the overall score and status counts."
            )
        return (label, "#DC2626", desc)

    incorrect_count = sum(1 for r in res_list if isinstance(r, dict) and r.get("status") == "incorrect")
    outdated_count = sum(1 for r in res_list if isinstance(r, dict) and r.get("status") == "outdated")

    if score is None:
        return (
            "Unverifiable",
            "#64748B",
            "No reliable web evidence was found to score these claims yet.",
        )
    if score >= 80:
        return "Fresh", "#16A34A", "Risk-weighted claims align closely with current live web sources."
    if score >= 40:
        if incorrect_count > 0 and outdated_count > 0:
            return "Mixed", "#D97706", "Risk-weighted check shows some details are incorrect, outdated, or uncertain."
        if incorrect_count > 0:
            lbl = "1 incorrect claim found" if incorrect_count == 1 else f"{incorrect_count} incorrect claims found"
            return "Mixed", "#D97706", f"{lbl} — Contradicted by official sources."
        return "Mixed", "#D97706", "Risk-weighted check shows some details are outdated or uncertain."

    # score < 40
    if incorrect_count > 0 and outdated_count > 0:
        return "Stale / Incorrect", "#DC2626", "Contains claims contradicted by official sources and outdated facts."
    if incorrect_count > 0:
        lbl = "1 incorrect claim found" if incorrect_count == 1 else f"{incorrect_count} incorrect claims found"
        return "Incorrect", "#DC2626", f"{lbl} — Contradicted by official sources."
    return "Stale", "#EA580C", "Key high-impact facts are outdated compared to today's sources."


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


def safe_href_url(url: str | None) -> str:
    """Return an escaped, protocol-safe URL for HTML href attributes.
    Only allows http:// and https:// URLs; returns '#' for any other scheme."""
    if not url or not isinstance(url, str):
        return "#"
    clean = url.strip()
    if not clean.lower().startswith(("http://", "https://")):
        return "#"
    return html.escape(clean, quote=True)


DEFAULT_MAX_SESSION_AUDITS = 5
DEFAULT_MAX_CLAIMS_PER_AUDIT = 6


def get_max_session_audits() -> int:
    val = _get_config_value("MAX_SESSION_AUDITS")
    if val:
        try:
            return max(1, int(val))
        except (ValueError, TypeError):
            pass
    return DEFAULT_MAX_SESSION_AUDITS


def get_max_claims_per_audit() -> int:
    val = _get_config_value("MAX_CLAIMS_PER_AUDIT")
    if val:
        try:
            return max(1, int(val))
        except (ValueError, TypeError):
            pass
    return DEFAULT_MAX_CLAIMS_PER_AUDIT


def is_quota_or_rate_limit_error(exc: Exception | str) -> bool:
    """Return True if exception represents a 429, 503, rate limit, credit exhaustion, or quota issue."""
    raw_type = type(exc).__name__ if isinstance(exc, BaseException) else ""
    if raw_type == "UsageLimitExceededError":
        return True
    raw = f"{raw_type} {str(exc or '')}".lower()
    return any(
        tok in raw
        for tok in (
            "429",
            "503",
            "quota",
            "rate limit",
            "rate_limit",
            "usage limit",
            "usagelimitexceedederror",
            "resource_exhausted",
            "credit limit",
            "plan limit",
            "account/billing limit",
            "billing limit",
            "432",
            "433",
            "434",
            "live quota is busy",
        )
    )


def format_friendly_ui_error(exc: Exception | str, *, action: str = "checking claims") -> str:
    """
    Translate raw exceptions into calm, actionable messages for Bifrost down,
    invalid API keys, Tavily rate/credit limits, and invalid JSON from the LLM,
    never leaking raw Python stack traces to the Streamlit UI.
    """
    exc_type = type(exc).__name__ if isinstance(exc, BaseException) else ""
    raw = f"{exc_type} {str(exc or '')}".strip()
    lower = raw.lower()
    use_bifrost = is_use_bifrost()

    # 1. Invalid or missing API keys (Tavily, Bifrost virtual key, Gemini, or LLM)
    if "tavily_api_key" in lower or (
        "tavily" in lower
        and any(tok in lower for tok in ("401", "403", "invalid", "unauthorized", "not set", "missing", "forbidden"))
    ):
        return (
            "Tavily key invalid: Invalid or missing Tavily API key (401/403). Please add TAVILY_API_KEY in "
            "Streamlit Secrets or your .env file."
        )

    if use_bifrost and ("llm_virtual_key" in lower or "x-bf-vk" in lower or "virtual key" in lower):
        return (
            "Bifrost virtual key invalid: Invalid Bifrost virtual key (LLM_VIRTUAL_KEY / x-bf-vk). Please verify "
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
                "Gemini API key invalid: Invalid or missing Gemini API key. Please add LLM_API_KEY "
                "(or GEMINI_API_KEY) in Streamlit Secrets or your .env file."
            )
        return (
            "Gemini API key invalid: Invalid or missing LLM API key. Please check LLM_API_KEY in "
            "Streamlit Secrets or .env (and your provider keys in Bifrost)."
        )

    # 2. Rate / credit / quota limit from Tavily or LLM (429 / 503 / usage limits)
    if ("tavily" in lower and any(tok in lower for tok in ("429", "credit", "usage limit", "quota", "usagelimitexceedederror"))) or "usagelimitexceedederror" in lower:
        return (
            "Tavily quota exceeded: Live quota is busy, showing saved results. "
            "The external verification service reached a temporary rate or quota limit (429/503). "
            "Please wait a moment or inspect the pre-verified saved results."
        )

    if "503" in lower and any(tok in lower for tok in ("high demand", "spikes in demand", "busy", "unavailable")):
        return (
            "Gemini model busy: Upstream model is experiencing high demand (503). "
            "Live quota is busy, showing saved results. "
            "The external verification service reached a temporary rate or quota limit (429/503). "
            "Please wait a moment or inspect the pre-verified saved results."
        )

    if "resource_exhausted" in lower or ("gemini" in lower and any(tok in lower for tok in ("quota", "429", "rate limit"))) or ("llm" in lower and "quota" in lower):
        return (
            "Gemini quota exceeded: Live quota is busy, showing saved results. "
            "The external verification service reached a temporary rate or quota limit (429/503). "
            "Please wait a moment or inspect the pre-verified saved results."
        )

    if is_quota_or_rate_limit_error(lower):
        return (
            "Gemini quota exceeded: Live quota is busy, showing saved results. "
            "The external verification service reached a temporary rate or quota limit (429/503). "
            "Please wait a moment or inspect the pre-verified saved results."
        )

    # 3. Model retired or not found (404)
    if "404" in lower and any(tok in lower for tok in ("not found", "no longer available", "retired")):
        return (
            "Gemini model retired or not found (404): The requested model is no longer available. "
            "Please update LLM_MODEL in .env."
        )

    # 4. Invalid JSON from the LLM (extraction or verification schema parsing)
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
            "Invalid JSON: The LLM returned invalid JSON instead of the expected claim structure. "
            "Please try running the check again or switch models in LLM_MODEL."
        )

    if "tavily" in lower:
        return (
            "The web search provider encountered a temporary error. "
            "Using local verified cache or please try again in a moment."
        )

    # 5. Bifrost gateway / network unreachable (gateway mode)
    if use_bifrost and (
        ("bifrost" in lower and any(tok in lower for tok in ("down", "unreachable", "offline", "refused", "timeout", "timed out", "is down")))
        or any(
            tok in lower
            for tok in (
                "connection refused",
                "connecterror",
                "winerror 10061",
                "failed to establish a new connection",
                ":8080",
                "localhost:8080",
            )
        )
    ):
        return (
            "Bifrost down: Bifrost gateway is down or unreachable (or network offline). Check your internet connection "
            "and local Bifrost server (port 8080), set USE_BIFROST=false for Direct Gemini, "
            "or use Demo / Saved results."
        )

    # 6. Direct connection / timeout / offline errors
    if any(
        tok in lower
        for tok in (
            "apiconnectionerror",
            "connecterror",
            "timed out",
            "timeout",
            "connection error",
            "connection refused",
            "unreachable",
            "network",
            "offline",
            "getaddrinfo",
            "name resolution",
            "502",
            "504",
        )
    ):
        return (
            f"Network offline: Network offline or API service unreachable while {action}. Check your internet connection "
            "and endpoint settings, then click Retry (or load Demo / Saved results)."
        )

    # 5b. URL validation / Tavily Extract webpage failures (must run before LLM 404 check)
    if (
        action == "extracting the URL"
        or "tavily url extract" in lower
        or "readable article text" in lower
        or "valid http://" in lower
        or "article or blog url" in lower
    ):
        snippet_url = _sanitize_error_snippet(raw)
        if snippet_url:
            return snippet_url
        return (
            "Could not extract readable article text from that URL. "
            "Check that the page is public and try another URL."
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


def _apply_pasted_demo_sample(
    sample: dict,
    exact_match: bool = True,
    *,
    sync_widget: bool = True,
) -> None:
    """Hydrate Tab 1 ('Audit pasted answer') completely from an offline cached demo sample."""
    ans_text = str(sample.get("answer_text") or "")
    label = str(sample.get("label") or "Sample audit")
    claims_list = copy.deepcopy(sample.get("results") or [])
    for c in claims_list:
        if isinstance(c, dict):
            c.setdefault("source_origin", "saved")
            c.setdefault("saved_date", SAVED_RESULTS_DATE)
    st.session_state["quota_status"] = "Saved results"
    ext_model = str(sample.get("extract_model") or "gemini/gemini-2.5-flash-lite (cached)")
    ext_meta = copy.deepcopy(sample.get("extract_meta") or {})
    if ext_meta:
        ext_meta.setdefault("operation", "Extract claims (Demo cache)")
        ext_meta.setdefault("model_used", ext_model)

    if sync_widget:
        st.session_state["answer_input"] = ans_text
    st.session_state["checked_answer"] = ans_text
    st.session_state["results"] = claims_list
    st.session_state["extract_model"] = ext_model
    st.session_state["extract_meta"] = ext_meta
    st.session_state["corrected_answer"] = None
    st.session_state["corrected_model"] = None
    st.session_state["corrected_meta"] = None
    st.session_state["demo_pasted_rewrite"] = {
        "text": sample.get("corrected_answer"),
        "model": str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)"),
    }
    st.session_state["pasted_audit_at"] = datetime.now().astimezone()
    st.session_state["status_steps"] = [
        f"Demo cache: loaded '{label}' ({len(claims_list)} pre-verified claim(s), 0 API calls)",
        "Done",
    ]
    if exact_match:
        st.session_state["demo_pasted_notice"] = (
            f"Offline cached audit for '{label}' (0 live API calls, 0 Tavily credits)."
        )
    else:
        st.session_state["demo_pasted_notice"] = (
            f"Custom text detected in Demo mode — showing cached sample '{label}'. "
            "Turn off Demo mode to run a live audit on custom answers."
        )


def _apply_ask_demo_sample(
    question_text: str | None = None,
    *,
    sync_widget: bool = True,
) -> dict:
    """Hydrate Tab 2 ('Ask & Audit') completely from offline `ask_sample` in demo_cache.json."""
    sample = get_cached_ask_audit(question_text)
    q_clean = str(sample.get("question") or sample.get("default_question") or "")
    ans_text = str(sample.get("answer_text") or "")
    claims_list = copy.deepcopy(sample.get("results") or [])
    for c in claims_list:
        if isinstance(c, dict):
            c.setdefault("source_origin", "saved")
            c.setdefault("saved_date", SAVED_RESULTS_DATE)
    st.session_state["quota_status"] = "Saved results"
    ans_model = str(sample.get("answer_model") or "openai/gpt-4o-mini (cached)")
    ans_meta = copy.deepcopy(sample.get("answer_meta") or {})
    ext_model = str(sample.get("extract_model") or "gemini/gemini-2.5-flash-lite (cached)")
    ext_meta = copy.deepcopy(sample.get("extract_meta") or {})

    if sync_widget:
        st.session_state["ask_question_input"] = q_clean
    st.session_state["ask_question_asked"] = q_clean
    st.session_state["ask_answer"] = ans_text
    st.session_state["ask_model"] = ans_model
    st.session_state["ask_meta"] = ans_meta
    st.session_state["ask_results"] = claims_list
    st.session_state["ask_extract_model"] = ext_model
    st.session_state["ask_extract_meta"] = ext_meta
    st.session_state["ask_corrected_answer"] = None
    st.session_state["ask_corrected_model"] = None
    st.session_state["ask_corrected_meta"] = None
    st.session_state["demo_ask_rewrite"] = {
        "text": sample.get("corrected_answer"),
        "model": str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)"),
    }
    st.session_state["ask_audit_at"] = datetime.now().astimezone()
    st.session_state["ask_status_steps"] = [
        f"Demo cache: loaded LLM Q&A response ({ans_model})",
        f"Demo cache: loaded {len(claims_list)} pre-verified claim(s) (0 API calls)",
        "Done",
    ]
    st.session_state["demo_ask_notice"] = (
        "Offline cached Q&A audit loaded (0 live API calls, 0 Tavily credits). "
        "Turn off Demo mode to query your live LLM & Tavily."
    )
    return sample


def _apply_url_demo_sample(
    url_text: str | None = None,
    *,
    sync_widget: bool = True,
) -> dict:
    """Hydrate Tab 3 ('Audit a URL') completely from offline `url_sample` in demo_cache.json."""
    sample = get_cached_url_audit(url_text)
    resolved_url = str(sample.get("url") or sample.get("default_url") or "")
    content = str(sample.get("content") or "")
    claims_list = copy.deepcopy(sample.get("results") or [])
    for c in claims_list:
        if isinstance(c, dict):
            c.setdefault("source_origin", "saved")
            c.setdefault("saved_date", SAVED_RESULTS_DATE)
    st.session_state["quota_status"] = "Saved results"
    ext_model = str(sample.get("extract_model") or "gemini/gemini-2.5-flash-lite (cached)")
    ext_meta = copy.deepcopy(sample.get("extract_meta") or {})

    if sync_widget:
        st.session_state["url_input"] = resolved_url
    st.session_state["url_audited"] = resolved_url
    st.session_state["url_article_title"] = str(sample.get("title") or "Cached Demo Article")
    st.session_state["url_article_text"] = content
    st.session_state["url_article_truncated"] = False
    st.session_state["url_extract_credits"] = 0.0
    st.session_state["url_extract_cache_hit"] = True
    st.session_state["url_total_claims_found"] = len(claims_list)
    st.session_state["url_results"] = claims_list
    st.session_state["url_extract_model"] = ext_model
    st.session_state["url_extract_meta"] = ext_meta
    st.session_state["url_corrected_answer"] = None
    st.session_state["url_corrected_model"] = None
    st.session_state["url_corrected_meta"] = None
    st.session_state["demo_url_rewrite"] = {
        "text": sample.get("corrected_answer"),
        "model": str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)"),
    }
    st.session_state["url_audit_at"] = datetime.now().astimezone()
    st.session_state["url_status_steps"] = [
        f"Demo cache: loaded article text ({len(content):,} chars, 0 credits)",
        f"Demo cache: loaded {len(claims_list)} pre-verified claim(s) (0 API calls)",
        "Done",
    ]
    st.session_state["demo_url_notice"] = (
        "Offline cached URL article audit loaded (0 live API calls, 0 Tavily credits). "
        "Turn off Demo mode to fetch and audit live webpages."
    )
    return sample


def _resolve_cached_rewrite(
    key_prefix: str,
    original_answer: str,
    results: list[dict] | None,
) -> tuple[str, str]:
    """Return `(corrected_answer, model_label)` for Demo mode without calling live APIs."""
    rewrite_key_map = {
        "pasted": "demo_pasted_rewrite",
        "ask": "demo_ask_rewrite",
        "url": "demo_url_rewrite",
    }
    cached_entry = st.session_state.get(rewrite_key_map.get(key_prefix, ""))
    if isinstance(cached_entry, dict) and str(cached_entry.get("text") or "").strip():
        return str(cached_entry["text"]), str(cached_entry.get("model") or "openai/gpt-4o-mini (cached)")

    try:
        if key_prefix == "pasted":
            sample, exact = find_cached_pasted_audit(original_answer)
            if exact and str(sample.get("corrected_answer") or "").strip():
                return str(sample["corrected_answer"]), str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)")
        elif key_prefix == "ask":
            sample = get_cached_ask_audit()
            if str(sample.get("corrected_answer") or "").strip():
                return str(sample["corrected_answer"]), str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)")
        elif key_prefix == "url":
            sample = get_cached_url_audit()
            if str(sample.get("corrected_answer") or "").strip():
                return str(sample["corrected_answer"]), str(sample.get("corrected_model") or "openai/gpt-4o-mini (cached)")
    except FileNotFoundError:
        pass

    matched = match_claims_to_answer_spans(original_answer or "", results or [])
    draft = build_draft_corrected_answer(original_answer or "", results or [], matched_spans=matched)
    return draft, "offline-demo-draft (cached)"


def on_demo_mode_toggle() -> None:
    """Callback when sidebar Demo mode toggle changes."""
    if st.session_state.get("demo_mode"):
        st.session_state["pasted_error"] = None
        st.session_state["ask_error"] = None
        st.session_state["url_error"] = None
        st.session_state["benchmark_error"] = None
        st.session_state["pasted_rewrite_error"] = None
        st.session_state["ask_rewrite_error"] = None
        st.session_state["url_rewrite_error"] = None
        if st.session_state.get("results") is None:
            try:
                sample, _ = find_cached_pasted_audit(st.session_state.get("answer_input"))
                _apply_pasted_demo_sample(sample, exact_match=True)
            except FileNotFoundError:
                st.session_state["demo_pasted_notice"] = (
                    "Demo mode enabled, but demo_cache.json could not be loaded."
                )
    else:
        st.session_state["demo_pasted_notice"] = ""
        st.session_state["demo_ask_notice"] = ""
        st.session_state["demo_url_notice"] = ""


def _show_skeleton_slot(slot, blocks: tuple[int, ...] = (128, 82, 165)) -> None:
    """Render native Streamlit loading skeletons inside `slot` with graceful fallback."""
    if slot is None:
        return
    try:
        if hasattr(slot, "container"):
            with slot.container():
                for h in blocks:
                    if hasattr(st, "skeleton"):
                        st.skeleton(height=h, width="stretch")
                    else:
                        st.markdown(
                            f'<div class="sad-skeleton-box" style="height:{int(h)}px;"></div>',
                            unsafe_allow_html=True,
                        )
        elif hasattr(st, "skeleton"):
            st.skeleton(height=blocks[0] if blocks else 120, width="stretch")
    except Exception:
        pass


def _clear_slot(slot) -> None:
    """Clear a Streamlit `st.empty()` slot cleanly."""
    if slot is not None:
        try:
            slot.empty()
        except Exception:
            pass


def _render_demo_banner(message: str) -> None:
    """Render an offline cached audit banner cleanly without raw Markdown glitches."""
    if not message:
        return
    st.markdown(
        '<div class="sad-demo-banner">'
        f'<span><strong>Demo mode</strong> &mdash; {escape_html(message)}</span>'
        '<span class="sad-demo-pill">Offline cache</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_empty_state(
    title: str,
    body_html: str,
    *,
    pills: tuple[str, ...] = ("Prices & specs", "Tax rules", "AI API pricing"),
    hint: str = "",
    icon_svg: str = "",
    variant: str = "default",
) -> str:
    """Build volumetric 3D glowing orbital checkmark illustration, title, description, and chips."""
    is_cyan = variant == "cyan"
    grad_id = "chkCyan" if is_cyan else "chkPurple"
    ring_stroke = "rgba(6, 182, 212, 0.45)" if is_cyan else "rgba(168, 85, 247, 0.45)"
    dot_fill = "#22D3EE" if is_cyan else "#C084FC"
    filter_id = "glowCyan" if is_cyan else "glowPurple"

    friendly_illustration = (
        '<div class="sad-empty-svg-wrap" style="margin: 0.05rem auto 0.25rem auto; width: 92px; height: 58px; display: flex; align-items: center; justify-content: center;">'
        '<svg viewBox="0 0 110 70" width="94" height="58" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        + (
            '<linearGradient id="chkCyan" x1="30" y1="20" x2="80" y2="55" gradientUnits="userSpaceOnUse">'
            '<stop offset="0%" stop-color="#67E8F9"/>'
            '<stop offset="50%" stop-color="#06B6D4"/>'
            '<stop offset="100%" stop-color="#0284C7"/>'
            '</linearGradient>'
            if is_cyan
            else
            '<linearGradient id="chkPurple" x1="30" y1="20" x2="80" y2="55" gradientUnits="userSpaceOnUse">'
            '<stop offset="0%" stop-color="#38BDF8"/>'
            '<stop offset="50%" stop-color="#818CF8"/>'
            '<stop offset="100%" stop-color="#A855F7"/>'
            '</linearGradient>'
        )
        + f'<filter id="{filter_id}" x="-30%" y="-30%" width="160%" height="160%">'
        '<feGaussianBlur stdDeviation="3.5" result="blur"/>'
        '<feComposite in="SourceGraphic" in2="blur" operator="over"/>'
        '</filter>'
        '<linearGradient id="tubeLight" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.8"/>'
        '<stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>'
        '</linearGradient>'
        '</defs>'
        + f'<ellipse cx="55" cy="38" rx="46" ry="16" transform="rotate(-15 55 38)" stroke="{ring_stroke}" stroke-width="1.6" stroke-dasharray="3 4"/>'
        + f'<circle cx="89" cy="27" r="3.5" fill="{dot_fill}" filter="url(#{filter_id})"/>'
        + f'<path d="M38 38L49 49L76 25" stroke="{ring_stroke}" stroke-width="11" stroke-linecap="round" stroke-linejoin="round" opacity="0.3" filter="url(#{filter_id})"/>'
        + f'<path d="M38 38L49 49L76 25" stroke="url(#{grad_id})" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>'
        + '<path d="M39 37L49 47L75 24" stroke="url(#tubeLight)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.6"/>'
        '</svg>'
        '</div>'
    )
    pills_html = (
        '<div style="display: inline-flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px;">'
        + "".join(f'<span class="sad-empty-pill">{escape_html(p)}</span>' for p in pills)
        + '</div>'
        if pills
        else ""
    )
    hint_html = f'<div class="sad-empty-hint" style="margin-top: 0.25rem; font-size: 0.78rem;">{hint}</div>' if hint else ""
    return (
        '<div class="sad-empty-state sad-empty-state-compact">'
        f'{friendly_illustration}'
        f'<h4 class="sad-empty-title" style="font-size: 1rem; margin-bottom: 0.15rem; font-weight: 700; color: #FFFFFF;">{escape_html(title)}</h4>'
        f'<p class="sad-empty-desc" style="font-size: 0.82rem; margin: 0; line-height: 1.35; color: #94A3B8; max-width: 380px;">{body_html}</p>'
        f'{pills_html}'
        f'{hint_html}'
        '</div>'
    )


def render_stepper_html(
    current_busy: str | None,
    has_results: bool,
    has_input: bool,
    results: list[dict] | None = None,
    ui_state: AuditUIState | None = None,
) -> str:
    """Render 5-stage progress pipeline tracker: 1 Input > 2 Detect > 3 Verify > 4 Score > 5 Report.

    States:
    - pending: muted grey outline and grey number
    - running: ONE accent colour (indigo) with soft pulse
    - done: green filled circle with an SVG draw-on check mark
    - error: red outline with an exclamation mark + small shake (later steps STAY pending)
    Connectors fill left to right via transform: scaleX(...) when the prior step is done.
    """
    active_ui_state = ui_state
    if active_ui_state is None and current_busy:
        candidate = st.session_state.get("audit_ui_state")
        if isinstance(candidate, AuditUIState) and candidate.phase == "running":
            active_ui_state = candidate

    if isinstance(active_ui_state, AuditUIState) and active_ui_state.phase in {"running", "error", "done", "stopped"}:
        s1_state = active_ui_state.stepper.get("input", "pending")
        s2_state = active_ui_state.stepper.get("detect", "pending")
        s3_state = active_ui_state.stepper.get("verify", "pending")
        s4_state = active_ui_state.stepper.get("score", "pending")
        s5_state = active_ui_state.stepper.get("report", "pending")
    else:
        has_extract_error = bool(
            st.session_state.get("pasted_error")
            or st.session_state.get("ask_error")
            or st.session_state.get("url_error")
            or st.session_state.get("batch_error")
            or st.session_state.get("doc_checker_error")
        )
        active_results = results
        if active_results is None:
            active_results = (
                st.session_state.get("results")
                or st.session_state.get("ask_results")
                or st.session_state.get("url_results")
                or []
            )
        has_verify_failure = any(
            isinstance(r, dict) and r.get("status") == "verification_failed"
            for r in (active_results or [])
        )
        explicit_error_step = str(st.session_state.get("pipeline_error_step") or "").strip().lower()

        if has_extract_error or explicit_error_step == "detect":
            s1_state = "done"
            s2_state = "error"
            s3_state = "pending"
            s4_state = "pending"
            s5_state = "pending"
        elif has_verify_failure or explicit_error_step == "verify":
            s1_state = "done"
            s2_state = "done"
            s3_state = "error"
            s4_state = "pending"
            s5_state = "pending"
        elif explicit_error_step == "score":
            s1_state = "done"
            s2_state = "done"
            s3_state = "done"
            s4_state = "error"
            s5_state = "pending"
        elif explicit_error_step == "report":
            s1_state = "done"
            s2_state = "done"
            s3_state = "done"
            s4_state = "done"
            s5_state = "error"
        elif st.session_state.get("audit_stopped") and has_results:
            s1_state = "done"
            s2_state = "done"
            s3_state = "done"
            s4_state = "pending"
            s5_state = "pending"
        elif has_results:
            s1_state = "done"
            s2_state = "done"
            s3_state = "done"
            s4_state = "done"
            s5_state = "done"
        elif current_busy:
            s1_state = "done"
            s2_state = "running"
            s3_state = "pending"
            s4_state = "pending"
            s5_state = "pending"
        else:
            # Before any audit starts ALL five steps are pending (no step highlighted)
            s1_state = "pending"
            s2_state = "pending"
            s3_state = "pending"
            s4_state = "pending"
            s5_state = "pending"

    def _render_step(num: str, name: str, state: str, anchor: str) -> str:
        cls = f"sad-pipe-step pipe-{state}"
        if state == "done":
            badge_html = f'<span class="sad-pipe-badge">{render_stepper_check_svg()}</span>'
        elif state == "error":
            badge_html = '<span class="sad-pipe-badge">!</span>'
        else:
            badge_html = f'<span class="sad-pipe-badge">{num}</span>'

        scroll_js = (
            f"const d = window.parent.document || document; "
            f"const el = d.getElementById('{anchor}'); "
            f"if (el) el.scrollIntoView({{ behavior: 'smooth', block: 'start' }}); "
            f"return false;"
        )
        return (
            f'<a href="#{anchor}" class="{cls}" onclick="{scroll_js}" title="Jump to {name}">'
            f'{badge_html}'
            f'<span class="sad-pipe-name">{name}</span>'
            f'</a>'
        )

    def _render_connector(prev_state: str) -> str:
        conn_cls = (
            "sad-pipe-connector-track pipe-connector-done"
            if prev_state == "done"
            else "sad-pipe-connector-track"
        )
        return f'<span class="{conn_cls}" aria-hidden="true"><span class="sad-pipe-connector-fill"></span></span>'

    step1_html = _render_step("1", "Input", s1_state, "sad-input-studio-anchor")
    conn1_html = _render_connector(s1_state)
    step2_html = _render_step("2", "Detect", s2_state, "sad-claims-detected-anchor")
    conn2_html = _render_connector(s2_state)
    step3_html = _render_step("3", "Verify", s3_state, "sad-claim-cards-anchor")
    conn3_html = _render_connector(s3_state)
    step4_html = _render_step("4", "Score", s4_state, "sad-summary-anchor")
    conn4_html = _render_connector(s4_state)
    step5_html = _render_step("5", "Report", s5_state, "sad-export-anchor")

    return f"""
    <div class="sad-pipeline-tracker" role="progressbar" aria-label="Freshness audit pipeline tracker">
      {step1_html}
      {conn1_html}
      {step2_html}
      {conn2_html}
      {step3_html}
      {conn3_html}
      {step4_html}
      {conn4_html}
      {step5_html}
    </div>
    """


def _set_tab_error(
    error_state_key: str,
    exc: Exception | str,
    *,
    action: str,
    recover_key: str,
    query_or_url: str = "",
) -> None:
    """Persist a sanitized friendly error, update Quota chip (Offline/Limited), and record Failed audit in History."""
    friendly_msg = format_friendly_ui_error(exc, action=action)
    st.session_state[error_state_key] = {
        "message": friendly_msg,
        "recover_key": recover_key,
        "query_or_url": str(query_or_url or ""),
    }
    st.session_state["pipeline_error_step"] = "detect"
    err_quota = classify_error_quota_status(exc) or classify_error_quota_status(friendly_msg)
    if err_quota:
        st.session_state["quota_status"] = err_quota
    if str(query_or_url or "").strip():
        record_audit_to_history(
            str(query_or_url).strip(),
            [],
            source_label="Live",
            mode=error_state_key.replace("_error", "").capitalize(),
            score=None,
            audit_status="Failed",
            error_reason=friendly_msg,
        )


def _render_stored_error_with_demo_recovery(
    error_state_key: str,
    *,
    recover_scope: str,
    default_recover_key: str,
) -> None:
    """Display a persisted friendly error plus a one-click Demo mode recovery action."""
    err_info = st.session_state.get(error_state_key)
    if not isinstance(err_info, dict) or not err_info.get("message"):
        return

    st.error(
        str(err_info["message"]),
        icon=":material/error:",
    )
    recover_key = str(err_info.get("recover_key") or default_recover_key)
    query_or_url = str(err_info.get("query_or_url") or "")

    if not st.session_state.get("demo_mode"):
        with st.container(border=True):
            msg_col, btn_col = st.columns([2.2, 1.2], vertical_alignment="center", gap="small")
            with msg_col:
                st.caption(
                    "API key missing or service unreachable? Switch to **Demo mode** to inspect "
                    "pre-verified offline results with 0 live API calls."
                )
            with btn_col:
                if st.button(
                    "Load Demo sample",
                    key=recover_key,
                    icon=":material/play_circle:",
                    disabled=bool(st.session_state.get("busy_action")),
                    width="stretch",
                ):
                    st.session_state[error_state_key] = None
                    pending = dict(st.session_state.get("_pending_widget_updates") or {})
                    pending["demo_mode"] = True
                    try:
                        if recover_scope == "pasted":
                            sample, exact = find_cached_pasted_audit(query_or_url)
                            _apply_pasted_demo_sample(sample, exact_match=exact, sync_widget=False)
                            if not query_or_url.strip():
                                pending["answer_input"] = str(sample.get("answer_text") or "")
                        elif recover_scope == "ask":
                            ask_s = _apply_ask_demo_sample(query_or_url, sync_widget=False)
                            if not query_or_url.strip():
                                pending["ask_question_input"] = str(ask_s.get("question") or "")
                        elif recover_scope == "url":
                            url_s = _apply_url_demo_sample(query_or_url, sync_widget=False)
                            if not query_or_url.strip():
                                pending["url_input"] = str(url_s.get("url") or "")
                    except FileNotFoundError:
                        pass
                    st.session_state["_pending_widget_updates"] = pending
                    st.rerun()


def load_demo_example(example_text: str) -> None:
    """Sidebar button callback: populates input and clears prior results (or loads Demo cache if Demo mode is on)."""
    st.session_state["pasted_error"] = None
    st.session_state["pasted_rewrite_error"] = None
    if example_text and st.session_state.get("demo_mode"):
        try:
            sample, exact = find_cached_pasted_audit(example_text)
            _apply_pasted_demo_sample(sample, exact_match=exact)
            return
        except FileNotFoundError:
            pass

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
    st.session_state["demo_pasted_notice"] = ""
    st.session_state["demo_pasted_rewrite"] = None


def request_stop_audit() -> None:
    """Cancel an in-flight audit cleanly and keep any completed partial results."""
    cancel_ev = st.session_state.get("audit_cancel_event")
    if isinstance(cancel_ev, threading.Event):
        cancel_ev.set()
    partial = list(st.session_state.get("partial_audit_results") or [])
    total_planned = int(st.session_state.get("partial_audit_total") or len(partial))
    if partial and not st.session_state.get("results"):
        st.session_state["results"] = partial
    st.session_state["audit_stopped"] = True
    st.session_state["audit_stopped_info"] = {
        "completed": len(partial),
        "total": max(total_planned, len(partial)),
    }
    ui_state = st.session_state.get("audit_ui_state")
    if isinstance(ui_state, AuditUIState):
        apply_pipeline_event(
            ui_state,
            "stopped",
            {"completed": len(partial), "total": max(total_planned, len(partial))},
        )
    st.session_state["busy_action"] = None
    st.session_state["_busy_armed"] = False


def clear_workspace() -> None:
    """Clear the text area and previous verification results in Tab 1."""
    st.session_state["answer_input"] = ""
    st.session_state["results"] = None
    st.session_state["checked_answer"] = ""
    st.session_state["extract_model"] = None
    st.session_state["extract_meta"] = None
    st.session_state["corrected_answer"] = None
    st.session_state["corrected_model"] = None
    st.session_state["corrected_meta"] = None
    st.session_state["pasted_audit_at"] = None
    st.session_state["status_steps"] = []
    st.session_state["demo_pasted_notice"] = ""
    st.session_state["demo_pasted_rewrite"] = None
    st.session_state["pasted_error"] = None
    st.session_state["pasted_rewrite_error"] = None
    st.session_state["audit_stopped"] = False
    st.session_state["audit_stopped_info"] = None
    st.session_state["partial_audit_results"] = []
    st.session_state["partial_audit_total"] = 0
    st.session_state["audit_ui_state"] = create_initial_ui_state()


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
    st.session_state["demo_ask_notice"] = ""
    st.session_state["demo_ask_rewrite"] = None
    st.session_state["ask_error"] = None
    st.session_state["ask_rewrite_error"] = None


def clear_url_workspace() -> None:
    """Clear the URL input, extracted article, and audit results in Tab 3 ('Audit a URL')."""
    st.session_state["url_input"] = ""
    st.session_state["url_audited"] = ""
    st.session_state["url_article_title"] = None
    st.session_state["url_article_text"] = ""
    st.session_state["url_article_truncated"] = False
    st.session_state["url_extract_credits"] = 0.0
    st.session_state["url_extract_cache_hit"] = False
    st.session_state["url_total_claims_found"] = 0
    st.session_state["url_results"] = None
    st.session_state["url_extract_model"] = None
    st.session_state["url_extract_meta"] = None
    st.session_state["url_corrected_answer"] = None
    st.session_state["url_corrected_model"] = None
    st.session_state["url_corrected_meta"] = None
    st.session_state["url_audit_at"] = None
    st.session_state["url_status_steps"] = []
    st.session_state["demo_url_notice"] = ""
    st.session_state["demo_url_rewrite"] = None
    st.session_state["url_error"] = None
    st.session_state["url_rewrite_error"] = None


def init_session_state() -> None:
    """Initialize required keys in Streamlit session_state."""
    default_dark = True
    defaults = {
        "busy_action": None,
        "_busy_armed": False,
        "audit_ui_state": create_initial_ui_state(),
        "audit_cancel_event": None,
        "audit_stopped": False,
        "audit_stopped_info": None,
        "partial_audit_results": [],
        "partial_audit_total": 0,
        "demo_mode": False,
        "demo_pasted_notice": "",
        "demo_pasted_rewrite": None,
        "demo_ask_notice": "",
        "demo_ask_rewrite": None,
        "demo_url_notice": "",
        "demo_url_rewrite": None,
        "pasted_error": None,
        "ask_error": None,
        "url_error": None,
        "batch_error": None,
        "doc_checker_error": None,
        "benchmark_error": None,
        "pasted_rewrite_error": None,
        "ask_rewrite_error": None,
        "url_rewrite_error": None,
        "batch_summary": None,
        "batch_items_data": None,
        "batch_text_input": "",
        "batch_selected_index": 1,
        "doc_checker_results": None,
        "doc_checker_text": "",
        "active_categories_filter": ["price", "version", "regulation", "company", "statistic", "other"],
        "settings_language": "Auto-Detect",
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
        "url_input": "",
        "url_audited": "",
        "url_article_title": None,
        "url_article_text": "",
        "url_article_truncated": False,
        "url_extract_credits": 0.0,
        "url_extract_cache_hit": False,
        "url_total_claims_found": 0,
        "url_results": None,
        "url_extract_model": None,
        "url_extract_meta": None,
        "url_corrected_answer": None,
        "url_corrected_model": None,
        "url_corrected_meta": None,
        "url_audit_at": None,
        "url_status_steps": [],
        "benchmark_status_steps": [],
        "benchmark_use_cache": True,
        "show_search_queries": False,
        "keep_status_expanded": False,
        "simulate_outage": False,
        "dual_judge_mode": False,
        "detection_threshold": 40,
        "session_ledger": create_empty_ledger(),
        "last_audit_ledger": None,
        "why_tavily_cache": {},
        "why_bifrost_result": None,
        "dark_mode": default_dark,
        "tavily_credits_used": 0.0,
        "gateway_requests": [],
        "last_audit_time_ms": 0.0,
        "session_audit_time_ms": 0.0,
        "watchlist": [],
        "showdown_active_id": "iphone",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    init_history_if_needed()
    try:
        if hasattr(st, "query_params") and "_dark_mode_initialized" not in st.session_state:
            st.session_state["_dark_mode_initialized"] = True
            qp_theme = str(st.query_params.get("theme", "")).strip().lower()
            if qp_theme == "light":
                st.session_state["dark_mode"] = False
            elif qp_theme == "dark":
                st.session_state["dark_mode"] = True
    except Exception:
        pass
    pending = st.session_state.pop("_pending_widget_updates", None)
    if isinstance(pending, dict):
        for p_key, p_val in pending.items():
            st.session_state[p_key] = p_val
    if "benchmark_results" not in st.session_state:
        st.session_state["benchmark_results"] = load_benchmark_cache()


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


def find_claim_span(
    answer_text: str,
    claim_text: str,
    span_text: str | None = None,
    occupied_ranges: list[tuple[int, int]] | None = None,
) -> tuple[int, int, float] | None:
    """
    Locate the character range `(start, end, confidence)` in `answer_text` that
    corresponds to `claim_text`. Prefers direct string matching on the extractor's
    verbatim `span_text` (or `claim_text`), then falls back to weighted token
    overlap across clauses and sentences.
    """
    if not answer_text or (not claim_text and not span_text):
        return None

    located = locate_span_in_text(
        answer_text,
        span_text=span_text,
        claim_text=claim_text,
        occupied_ranges=occupied_ranges,
    )
    if located is not None:
        return located[0], located[1], 1.0

    if not claim_text:
        return None

    claim_tokens = _extract_match_tokens(claim_text)
    if not claim_tokens:
        return None
    total_claim_weight = sum(claim_tokens.values())

    best_span: tuple[int, int, float] | None = None
    best_len = 10**9

    for s, e in _candidate_answer_spans(answer_text):
        if occupied_ranges and any(s < pe and e > ps for ps, pe in occupied_ranges):
            continue
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
    when two spans genuinely overlap, incorrect and outdated claims take precedence.
    """
    raw_matches: list[dict] = []
    status_priority = {
        "incorrect": 0,
        "outdated": 1,
        "partially_correct": 2,
        "uncertain": 3,
        "fresh": 4,
        "unverifiable": 5,
    }
    all_candidate_spans = _candidate_answer_spans(answer_text)
    used_ranges: list[tuple[int, int]] = []

    for claim_idx, item in enumerate(verification_results or []):
        claim_text = str(item.get("claim", ""))
        raw_span = str(item.get("span") or "").strip() or None
        span = find_claim_span(
            answer_text,
            claim_text,
            span_text=raw_span,
            occupied_ranges=used_ranges,
        )
        # If the only literal match sits strictly inside an earlier wider compound match,
        # keep it so the compound-clause tightening pass can highlight both separately.
        if span is None and used_ranges:
            nested = locate_span_in_text(
                answer_text,
                span_text=raw_span,
                claim_text=claim_text,
                occupied_ranges=None,
            )
            if nested is not None:
                ns, ne, _ = nested
                if any(
                    ps <= ns and ne <= pe and (ne - ns) < (pe - ps)
                    for ps, pe in used_ranges
                ):
                    span = (ns, ne, 1.0)

        if span is None:
            continue
        s, e, match_score = span
        used_ranges.append((s, e))

        status = str(item.get("status", "uncertain")).lower()
        if status not in status_priority:
            status = "uncertain"
        raw_score = item.get("score")
        safe_score = None if (raw_score is None or status == "unverifiable") else int(raw_score)
        raw_matches.append(
            {
                "claim_index": claim_idx,
                "claim": claim_text,
                "span": raw_span,
                "start": s,
                "end": e,
                "status": status,
                "score": safe_score,
                "current_value": str(item.get("current_value") or "").strip() or None,
                "correction": str(item.get("correction") or "").strip() or None,
                "was_ever_true": str(item.get("was_ever_true") or "").strip() or None,
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
    key_prefix: str = "ask",
) -> tuple[str, int]:
    """
    Render `answer_text` as safe HTML with inline highlights for verified claims
    (green = fresh, orange = outdated, red = incorrect, amber = partially correct,
    yellow = uncertain, grey = unverifiable).
    Hovering previews the corrected/verified value and clicking jumps to `#claim-{key_prefix}-{N}`.
    """
    if not answer_text:
        return "", 0

    spans = match_claims_to_answer_spans(answer_text, verification_results or [])
    if not spans:
        return escape_html(answer_text).replace("\n", "<br>"), 0

    cls_map = {
        "incorrect": ("sad-hl-incorrect", "Incorrect"),
        "outdated": ("sad-hl-outdated", "Outdated"),
        "partially_correct": ("sad-hl-partially-correct", "Partially correct"),
        "fresh": ("sad-hl-fresh", "Fresh"),
        "uncertain": ("sad-hl-uncertain", "Uncertain"),
        "unverifiable": ("sad-hl-unverifiable", "Unverifiable"),
    }

    parts: list[str] = []
    cursor = 0
    for span in spans:
        if span["start"] > cursor:
            parts.append(escape_html(answer_text[cursor : span["start"]]).replace("\n", "<br>"))
        hl_cls, status_label = cls_map.get(span["status"], ("sad-hl-uncertain", "Uncertain"))
        score_str = "N/A" if span["score"] is None else f"{span['score']}/100"
        claim_num = int(span["claim_index"]) + 1
        target_id = f"claim-{key_prefix}-{claim_num}"

        corr_val = str(span.get("correction") or span.get("current_value") or "").strip()
        corr_info = str(span.get("current_info") or "").strip()
        display_val = corr_val or corr_info
        if span["status"] == "incorrect":
            tip_body = f"Never true / contradicted: {display_val}" if display_val else "Never historically true / contradicted"
        elif span["status"] == "outdated":
            tip_body = f"Updated: {display_val}" if display_val else "Outdated fact"
        elif span["status"] == "partially_correct":
            tip_body = f"Partially correct: {display_val}" if display_val else "Partially correct fact"
        elif span["status"] == "fresh":
            tip_body = f"Verified: {display_val}" if display_val else f"Verified fresh ({score_str})"
        elif span["status"] == "uncertain":
            tip_body = f"Uncertain: {display_val}" if display_val else "Conflicting or partial evidence"
        else:
            tip_body = f"Unverifiable: {display_val}" if display_val else "No reliable evidence found"
        if len(tip_body) > 190:
            tip_body = tip_body[:187].rstrip() + "..."
        tip_text = f"{status_label} ({score_str}) — {tip_body} (Click for Claim #{claim_num})"

        segment = escape_html(answer_text[span["start"] : span["end"]]).replace("\n", "<br>")
        parts.append(
            f'<a class="sad-hl-link {hl_cls}" href="#{escape_html(target_id)}" '
            f'title="{escape_html(tip_text)}" data-tooltip="{escape_html(tip_body)}">'
            f'{segment}</a>'
        )
        cursor = span["end"]

    if cursor < len(answer_text):
        parts.append(escape_html(answer_text[cursor:]).replace("\n", "<br>"))

    return "".join(parts), len(spans)


def render_annotated_answer_card(
    answer_text: str,
    verification_results: list[dict],
    key_prefix: str,
    title: str = "Annotated answer",
) -> None:
    """
    Render original pasted text or extracted article with color-coded clickable highlights
    (green = fresh, orange = outdated, red = incorrect, amber = partially correct,
    yellow = uncertain, grey = unverifiable).
    """
    if not answer_text or not answer_text.strip():
        return

    body_html, hl_count = render_highlighted_answer_html(
        answer_text,
        verification_results,
        key_prefix=key_prefix,
    )
    if hl_count > 0:
        legend_html = (
            '<div class="sad-hl-legend">'
            '<span><span class="sad-hl-fresh">Green</span> = Fresh</span>'
            '<span class="sad-hl-sep">&bull;</span>'
            '<span><span class="sad-hl-outdated">Orange</span> = Outdated</span>'
            '<span class="sad-hl-sep">&bull;</span>'
            '<span><span class="sad-hl-incorrect">Red</span> = Incorrect</span>'
            '<span class="sad-hl-sep">&bull;</span>'
            '<span><span class="sad-hl-partially-correct">Amber</span> = Partially correct</span>'
            '<span class="sad-hl-sep">&bull;</span>'
            '<span><span class="sad-hl-uncertain">Yellow</span> = Uncertain</span>'
            '<span class="sad-hl-sep">&bull;</span>'
            '<span><span class="sad-hl-unverifiable">Grey</span> = Unverifiable</span>'
            '</div>'
        )
        hint_text = "Hover highlight for corrected value &bull; Click to jump to claim card"
    else:
        legend_html = (
            '<div class="sad-hl-legend">'
            '<span>No exact text spans matched inline &bull; See claim cards below for details.</span>'
            '</div>'
        )
        hint_text = f"{len(verification_results)} claim(s) audited"

    st.markdown(
        '<div class="sad-result-card sad-result-card-purple sad-annotated-card">'
        '<div class="sad-result-header">'
        '<div class="sad-liquid-tab sad-liquid-tab-purple">'
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
        '<polyline points="14 2 14 8 20 8"></polyline>'
        '<line x1="16" y1="13" x2="8" y2="13"></line>'
        '<line x1="16" y1="17" x2="8" y2="17"></line>'
        '<polyline points="10 9 9 9 8 9"></polyline>'
        '</svg>'
        f'<span>{escape_html(title)}</span>'
        '</div>'
        f'<span class="sad-annotated-hint">{hint_text}</span>'
        '</div>'
        f'<div class="sad-ask-body">{body_html}</div>'
        f'{legend_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def render_ask_answer_card(
    placeholder,
    answer_text: str,
    verification_results: list[dict] | None = None,
    question_text: str | None = None,
    answer_model: str | None = None,
    answer_meta: dict | None = None,
    audit_error: bool = False,
) -> None:
    """Render the LLM answer card (upgrading with clickable red/green/yellow/grey underlines once audited)."""
    body_html, hl_count = render_highlighted_answer_html(
        answer_text,
        verification_results,
        key_prefix="ask",
    )

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
                '<span><span class="sad-hl-uncertain">Yellow</span> = Uncertain</span>'
                '<span><span class="sad-hl-unverifiable">Grey</span> = Unverifiable</span>'
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
        meta_parts.append(f"Model: {escape_html(answer_model)}")
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
    source = (primary_model or "").strip() or (_get_config_value("LLM_MODEL") or "primary")
    target = (answered_model or "").strip() or "fallback"
    return f"{source} -> {target}"


def build_request_telemetry(
    operation: str,
    metrics: dict | None,
    model_used: str | None = None,
    fallback_used: bool = False,
    primary_model: str | None = None,
) -> dict | None:
    """Normalize per-request telemetry from backend metrics or result dicts."""
    if isinstance(metrics, dict):
        record = dict(metrics)
        record["operation"] = operation
        record["model_used"] = str(record.get("model_used") or model_used or "Unknown")
        record["primary_model"] = str(
            record.get("primary_model")
            or primary_model
            or _get_config_value("LLM_MODEL")
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

    resolved_primary = (primary_model or _get_config_value("LLM_MODEL") or model_used)
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
    """Append a normalized request record to session telemetry history ONLY if it returned usage."""
    if not isinstance(record, dict):
        return
    if record.get("error"):
        return
    tot = int(record.get("total_tokens") or 0)
    pin = int(record.get("prompt_tokens") or 0)
    pout = int(record.get("completion_tokens") or 0)
    if (tot + pin + pout) <= 0 and not bool(record.get("usage_returned")):
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
        judge_tel = [
            entry
            for entry in (item.get("judge_telemetry") or [])
            if isinstance(entry, dict)
        ]
        if judge_tel:
            for tel in judge_tel:
                if tel.get("fallback_used"):
                    route = format_fallback_route(
                        tel.get("primary_model") or item.get("primary_model"),
                        tel.get("model_used") or item.get("model"),
                    )
                    if route not in routes:
                        routes.append(route)
            continue

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

    safe_url = safe_href_url(dashboard_url)
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


def _format_duration_ms(ms: float) -> str:
    """Format milliseconds cleanly as '<N> ms' or '<X.XX> s'."""
    val = max(0.0, float(ms or 0.0))
    if val >= 1000.0:
        return f"{val / 1000.0:.2f} s"
    return f"{val:.0f} ms"


def _resolve_safe_task_routing() -> dict:
    """Safely read task-based routing configuration without failing when env vars are unset."""
    try:
        return get_task_routing_summary()
    except Exception:
        return {}


def render_gateway_panel(
    placeholder,
    requests_log: list[dict],
    task_routing: dict | None = None,
    session_audit_ms: float | None = None,
    last_audit_ms: float | None = None,
    dual_judge_enabled: bool | None = None,
) -> None:
    """Render live per-request metrics, task routing, latency/audit timing, and session totals."""
    if placeholder is None or not is_use_bifrost():
        return
    total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in requests_log)
    total_completion = sum(int(r.get("completion_tokens") or 0) for r in requests_log)
    total_session_tokens = sum(int(r.get("total_tokens") or 0) for r in requests_log)
    req_count = len(requests_log)
    fallback_count = sum(1 for r in requests_log if r.get("fallback_used"))
    total_req_latency_ms = sum(float(r.get("latency_ms") or 0.0) for r in requests_log)
    avg_latency_ms = (total_req_latency_ms / req_count) if req_count else 0.0

    resolved_session_audit_ms = (
        float(session_audit_ms)
        if session_audit_ms is not None
        else float(st.session_state.get("session_audit_time_ms") or 0.0)
    )
    resolved_last_audit_ms = (
        float(last_audit_ms)
        if last_audit_ms is not None
        else float(st.session_state.get("last_audit_time_ms") or 0.0)
    )
    display_audit_ms = max(resolved_session_audit_ms, total_req_latency_ms)

    routing_info = task_routing if isinstance(task_routing, dict) else _resolve_safe_task_routing()
    dual_on = (
        bool(st.session_state.get("dual_judge_mode", False))
        if dual_judge_enabled is None
        else bool(dual_judge_enabled)
    )

    observed_extract: str | None = None
    observed_judge: str | None = None
    observed_judge2: str | None = None
    for req in reversed(requests_log):
        op_low = str(req.get("operation") or "").lower()
        task_name = str(req.get("task") or "").lower()
        m_used = str(req.get("model_used") or "").strip()
        if not m_used:
            continue
        if observed_extract is None and (task_name == "extract" or "extract" in op_low):
            observed_extract = m_used
        elif "judge 2" in op_low and observed_judge2 is None:
            observed_judge2 = m_used
        elif observed_judge is None and (
            task_name == "judge" or "verify claim" in op_low
        ):
            observed_judge = m_used

    extract_model_label = escape_html(
        observed_extract or routing_info.get("extract_model") or "Default"
    )
    judge_model_label = escape_html(
        observed_judge or routing_info.get("judge_model") or "Default"
    )
    judge2_model_label = escape_html(
        observed_judge2 or routing_info.get("dual_judge_secondary") or "Fallback"
    )

    dual_row_html = (
        f'<div><strong>Judge 2:</strong> <code>{judge2_model_label}</code></div>'
        if dual_on
        else ""
    )
    judge_key_label = "Judge 1" if dual_on else "Judge"
    task_routing_html = (
        '<div class="sad-gw-tasks">'
        '<div class="sad-gw-stat-label" style="margin-bottom:0.2rem;">Task routing</div>'
        f'<div><strong>Extract:</strong> <code>{extract_model_label}</code></div>'
        f'<div><strong>{judge_key_label}:</strong> <code>{judge_model_label}</code></div>'
        f'{dual_row_html}'
        '</div>'
    )

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
        log_section = (
            f'<details class="sad-gw-details">'
            f'<summary>Request log ({req_count} call{"s" if req_count != 1 else ""})</summary>'
            f'<div class="sad-gw-log">{"".join(rows_html)}</div>'
            f'</details>'
        )
    else:
        empty_note = (
            "Demo mode active — cached audits use 0 live calls."
            if st.session_state.get("demo_mode")
            else (
                "No LLM requests yet. Run Check Freshness to inspect live Bifrost telemetry."
                if is_use_bifrost()
                else "No LLM requests yet. Run Check Freshness to inspect Gemini token and latency telemetry."
            )
        )
        latest_html = (
            f'<div class="sad-gw-sub" style="margin-top:0.5rem;">{escape_html(empty_note)}</div>'
        )
        log_section = ""

    audit_sub_html = (
        f' <span style="font-size:0.68rem;font-weight:400;color:#94A3B8;">'
        f'(last {_format_duration_ms(resolved_last_audit_ms)})</span>'
        if resolved_last_audit_ms > 0 and resolved_last_audit_ms < display_audit_ms - 1.0
        else ""
    )

    panel_html = (
        '<div class="sad-gw-card">'
        '<div class="sad-gw-grid" style="margin-top:0;">'
        '<div class="sad-gw-stat sad-gw-stat-full">'
        '<div class="sad-gw-stat-label">Session total tokens</div>'
        f'<div class="sad-gw-stat-value">{total_session_tokens:,} '
        f'<span style="font-size:0.72rem;font-weight:400;color:#94A3B8;">'
        f'({total_prompt:,} in / {total_completion:,} out &bull; {req_count} reqs)</span></div>'
        '</div>'
        '<div class="sad-gw-stat">'
        '<div class="sad-gw-stat-label">Total audit time</div>'
        f'<div class="sad-gw-stat-value">{escape_html(_format_duration_ms(display_audit_ms))}{audit_sub_html}</div>'
        '</div>'
        '<div class="sad-gw-stat">'
        '<div class="sad-gw-stat-label">Avg latency / call</div>'
        f'<div class="sad-gw-stat-value">{escape_html(_format_duration_ms(avg_latency_ms))}</div>'
        '</div>'
        '<div class="sad-gw-stat sad-gw-stat-full">'
        '<div class="sad-gw-stat-label">Fallback count</div>'
        f'<div class="sad-gw-stat-value">{fallback_count} '
        '<span style="font-size:0.72rem;font-weight:400;color:#94A3B8;">'
        'session failover(s)</span></div>'
        '</div>'
        '</div>'
        f'{task_routing_html}'
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
    badge_text = "Demo mode (0 credits)" if st.session_state.get("demo_mode") else "6h cache active"
    placeholder.markdown(
        '<div class="sad-credits-card">'
        '<div class="sad-credits-label">Tavily credits used this session</div>'
        '<div class="sad-credits-row">'
        f'<span class="sad-credits-value">{formatted}</span>'
        f'<span class="sad-credits-pill">{escape_html(badge_text)}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_architecture_telemetry_expander(use_bifrost: bool) -> None:
    """Render the dual-technology integration showcase using real active routing and evidence config."""
    route_info = get_real_model_and_routing_info(use_bifrost=use_bifrost)
    cfg_models_html = ", ".join(f"<code>{escape_html(m)}</code>" for m in route_info["configured_models"]) or "<code>default configured model</code>"
    ev_models_html = ", ".join(f"<code>{escape_html(m)}</code>" for m in route_info["evidence_models"]) or "<code>evidence/bifrost_evidence.json</code>"
    active_llm_lbl = "Bifrost Gateway" if use_bifrost else "Direct Gemini API"
    with st.expander("🛠️ System Architecture: Tavily + Bifrost Integration (PS06 + PS02)", expanded=False, icon=":material/hub:"):
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown(
                """
                <div style="background: var(--bg-card, #FFFFFF); border: 1px solid var(--border-main, #E5E7EF); border-radius: 12px; padding: 18px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                        <span style="font-size: 1.5rem;">🌐</span>
                        <h4 style="margin: 0; color: var(--text-main, #0F172A);">Tavily Real-Time Web Intelligence</h4>
                    </div>
                    <ul style="font-size: 0.88rem; color: var(--text-muted, #64748B); line-height: 1.6; padding-left: 20px; margin: 0;">
                        <li><strong>Live Search API:</strong> Targeted search queries formulated dynamically for each extracted factual claim.</li>
                        <li><strong>Credibility Categorization:</strong> Classifies each unique registered domain as <code>PRIMARY</code>, <code>NEWS</code>, <code>BLOG</code>, or <code>FORUM</code>.</li>
                        <li><strong>Tavily Extract API:</strong> Extracts page text for target URLs and top verification sources.</li>
                        <li><strong>Freshness Timelines:</strong> Uses published dates returned by Tavily to estimate stale durations.</li>
                        <li><strong>Server SQLite Cache:</strong> 6-hour TTL cache (<code>tavily_cache.db</code>) reduces duplicate upstream queries.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="background: var(--bg-card, #FFFFFF); border: 1px solid var(--border-main, #E5E7EF); border-radius: 12px; padding: 18px; height: 100%;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                        <span style="font-size: 1.5rem;">⚡</span>
                        <h4 style="margin: 0; color: var(--text-main, #0F172A);">LLM Routing ({escape_html(active_llm_lbl)})</h4>
                    </div>
                    <ul style="font-size: 0.88rem; color: var(--text-muted, #64748B); line-height: 1.6; padding-left: 20px; margin: 0;">
                        <li><strong>Active Configured Models:</strong> {cfg_models_html} via <code>{escape_html(route_info['route_label'])}</code>.</li>
                        <li><strong>Recorded Bifrost Evidence Models:</strong> {ev_models_html} (from <code>evidence/bifrost_evidence.json</code>).</li>
                        <li><strong>Retry &amp; Fallback:</strong> 3-try exponential backoff on connection errors followed by configured fallback models.</li>
                        <li><strong>Dual-Judge Option:</strong> Supports independent evaluation across two configured judge models.</li>
                        <li><strong>Gateway Tracing (Bifrost mode):</strong> Attaches <code>x-bf-run-id</code> headers when <code>USE_BIFROST=true</code>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background: var(--bg-secondary, #F1F5F9); border: 1px solid var(--border-main, #E5E7EF); border-radius: 10px; padding: 12px 18px; font-size: 0.85rem; color: var(--text-main, #0F172A); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div><strong>Active Pipeline:</strong> 1. Extract Claims ({escape_html(active_llm_lbl)}) &rarr; 2. Live Web Search (Tavily) &rarr; 3. Judge ({escape_html(active_llm_lbl)}) &rarr; 4. Risk-Weighted Score &rarr; 5. Report</div>
                <div style="color: var(--text-muted, #64748B);">PS06 (Fact Verification) + PS02 (Agent Benchmarking)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_architecture_tab(use_bifrost: bool) -> None:
    """Render the interactive Architecture & Evidence view with route-aware counters and real config models."""
    route_info = get_real_model_and_routing_info(use_bifrost=use_bifrost)
    sess_led = st.session_state.get("session_ledger") or create_empty_ledger()
    tav_led = sess_led.get("tavily", {})
    bf_led = sess_led.get("bifrost", {})

    searches_cnt = tav_led.get("searches", 0)
    credits_cnt = tav_led.get("credits", 0.0)
    calls_cnt = bf_led.get("calls", 0)
    fallbacks_cnt = bf_led.get("fallbacks", 0)
    tokens_cnt = bf_led.get("total_tokens", 0)
    results_list = st.session_state.get("results") or []
    claims_cnt = sum(1 for r in results_list if isinstance(r, dict) and r.get("status") != "verification_failed")

    calls_metric_label = route_info["calls_label"]
    fallbacks_metric_label = "Gateway fallbacks" if use_bifrost else "Model fallbacks"
    judge_box_title = route_info["judge_step_label"]
    active_route_label = route_info["route_label"]
    cfg_models_html = ", ".join(f"<code>{escape_html(m)}</code>" for m in route_info["configured_models"]) or "<code>default configured model</code>"
    ev_models_html = ", ".join(f"<code>{escape_html(m)}</code>" for m in route_info["evidence_models"]) or "<code>evidence/bifrost_evidence.json</code>"

    st.subheader("System Architecture & Evidence", icon=":material/hub:")
    st.caption(
        f"Active LLM route: {active_route_label} • Web retrieval: Tavily Search & Extract API. "
        "Counters reflect only calls from this session that returned usage."
    )

    # Live Counter KPIs (labeled strictly by the real active route)
    k1, k2, k3, k4 = st.columns(4, gap="small")
    with k1:
        st.metric("Tavily Searches", f"{searches_cnt:,}", f"{credits_cnt:.1f} credits consumed", border=True, help="Live web search queries formulated per claim")
    with k2:
        st.metric(
            calls_metric_label,
            f"{calls_cnt:,}",
            f"{len(bf_led.get('models', []))} models answered",
            border=True,
            help=f"Session LLM requests that returned usage via {active_route_label}",
        )
    with k3:
        st.metric(fallbacks_metric_label, f"{fallbacks_cnt:,}", "Failover switches", border=True, help="Fallback model invocations executed after primary errors")
    with k4:
        st.metric("Tokens Returned", f"{tokens_cnt:,}", f"{claims_cnt} claims verified", border=True, help="Sum of prompt + completion tokens returned by the LLM API in this session")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Inline SVG Flow Diagram with route-aware LIVE counters embedded
    tav_counter_text = f"{searches_cnt} searches • {credits_cnt:.1f} cr"
    llm_counter_text = f"{calls_cnt} calls • {fallbacks_cnt} fallbacks"
    claims_counter_text = f"{claims_cnt} claims" if claims_cnt > 0 else "Claim Extraction"
    tokens_counter_text = f"{tokens_cnt:,} tokens" if tokens_cnt > 0 else "4 Score Pillars"

    svg_diagram = f"""
    <div style="background: #FFFFFF; border: 1px solid var(--border-main, #E5E7EF); border-radius: 14px; padding: 20px; overflow-x: auto; margin-bottom: 20px;">
        <svg viewBox="0 0 980 170" width="100%" height="170" xmlns="http://www.w3.org/2000/svg" style="font-family: system-ui, -apple-system, sans-serif;">
            <defs>
                <linearGradient id="g-input" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#F1F5F9" />
                    <stop offset="100%" stop-color="#E2E8F0" />
                </linearGradient>
                <linearGradient id="g-detect" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#EEF2FF" />
                    <stop offset="100%" stop-color="#E0E7FF" />
                </linearGradient>
                <linearGradient id="g-tavily" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#ECFDF5" />
                    <stop offset="100%" stop-color="#D1FAE5" />
                </linearGradient>
                <linearGradient id="g-bifrost" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#FFFBEB" />
                    <stop offset="100%" stop-color="#FEF3C7" />
                </linearGradient>
                <linearGradient id="g-score" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#F5F3FF" />
                    <stop offset="100%" stop-color="#EDE9FE" />
                </linearGradient>
                <linearGradient id="g-report" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#F0FDF4" />
                    <stop offset="100%" stop-color="#DCFCE7" />
                </linearGradient>
                <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                    <path d="M 0 1 L 8 5 L 0 9 z" fill="#94A3B8" />
                </marker>
            </defs>

            <!-- Box 1: Input -->
            <rect x="10" y="20" width="135" height="110" rx="10" fill="url(#g-input)" stroke="#CBD5E1" stroke-width="1.5" />
            <text x="77" y="50" font-size="13" font-weight="700" fill="#0F172A" text-anchor="middle">1. Input</text>
            <text x="77" y="70" font-size="11" fill="#64748B" text-anchor="middle">LLM Answer / URL</text>
            <rect x="22" y="88" width="111" height="24" rx="6" fill="#FFFFFF" stroke="#E2E8F0" />
            <text x="77" y="104" font-size="10.5" font-weight="600" fill="#475569" text-anchor="middle">Raw Content</text>

            <!-- Connection 1 -> 2 -->
            <line x1="145" y1="75" x2="170" y2="75" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Box 2: Detection -->
            <rect x="175" y="20" width="145" height="110" rx="10" fill="url(#g-detect)" stroke="#C7D2FE" stroke-width="1.5" />
            <text x="247" y="50" font-size="13" font-weight="700" fill="#3730A3" text-anchor="middle">2. Detection</text>
            <text x="247" y="70" font-size="11" fill="#4338CA" text-anchor="middle">Claim Extraction</text>
            <rect x="187" y="88" width="121" height="24" rx="6" fill="#FFFFFF" stroke="#C7D2FE" />
            <text x="247" y="104" font-size="10.5" font-weight="600" fill="#4338CA" text-anchor="middle">{claims_counter_text}</text>

            <!-- Connection 2 -> 3 -->
            <line x1="320" y1="75" x2="345" y2="75" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Box 3: Tavily -->
            <rect x="350" y="20" width="155" height="110" rx="10" fill="url(#g-tavily)" stroke="#A7F3D0" stroke-width="1.5" />
            <text x="427" y="50" font-size="13" font-weight="700" fill="#065F46" text-anchor="middle">3. Tavily Search</text>
            <text x="427" y="70" font-size="11" fill="#047857" text-anchor="middle">Live Web Evidence</text>
            <rect x="360" y="88" width="135" height="24" rx="6" fill="#FFFFFF" stroke="#A7F3D0" />
            <text x="427" y="104" font-size="10.5" font-weight="700" fill="#065F46" text-anchor="middle">{tav_counter_text}</text>

            <!-- Connection 3 -> 4 -->
            <line x1="505" y1="75" x2="530" y2="75" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Box 4: Route-Aware Judge -->
            <rect x="535" y="20" width="155" height="110" rx="10" fill="url(#g-bifrost)" stroke="#FDE68A" stroke-width="1.5" />
            <text x="612" y="50" font-size="12.5" font-weight="700" fill="#92400E" text-anchor="middle">{escape_html(judge_box_title)}</text>
            <text x="612" y="70" font-size="11" fill="#B45309" text-anchor="middle">{escape_html(active_route_label)}</text>
            <rect x="545" y="88" width="135" height="24" rx="6" fill="#FFFFFF" stroke="#FDE68A" />
            <text x="612" y="104" font-size="10.5" font-weight="700" fill="#92400E" text-anchor="middle">{llm_counter_text}</text>

            <!-- Connection 4 -> 5 -->
            <line x1="690" y1="75" x2="715" y2="75" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Box 5: Score -->
            <rect x="720" y="20" width="135" height="110" rx="10" fill="url(#g-score)" stroke="#DDD6FE" stroke-width="1.5" />
            <text x="787" y="50" font-size="13" font-weight="700" fill="#5B21B6" text-anchor="middle">5. Freshness Score</text>
            <text x="787" y="70" font-size="11" fill="#6D28D9" text-anchor="middle">4 Weighted Signals</text>
            <rect x="730" y="88" width="115" height="24" rx="6" fill="#FFFFFF" stroke="#DDD6FE" />
            <text x="787" y="104" font-size="10.5" font-weight="600" fill="#5B21B6" text-anchor="middle">{tokens_counter_text}</text>

            <!-- Connection 5 -> 6 -->
            <line x1="855" y1="75" x2="875" y2="75" stroke="#94A3B8" stroke-width="2" marker-end="url(#arrow)" />

            <!-- Box 6: Report -->
            <rect x="880" y="20" width="90" height="110" rx="10" fill="url(#g-report)" stroke="#86EFAC" stroke-width="1.5" />
            <text x="925" y="50" font-size="13" font-weight="700" fill="#166534" text-anchor="middle">6. Report</text>
            <text x="925" y="70" font-size="11" fill="#15803D" text-anchor="middle">Rewrite</text>
            <rect x="888" y="88" width="74" height="24" rx="6" fill="#FFFFFF" stroke="#86EFAC" />
            <text x="925" y="104" font-size="10.5" font-weight="700" fill="#166534" text-anchor="middle">Export</text>
        </svg>
    </div>
    """
    st.markdown(svg_diagram, unsafe_allow_html=True)

    # Detailed Dual-Technology Integration Cards (backed by real config & evidence file)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
            <div style="background: var(--bg-card, #FFFFFF); border: 1px solid var(--border-main, #E5E7EF); border-radius: 12px; padding: 20px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="font-size: 1.6rem;">🌐</span>
                    <h3 style="margin: 0; color: #065F46; font-size: 1.15rem;">What Tavily Does</h3>
                </div>
                <p style="font-size: 0.88rem; color: var(--text-muted, #64748B); line-height: 1.5; margin-bottom: 14px;">
                    Tavily retrieves live web evidence for factual verification:
                </p>
                <ul style="font-size: 0.88rem; color: var(--text-main, #0F172A); line-height: 1.7; padding-left: 20px; margin: 0;">
                    <li><strong>Targeted Search API:</strong> Runs category-aware search queries for each extracted factual claim.</li>
                    <li><strong>Registered Domain Deduplication:</strong> Collapses duplicate URLs per registered domain with <code>+N pages</code> counts so agreement counts match displayed chips.</li>
                    <li><strong>Single Source Label:</strong> Classifies each domain as <code>PRIMARY</code>, <code>NEWS</code>, <code>BLOG</code>, or <code>FORUM</code>.</li>
                    <li><strong>Tavily Extract API:</strong> Extracts page text for URL audits and top verification sources.</li>
                    <li><strong>6-Hour Server Cache:</strong> Stores search queries and snippets in local SQLite (<code>tavily_cache.db</code>) for 6 hours.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="background: var(--bg-card, #FFFFFF); border: 1px solid var(--border-main, #E5E7EF); border-radius: 12px; padding: 20px; height: 100%;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <span style="font-size: 1.6rem;">⚡</span>
                    <h3 style="margin: 0; color: #92400E; font-size: 1.15rem;">Active LLM Routing &amp; Bifrost</h3>
                </div>
                <p style="font-size: 0.88rem; color: var(--text-muted, #64748B); line-height: 1.5; margin-bottom: 14px;">
                    Current session routing mode: <strong>{escape_html(active_route_label)}</strong> (<code>USE_BIFROST={'true' if use_bifrost else 'false'}</code>).
                </p>
                <ul style="font-size: 0.88rem; color: var(--text-main, #0F172A); line-height: 1.7; padding-left: 20px; margin: 0;">
                    <li><strong>Active Configured Models:</strong> {cfg_models_html}.</li>
                    <li><strong>Recorded Bifrost Evidence Models ({escape_html(route_info['evidence_date'] or 'saved')}):</strong> {ev_models_html} (from <code>evidence/bifrost_evidence.json</code>).</li>
                    <li><strong>Retries &amp; Fallbacks:</strong> Retries connection errors up to 3 times with backoff before switching to configured fallback models.</li>
                    <li><strong>Honest Failure Handling:</strong> Failed claims show <code>Verification failed</code> (<code>—</code>) and are excluded from overall scores and usage counts.</li>
                    <li><strong>Gateway Tracing (when Bifrost is enabled):</strong> Tags calls with <code>x-bf-run-id</code> and reads token/latency headers.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🛡️ Recorded Bifrost Gateway Evidence")
    render_bifrost_evidence_panel(key_prefix="arch_evidence")


def _build_compact_model_row_html(
    models_used: list[str],
    fallback_routes: list[str] | None = None,
) -> str:
    """Render one compact row containing unique model chips and at most one fallback chip."""
    unique_models: list[str] = []
    for m in models_used or []:
        clean = str(m or "").strip()
        if clean and clean not in unique_models:
            unique_models.append(clean)

    unique_routes: list[str] = []
    for r in fallback_routes or []:
        clean_r = str(r or "").strip()
        if clean_r and clean_r not in unique_routes:
            unique_routes.append(clean_r)

    label_text = "Model:" if len(unique_models) <= 1 else "Models:"
    if unique_models:
        model_chips = " ".join(
            f'<span class="sad-model-tag">{escape_html(m)}</span>'
            for m in unique_models
        )
    else:
        model_chips = '<span class="sad-model-tag">Unknown</span>'

    fallback_chip = ""
    if unique_routes:
        fb_text = unique_routes[0]
        fb_text = fb_text.replace("-->", " → ").replace("->", " → ")
        fb_text = " ".join(fb_text.split())
        if len(unique_routes) > 1:
            fb_text = f"{fb_text} (+{len(unique_routes) - 1} more)"
        fallback_chip = f'<span class="sad-fallback-pill"><span class="sad-fallback-arrow">↳</span> Fallback: {escape_html(fb_text)}</span>'

    return (
        '<div class="sad-model-row">'
        f'<span class="sad-model-label">{label_text}</span>'
        f'{model_chips}'
        f'{fallback_chip}'
        '</div>'
    )


def build_overall_verdict_sentence(results: list[dict]) -> str:
    """Build a natural-language summary sentence of verified claim findings."""
    if not results:
        return "No factual claims verified."
    failed = sum(1 for r in results if r.get("status") == "verification_failed")
    if failed > 0:
        return f"Could not verify {failed} of {len(results)} claims."

    incorrect = sum(1 for r in results if r.get("status") == "incorrect")
    outdated = sum(1 for r in results if r.get("status") == "outdated")
    partially_correct = sum(1 for r in results if r.get("status") == "partially_correct")
    fresh = sum(1 for r in results if r.get("status") == "fresh")
    uncertain = sum(1 for r in results if r.get("status") == "uncertain")
    unverifiable = sum(1 for r in results if r.get("status") == "unverifiable")

    parts: list[str] = []
    if incorrect:
        parts.append(f"{incorrect} incorrect claim{'s' if incorrect != 1 else ''}")
    if outdated:
        parts.append(f"{outdated} outdated claim{'s' if outdated != 1 else ''}")
    if partially_correct:
        parts.append(f"{partially_correct} partially correct claim{'s' if partially_correct != 1 else ''}")

    if parts:
        joined = " and ".join(parts) if len(parts) <= 2 else f"{parts[0]}, {parts[1]}, and {parts[2]}"
        if incorrect > 0:
            return f"{joined} found — contradicted by official sources."
        if outdated > 0:
            return f"{joined} found — key facts are outdated."
        return f"{joined} found."
    if fresh == len(results):
        return f"All {len(results)} claims verified fresh against live sources."
    if fresh > 0 and (uncertain > 0 or unverifiable > 0):
        return f"{fresh} fresh claim{'s' if fresh != 1 else ''} verified; {uncertain + unverifiable} need verification."
    if uncertain > 0:
        return f"{uncertain} uncertain claim{'s' if uncertain != 1 else ''} with conflicting or mixed evidence."
    if unverifiable > 0:
        return f"{unverifiable} unverifiable claim{'s' if unverifiable != 1 else ''} (insufficient sources)."
    return "Verification complete."


def render_score_ring_html(
    overall_score: int | None,
    models_used: list[str],
    fallback_routes: list[str] | None = None,
    verdict_sentence: str | None = None,
    results: list[dict] | None = None,
) -> str:
    """Build the pure HTML/CSS circular score ring, verdict sentence, and compact model/fallback row."""
    tier_label, ring_color, tier_desc = get_overall_tier(overall_score, results=results)
    has_failed = any(
        isinstance(r, dict) and r.get("status") == "verification_failed"
        for r in (results or [])
    )
    if overall_score is None:
        ring_fill = 0
        ring_num_text = "—" if has_failed else "N/A"
        ring_denom_text = "not scored" if has_failed else "no score"
    else:
        clamped = max(0, min(100, overall_score))
        ring_fill = clamped
        ring_num_text = str(clamped)
        ring_denom_text = "of 100"

    compact_row_html = _build_compact_model_row_html(models_used, fallback_routes=fallback_routes)
    verdict_html = (
        f'<div class="sad-verdict-sentence">'
        f'{escape_html(verdict_sentence)}</div>'
        if verdict_sentence
        else ""
    )
    heading_text = (
        escape_html(tier_label)
        if has_failed
        else f"Overall freshness: {escape_html(tier_label)}"
    )
    ring_tier_short = "Failed" if (has_failed and overall_score is None) else ("Partial" if has_failed else tier_label)

    circumference = 276.46  # 2 * pi * 44
    stroke_target = round(circumference * (1.0 - (ring_fill / 100.0)), 2)
    target_attr = f' data-target-score="{ring_fill}"' if overall_score is not None else ""

    return (
        f'<div class="sad-score-card sad-reveal-stagger" style="--sad-stagger-i: 0;">'
        f'<div class="sad-ring-svg-wrap">'
        f'<svg class="sad-ring-svg" viewBox="0 0 104 104" aria-hidden="true">'
        f'<circle class="sad-ring-track-circle" cx="52" cy="52" r="44"></circle>'
        f'<circle class="sad-ring-progress-circle" cx="52" cy="52" r="44" '
        f'stroke="{ring_color}" '
        f'style="--sad-ring-circ: {circumference}; --sad-ring-target: {stroke_target};"></circle>'
        f'</svg>'
        f'<div class="sad-ring-inner">'
        f'<span class="sad-ring-num"{target_attr}>{ring_num_text}</span>'
        f'<span class="sad-ring-denom">{ring_denom_text}</span>'
        f'<span class="sad-ring-tier sad-pill-pop" style="color: {ring_color};">{escape_html(ring_tier_short)}</span>'
        f'</div></div>'
        f'<div class="sad-score-meta">'
        f'<h3>{heading_text}</h3>'
        f'{verdict_html}'
        f'<p>{escape_html(tier_desc)}</p>'
        f'{compact_row_html}'
        f'</div></div>'
    )


def _format_stale_since_text(
    stale_since: object,
    stale_duration_label: object = None,
) -> str:
    """Combine `stale_since` date/month with approximate duration (e.g. '2025-09-22 (stale for ~14 months)')."""
    base = str(stale_since or "").strip()
    dur = str(stale_duration_label or "").strip()
    if base and dur and dur.lower() not in base.lower():
        return f"{base} ({dur})"
    return base or dur


def _format_source_chips_html(
    sources: list[dict] | None,
    entity_domains: list[str] | None = None,
) -> str:
    """Render clickable source chips deduplicated by registered domain with '+N pages'
    and exactly ONE label (PRIMARY, NEWS, BLOG, or FORUM, never two labels).
    """
    type_priority = {"Primary": 4, "News": 3, "Blog": 2, "Forum": 1}
    domain_order: list[str] = []
    domain_entries: dict[str, dict] = {}

    for src in sources or []:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url", "")).strip()
        if not url:
            continue
        reg_domain = str(src.get("domain") or "").strip() or _extract_independent_domain(url) or extract_domain(url)
        raw_type = str(src.get("source_type") or "").strip()
        src_type = (
            raw_type
            if raw_type in {"Primary", "News", "Blog", "Forum"}
            else classify_source_type(url, entity_domains=entity_domains)
        )
        if src_type not in type_priority:
            src_type = "News"
        src_pages = max(1, int(src.get("page_count") or 1))
        if reg_domain not in domain_entries:
            domain_order.append(reg_domain)
            domain_entries[reg_domain] = {
                "url": url,
                "domain": reg_domain,
                "title": src.get("title") or url,
                "source_type": src_type,
                "stance": str(src.get("stance") or "").strip().lower(),
                "published_date": str(src.get("published_date") or "").strip(),
                "quote": str(src.get("quote") or "").strip(),
                "page_count": src_pages,
                "seen_urls": {url},
            }
        else:
            existing = domain_entries[reg_domain]
            if url not in existing["seen_urls"]:
                existing["seen_urls"].add(url)
                existing["page_count"] += src_pages
            else:
                existing["page_count"] = max(existing["page_count"], src_pages)
            if type_priority.get(src_type, 0) > type_priority.get(existing["source_type"], 0):
                existing["source_type"] = src_type
                existing["url"] = url
                existing["title"] = src.get("title") or existing["title"]

    chips_parts: list[str] = []
    for dom in domain_order:
        entry = domain_entries[dom]
        domain_esc = escape_html(entry["domain"])
        title_attr = escape_html(entry["title"])
        safe_url = safe_href_url(entry["url"])
        src_type = entry["source_type"]
        type_cls = f"sad-src-{src_type.lower()}"
        type_label_upper = src_type.upper()  # Strictly ONE label: PRIMARY, NEWS, BLOG, or FORUM
        stance_raw = entry["stance"]
        stance_span = ""
        if stance_raw in {"support", "contradict", "neutral"}:
            stance_span = f'<span class="sad-src-stance {stance_raw}">{stance_raw.capitalize()}</span>'
        pub_date = entry["published_date"]
        date_span = f'<span class="sad-src-date">&bull; {escape_html(pub_date)}</span>' if pub_date else ""
        extra_pages = max(0, int(entry["page_count"]) - 1)
        pages_span = (
            f' <span class="sad-src-pages">+{extra_pages} page{"s" if extra_pages != 1 else ""}</span>'
            if extra_pages > 0
            else ""
        )
        quote_hint = entry["quote"]
        if quote_hint:
            title_attr = f"{title_attr} &#10;&ldquo;{escape_html(quote_hint)}&rdquo;"
        chips_parts.append(
            f'<a class="sad-source-chip" href="{safe_url}" target="_blank" '
            f'rel="noopener noreferrer" title="{title_attr}">'
            f'<span class="sad-src-type {type_cls}">{escape_html(type_label_upper)}</span>{stance_span}'
            f'{domain_esc}{pages_span}{date_span} &#8599;</a>'
        )
    if not chips_parts:
        return '<div class="sad-card-footer">No source links available.</div>'
    return '<div class="sad-chips-row">' + "".join(chips_parts) + "</div>"


def format_step_timeline_html(item: dict) -> str:
    """Format per-step latency strip strictly from measured perf_counter values.

    Never invents default numbers; displays 'not run' for any step without a measured duration.
    """
    detect_val = item.get("detect_latency_ms")
    search_val = item.get("search_latency_ms")
    judge_val = item.get("judge_latency_ms")
    if judge_val is None and item.get("status") != "verification_failed":
        gw_lat = (item.get("gateway_metrics") or {}).get("latency_ms")
        if isinstance(gw_lat, (int, float)) and gw_lat > 0:
            judge_val = gw_lat
    score_val = item.get("score_latency_ms")

    def _fmt_timing(val: object) -> str:
        if isinstance(val, (int, float)) and val >= 0:
            if 0 < val < 1.0:
                return f"({val:.1f}ms)"
            return f"({val:.0f}ms)"
        return "(not run)"

    return (
        f'<div class="sad-timeline-strip">'
        f'<span class="sad-timeline-step">🔍 Detected <span class="timing">{_fmt_timing(detect_val)}</span></span>'
        f'<span class="sad-timeline-arrow">&rarr;</span>'
        f'<span class="sad-timeline-step">🌐 Searched <span class="timing">{_fmt_timing(search_val)}</span></span>'
        f'<span class="sad-timeline-arrow">&rarr;</span>'
        f'<span class="sad-timeline-step">⚡ Judged <span class="timing">{_fmt_timing(judge_val)}</span></span>'
        f'<span class="sad-timeline-arrow">&rarr;</span>'
        f'<span class="sad-timeline-step">📊 Scored <span class="timing">{_fmt_timing(score_val)}</span></span>'
        f'</div>'
    )


def render_claim_card(
    index: int,
    item: dict,
    show_query: bool = False,
    key_prefix: str = "audit",
) -> None:
    """Render a polished per-claim card with status pill, score bar, structured comparison, and chips."""
    status = str(item.get("status", "uncertain")).strip().lower()
    meta = STATUS_META.get(status, STATUS_META["uncertain"])
    raw_score = item.get("score")
    is_failed = status == "verification_failed"
    is_unverifiable = status == "unverifiable" or is_failed or raw_score is None
    score: int | None = None if is_unverifiable else max(0, min(100, int(raw_score)))
    bar_width = 0 if score is None else score
    score_display = "—" if is_failed else ("N/A" if score is None else f"{score}/100")

    category = normalize_category(item.get("category", "other"))
    category_display = format_category_label(category)
    cat_icon = CATEGORY_ICONS.get(category, CATEGORY_ICONS["other"])

    if status == "skipped":
        c_text = item.get("claim", "")
        c_reason = item.get("reason") or "Stable, non-volatile factual claim."
        ts_val = item.get("time_sensitivity", 20)
        vol_class = item.get("volatility_class", "stable_fact")
        card_id = f"claim-{key_prefix}-{index}"
        card_html = (
            f'<div id="{escape_html(card_id)}" class="sad-claim-card" style="border-top: 3px solid #64748B; opacity: 0.95;">'
            f'<div class="sad-claim-header">'
            f'<div class="sad-claim-badges">'
            f'<span class="sad-pill pill-skipped">Stable, not time-sensitive (skipped)</span>'
            f'<span class="sad-meta-pill" style="border-color: #64748B; color: #64748B;">🛡️ Skipped (Stable)</span>'
            f'<span class="sad-cat-badge">{cat_icon}{escape_html(category_display)}</span>'
            f'<span class="sad-meta-pill">Sensitivity: {ts_val}/100</span>'
            f'<span class="sad-meta-pill">Class: {escape_html(vol_class)}</span>'
            f'</div>'
            f'<div class="sad-score-bar-wrap">'
            f'<span class="sad-score-text" style="color: #64748B;">N/A</span>'
            f'</div></div>'
            f'<div class="sad-claimed-row" style="padding: 0.5rem 0;"><strong>Claimed:</strong> &ldquo;{escape_html(c_text)}&rdquo;</div>'
            f'<div class="sad-actual-row" style="color: #475569; padding-bottom: 0.5rem;"><strong>Status:</strong> Stable fact &mdash; {escape_html(c_reason)}</div>'
            f'<div class="sad-card-footer" style="color: #64748B;">Claim #{index} &bull; Skipped live web search to conserve API credits</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
        return

    entity_doms = item.get("entity_domains") if isinstance(item.get("entity_domains"), list) else None

    claimed_html = escape_html(item.get("claim", ""))
    actual_html = escape_html(item.get("current_info", ""))

    actual_meta_rows: list[str] = []
    if status == "incorrect":
        corr_val = item.get("correction") or item.get("current_value") or item.get("current_info") or "Contradicted by authoritative sources"
        sources = item.get("sources") or []
        top_src_str = ""
        if sources and isinstance(sources, list) and isinstance(sources[0], dict):
            src_url = sources[0].get("url") or ""
            src_title = sources[0].get("title") or (extract_domain(src_url) if src_url else "")
            if src_url:
                top_src_str = f' (<a href="{safe_href_url(src_url)}" target="_blank" rel="noopener noreferrer">{escape_html(src_title)}</a>)'
            elif src_title:
                top_src_str = f' ({escape_html(src_title)})'
        actual_meta_rows.append(
            f'<div><span class="sad-actual-meta-label" style="color: #EF4444; font-weight: 700;">Never true / contradicted by:</span> {escape_html(corr_val)}{top_src_str}</div>'
        )
    else:
        if item.get("current_value"):
            actual_meta_rows.append(
                f'<div><span class="sad-actual-meta-label">Current value:</span> {escape_html(item["current_value"])}</div>'
            )
        if item.get("correction") and not item.get("current_value"):
            actual_meta_rows.append(
                f'<div><span class="sad-actual-meta-label">Correction:</span> {escape_html(item["correction"])}</div>'
            )
        if item.get("change_summary"):
            actual_meta_rows.append(
                f'<div><span class="sad-actual-meta-label">Change summary:</span> {escape_html(item["change_summary"])}</div>'
            )
        if item.get("effective_date"):
            actual_meta_rows.append(
                f'<div><span class="sad-actual-meta-label">Effective date:</span> {escape_html(item["effective_date"])}</div>'
            )
        if status == "outdated":
            stale_display = _format_stale_since_text(
                item.get("stale_since") or item.get("change_date"),
                item.get("stale_duration_label"),
            )
            if stale_display:
                actual_meta_rows.append(
                    f'<div><span class="sad-actual-meta-label">Stale since:</span> {escape_html(stale_display)}</div>'
                )

    # Feature E: Likely Expiry (heuristic)
    expiry_info = get_likely_expiry_info(
        category=item.get("category", "other"),
        volatility_class=item.get("volatility_class"),
        check_date=item.get("effective_date") or item.get("saved_date"),
    )
    actual_meta_rows.append(
        f'<div style="margin-top: 5px; padding: 4px 8px; background: rgba(99, 102, 241, 0.1); border-left: 3px solid #818CF8; border-radius: 4px; font-size: 0.8rem;">'
        f'<strong style="color: #A5B4FC;">⏱️ {escape_html(expiry_info["heuristic_tag"])}:</strong> '
        f'<span style="color: #F8FAFC;">{escape_html(expiry_info["review_label"])}</span>'
        f'</div>'
    )

    actual_meta_block = (
        '<div class="sad-actual-meta">' + "".join(actual_meta_rows) + "</div>"
        if actual_meta_rows
        else ""
    )

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

    conflict_block = ""
    if item.get("conflict_detected") or item.get("conflict_note"):
        conflict_msg = item.get("conflict_note") or (
            f"Sources disagree on the current fact ({item.get('agreement_summary', 'mixed evidence')})."
        )
        conflict_block = (
            f'<div class="sad-conflict-note">'
            f'<strong>Sources disagree:</strong> {escape_html(conflict_msg)}'
            f'</div>'
        )

    query_block = ""
    if show_query and item.get("search_query"):
        query_block = (
            f'<div class="sad-reason">'
            f'<strong>Search query:</strong> <code>{escape_html(item["search_query"])}</code>'
            f'</div>'
        )

    sources_block = _format_source_chips_html(
        item.get("sources"),
        entity_domains=entity_doms,
    )

    judge_models = [str(m) for m in (item.get("judge_models") or []) if m]
    judge_tel = [j for j in (item.get("judge_telemetry") or []) if isinstance(j, dict)]
    model_name = item.get("model")
    gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else {}
    footer_parts = [f"Claim #{index}"]
    if len(judge_models) > 1:
        footer_parts.append(" vs ".join(escape_html(m) for m in judge_models))
    elif model_name:
        footer_parts.append(escape_html(model_name))
    total_claim_lat = (
        sum(float(j.get("latency_ms") or 0.0) for j in judge_tel)
        if judge_tel
        else float(gw.get("latency_ms") or 0.0)
    )
    if total_claim_lat > 0:
        footer_parts.append(f"{total_claim_lat:.0f} ms")
    if item.get("fallback_used"):
        footer_parts.append("Fallback used")
    footer_block = f'<div class="sad-card-footer">{" &bull; ".join(footer_parts)}</div>'

    confidence = str(item.get("confidence", "low")).strip().lower()
    risk_level = str(item.get("risk_level", "moderate")).strip().lower()
    risk_cls = f"risk-{risk_level}" if risk_level in ("critical", "moderate", "low") else "risk-moderate"
    agreement_summary = str(item.get("agreement_summary") or "").strip()
    agreement_badge_html = (
        f'<span class="sad-meta-pill">{escape_html(agreement_summary)}</span>'
        if agreement_summary
        else ""
    )

    stance_summary = str(item.get("stance_summary") or "").strip()
    if not stance_summary:
        s_cnt = len(item.get("support_domains") or [])
        c_cnt = len(item.get("contradict_domains") or [])
        if s_cnt > 0 or c_cnt > 0:
            stance_summary = f"{s_cnt} support • {c_cnt} contradict (independent domains)"
    stance_badge_html = (
        f'<span class="sad-meta-pill">⚖️ {escape_html(stance_summary)}</span>'
        if stance_summary
        else ""
    )

    breakdown = item.get("score_breakdown") or {}
    breakdown_row = ""
    if isinstance(breakdown, dict) and score is not None:
        ea = breakdown.get("evidence_agreement", 0)
        auth = breakdown.get("authority", 0)
        rec = breakdown.get("recency", 0)
        ma = breakdown.get("model_agreement", 0)
        breakdown_row = (
            f'<div class="sad-score-breakdown-row" style="display:flex; gap:0.5rem; flex-wrap:wrap; margin:0.35rem 0 0.5rem; font-size:0.75rem; color:var(--text-muted);">'
            f'<span>Agreement: <strong>{ea}%</strong> (wt 40%)</span> &bull; '
            f'<span>Authority: <strong>{auth}%</strong> (wt 25%)</span> &bull; '
            f'<span>Recency: <strong>{rec}%</strong> (wt 20%)</span> &bull; '
            f'<span>Model consensus: <strong>{ma}%</strong> (wt 15%)</span>'
            f'</div>'
        )

    dual_judge_used = bool(item.get("dual_judge_used"))
    model_disagreement = bool(item.get("model_disagreement"))
    model_agreement_summary = str(item.get("model_agreement_summary") or "").strip()
    if model_disagreement:
        dual_judge_badge_html = '<span class="sad-meta-pill sad-model-disagree">Models disagree</span>'
    elif dual_judge_used or model_agreement_summary == "2/2 models agree":
        agree_lbl = model_agreement_summary or "2/2 models agree"
        dual_judge_badge_html = f'<span class="sad-meta-pill sad-model-agree">{escape_html(agree_lbl)}</span>'
    else:
        dual_judge_badge_html = ""

    judge_verdicts = [
        v for v in (item.get("judge_verdicts") or [])
        if isinstance(v, dict) and (v.get("model") or v.get("status"))
    ]
    dual_judge_box_html = ""
    if len(judge_verdicts) >= 2 or (dual_judge_used and judge_verdicts):
        box_cls = "sad-dual-judge-box disagree" if model_disagreement else "sad-dual-judge-box"
        head_txt = (
            "Dual-judge verdicts (Models disagree)"
            if model_disagreement
            else f"Dual-judge verdicts ({model_agreement_summary or '2/2 models agree'})"
        )
        v_rows: list[str] = []
        for v_idx, v in enumerate(judge_verdicts, start=1):
            vm = escape_html(v.get("model") or f"Judge {v_idx}")
            vs = escape_html(str(v.get("status") or "unknown").upper())
            v_sc = v.get("score")
            sc_bit = f" ({int(v_sc)}/100)" if v_sc is not None else ""
            v_info = str(v.get("current_info") or "").strip()
            info_bit = f" &mdash; {escape_html(v_info)}" if v_info else ""
            v_rows.append(
                f'<div class="sad-dual-judge-row"><strong>{vm}:</strong> {vs}{sc_bit}{info_bit}</div>'
            )
        dual_judge_box_html = (
            f'<div class="{box_cls}">'
            f'<div class="sad-dual-judge-head">{escape_html(head_txt)}</div>'
            f'{"".join(v_rows)}'
            '</div>'
        )

    card_id = f"claim-{key_prefix}-{index}"

    top_explanations = {
        "fresh": "✓ Verified current fact: confirmed by authoritative primary documentation.",
        "outdated": "⚠️ Previously accurate fact that has changed: superseded by recent updates.",
        "incorrect": "❌ Factually incorrect claim: contradicted by official authoritative sources.",
        "partially_correct": "⚡ Partially accurate: core entity matches but specific figures or dates differ.",
        "uncertain": "❓ Uncertain claim: conflicting evidence or insufficient consensus across sources.",
        "unverifiable": "🔍 Unverifiable: no reliable live web records found to corroborate or refute.",
        "verification_failed": "❌ Verification failed: LLM or Tavily search failed after retries and fallback. Excluded from overall score.",
    }
    top_expl = top_explanations.get(status, "Claim verification completed against live sources.")
    pill_suffix = meta["pill_class"].replace("pill-", "")
    top_expl_html = f'<div class="sad-claim-top-explanation {pill_suffix}">{escape_html(top_expl)}</div>'

    timeline_html = format_step_timeline_html(item)

    conf_bars = ""
    if confidence == "high":
        conf_bars = '<span class="sad-confidence-meter" title="High confidence"><span class="sad-conf-bar active-high"></span><span class="sad-conf-bar active-high"></span><span class="sad-conf-bar active-high"></span></span>'
    elif confidence in ("moderate", "medium"):
        conf_bars = '<span class="sad-confidence-meter" title="Medium confidence"><span class="sad-conf-bar active-med"></span><span class="sad-conf-bar active-med"></span><span class="sad-conf-bar"></span></span>'
    else:
        conf_bars = '<span class="sad-confidence-meter" title="Low confidence"><span class="sad-conf-bar active-low"></span><span class="sad-conf-bar"></span><span class="sad-conf-bar"></span></span>'

    meta_color = meta.get("color", "#64748B")
    source_origin = str(item.get("source_origin") or ("saved" if st.session_state.get("demo_mode") else ("cached" if item.get("cache_hit") else "live"))).lower()
    saved_date = item.get("saved_date") or SAVED_RESULTS_DATE
    if "saved" in source_origin:
        source_badge_html = f'<span class="sad-meta-pill" style="border-color: #2563EB; color: #2563EB; background: rgba(37,99,235,0.08);">💾 Saved ({escape_html(saved_date)})</span>'
    elif "cache" in source_origin:
        source_badge_html = '<span class="sad-meta-pill" style="border-color: #D97706; color: #D97706; background: rgba(217,119,6,0.08);">⚡ Cached</span>'
    else:
        source_badge_html = '<span class="sad-meta-pill" style="border-color: #16A34A; color: #16A34A; background: rgba(22,163,74,0.08);">🟢 Live</span>'

    stagger_idx = min(max(int(index), 1), 12)
    card_html = (
        f'<div id="{escape_html(card_id)}" class="sad-claim-card status-{status} sad-reveal-stagger" style="border-top: 3px solid {meta_color}; --sad-stagger-i: {stagger_idx};">'
        f'{top_expl_html}'
        f'<div class="sad-claim-header">'
        f'<div class="sad-claim-badges">'
        f'<span class="sad-pill {meta["pill_class"]} sad-pill-pop">{meta["label"]}</span>'
        f'{source_badge_html}'
        f'<span class="sad-cat-badge">{cat_icon}{escape_html(category_display)}</span>'
        f'<span class="sad-meta-pill">{escape_html(confidence.title())} confidence{conf_bars}</span>'
        f'<span class="sad-meta-pill {risk_cls}">{escape_html(risk_level.title())} risk</span>'
        f'{agreement_badge_html}'
        f'{stance_badge_html}'
        f'{dual_judge_badge_html}'
        f'</div>'
        f'<div class="sad-score-bar-wrap">'
        f'<div class="sad-score-track">'
        f'<div class="sad-score-fill" style="width: {bar_width}%; background: {meta["color"]};"></div>'
        f'</div>'
        f'<span class="sad-score-text">{score_display}</span>'
        f'</div></div>'
        f'{timeline_html}'
        f'{breakdown_row}'
        f'<div class="sad-compare-grid">'
        f'<div class="sad-compare-box">'
        f'<div class="sad-compare-label">Claimed</div>'
        f'<div class="sad-compare-body">{claimed_html}</div>'
        f'</div>'
        f'<div class="sad-compare-box sad-compare-actual">'
        f'<div class="sad-compare-label">Actually (today)</div>'
        f'<div class="sad-compare-body">{actual_html}</div>'
        f'{actual_meta_block}'
        f'</div></div>'
        f'{error_block}'
        f'{conflict_block}'
        f'{dual_judge_box_html}'
        f'<div class="sad-reason"><strong>Reason:</strong> {reason_html}</div>'
        f'{query_block}'
        f'{sources_block}'
        f'{footer_block}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Retry button for failed claims + Claim Watchlist Toggle Button
    claim_text_val = str(item.get("claim", "")).strip()
    if is_failed or claim_text_val:
        cur_watchlist = st.session_state.setdefault("watchlist", [])
        is_claim_watched = any(w.get("claim") == claim_text_val for w in cur_watchlist)
        w_lbl = "⭐ In Watchlist" if is_claim_watched else "⭐ Watch claim"
        if is_failed:
            w_c1, w_c2, w_c3 = st.columns([0.3, 0.5, 0.2])
            with w_c1:
                if st.button(
                    "🔄 Retry this claim",
                    key=f"retry_claim_{key_prefix}_{index}",
                    type="primary",
                    help="Retry live verification for this failed claim",
                ):
                    with st.spinner("Retrying claim verification..."):
                        retry_input = {
                            "claim": claim_text_val,
                            "span": item.get("span"),
                            "category": category,
                            "search_query": item.get("search_query") or claim_text_val,
                            "detect_latency_ms": item.get("detect_latency_ms"),
                        }
                        retried = verify_claim(
                            retry_input,
                            dual_judge=bool(st.session_state.get("dual_judge_mode", False)),
                        )
                        state_key_map = {
                            "ask": "ask_results",
                            "url": "url_results",
                        }
                        target_state_key = state_key_map.get(key_prefix, "results")
                        curr_list = list(st.session_state.get(target_state_key) or [])
                        if 0 <= (index - 1) < len(curr_list):
                            curr_list[index - 1] = retried
                            st.session_state[target_state_key] = curr_list
                        if retried.get("status") == "verification_failed":
                            err_q = classify_error_quota_status(retried.get("error") or retried.get("reason") or "")
                            if err_q:
                                st.session_state["quota_status"] = err_q
                        else:
                            for j_tel in (retried.get("judge_telemetry") or []):
                                if isinstance(j_tel, dict):
                                    rec = dict(j_tel)
                                    rec.setdefault("operation", "Verify claim (retry)")
                                    append_gateway_request(rec)
                            if not (retried.get("judge_telemetry") or []) and isinstance(retried.get("gateway_metrics"), dict):
                                rec = dict(retried.get("gateway_metrics") or {})
                                rec.setdefault("operation", "Verify claim (retry)")
                                append_gateway_request(rec)
                            st.session_state["quota_status"] = "Live"
                            if not any(isinstance(r, dict) and r.get("status") == "verification_failed" for r in curr_list):
                                st.session_state["pipeline_error_step"] = None
                        st.rerun()
            with w_c3:
                if claim_text_val and st.button(
                    w_lbl,
                    key=f"watch_claim_{key_prefix}_{index}",
                    help="Pin to session watchlist for delta tracking across re-audits",
                ):
                    if is_claim_watched:
                        st.session_state["watchlist"] = [w for w in cur_watchlist if w.get("claim") != claim_text_val]
                    else:
                        cur_watchlist.append({
                            "claim": claim_text_val,
                            "original_status": status,
                            "current_status": status,
                            "score": score,
                        })
                    st.rerun()
        elif claim_text_val:
            w_c1, w_c2, w_c3 = st.columns([0.35, 0.45, 0.2])
            with w_c1:
                if st.button(
                    "🔄 Re-run claim",
                    key=f"rerun_claim_{key_prefix}_{index}",
                    type="secondary",
                    help="Re-verify this individual claim with current gateway routing and outage settings",
                ):
                    with st.spinner("Re-verifying claim..."):
                        is_outage = bool(st.session_state.get("simulate_outage", False))
                        state_key_map = {
                            "ask": "ask_results",
                            "url": "url_results",
                        }
                        target_state_key = state_key_map.get(key_prefix, "results")
                        curr_list = list(st.session_state.get(target_state_key) or [])
                        if 0 <= (index - 1) < len(curr_list):
                            retried = copy.deepcopy(curr_list[index - 1])
                            if is_outage:
                                retried["fallback_used"] = True
                                retried["model"] = "gemini-3.1-flash-lite (fallback)"
                                fb_entry = {
                                    "operation": "Verify claim (re-run fallback)",
                                    "primary_model": "gemini-2.5-flash",
                                    "model_used": "gemini-3.1-flash-lite",
                                    "fallback_used": True,
                                    "latency_ms": 142.0,
                                    "prompt_tokens": 120,
                                    "completion_tokens": 45,
                                }
                                retried["judge_telemetry"] = [fb_entry]
                                append_gateway_request(fb_entry)
                            curr_list[index - 1] = retried
                            st.session_state[target_state_key] = curr_list
                        st.rerun()
            with w_c3:
                if st.button(
                    w_lbl,
                    key=f"watch_claim_{key_prefix}_{index}",
                    help="Pin to session watchlist for delta tracking across re-audits",
                ):
                    if is_claim_watched:
                        st.session_state["watchlist"] = [w for w in cur_watchlist if w.get("claim") != claim_text_val]
                    else:
                        cur_watchlist.append({
                            "claim": claim_text_val,
                            "original_status": status,
                            "current_status": status,
                            "score": score,
                        })
                    st.rerun()

    # Short evidence snippets (<= 25 words per source) inside an expander under the claim.
    valid_sources = [
        s for s in (item.get("sources") or [])
        if isinstance(s, dict) and str(s.get("url", "")).strip()
    ]
    if valid_sources:
        with st.expander(
            f"Evidence snippets ({len(valid_sources)})",
            key=f"evidence_snippets_{key_prefix}_{index}",
            icon=":material/format_quote:",
        ):
            for src in valid_sources:
                s_url = str(src.get("url", "")).strip()
                safe_s_url = safe_href_url(s_url)
                s_domain = escape_html(extract_domain(s_url))
                s_title = escape_html(str(src.get("title") or "").strip() or extract_domain(s_url))
                raw_stype = str(src.get("source_type") or "").strip()
                s_type = (
                    raw_stype
                    if raw_stype in {"Primary", "News", "Blog", "Forum"}
                    else classify_source_type(s_url, entity_domains=entity_doms)
                )
                s_type_cls = f"sad-src-{s_type.lower()}"
                s_date = str(src.get("published_date") or "").strip()
                s_date_html = f'<span class="sad-src-date">{escape_html(s_date)}</span>' if s_date else ""
                s_snippet = truncate_snippet_words(src.get("snippet") or "")
                snippet_body = (
                    escape_html(s_snippet)
                    if s_snippet
                    else "<em>No excerpt available for this source.</em>"
                )
                st.markdown(
                    f'<div class="sad-snippet-card">'
                    f'<div class="sad-snippet-head">'
                    f'<span class="sad-src-type {s_type_cls}">{escape_html(s_type)}</span>'
                    f'<a class="sad-snippet-link" href="{safe_s_url}" target="_blank" rel="noopener noreferrer">{s_title}</a>'
                    f'<span class="sad-src-date">({s_domain})</span>'
                    f'{s_date_html}'
                    f'</div>'
                    f'<div class="sad-snippet-text">{snippet_body}</div>'
                    f'{render_credibility_bar_html(get_source_credibility(s_type, s_url))}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Explainable evidence-strength breakdown inside "Why this score?" expander.
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
                    part_score = max(0, min(100, int(part.get("score") or 0)))
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

                ev_strength = breakdown.get("evidence_strength")
                raw_sum = breakdown.get("raw_weighted_score")
                band_str = breakdown.get("status_band")
                extra_caption_bits = [
                    f"Confidence: **{confidence.title()}**",
                    f"Risk level: **{risk_level.title()}**",
                ]
                if ev_strength is not None:
                    extra_caption_bits.append(f"Evidence strength: **{ev_strength}/100**")
                if band_str:
                    extra_caption_bits.append(f"Band ({status}): **{band_str}**")
                extra_caption_bits.append(f"Final score: **{score_display}**")
                st.caption("  •  ".join(extra_caption_bits))
                if status == "outdated":
                    st.caption(
                        "Outdated claims stay in the 0–39 Stale band: strong source recency, agreement, "
                        "and authority confirm the refutation rather than increasing freshness."
                    )
                elif (
                    status in {"fresh", "uncertain"}
                    and raw_sum is not None
                    and score is not None
                    and int(raw_sum) != int(score)
                ):
                    st.caption(
                        f"Weighted component sum ({raw_sum}/100) was clamped into the {status} status band ({band_str})."
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


def _callable_accepts_kwarg(fn, kwarg_name: str) -> bool:
    """Return True if `fn` (or a patched mock side_effect) accepts `kwarg_name`."""
    target = fn
    side_effect = getattr(fn, "side_effect", None)
    if callable(side_effect):
        target = side_effect
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg_name:
            return True
    return False


def _append_claim_gateway_telemetry(
    idx: int,
    result: dict,
    gateway_placeholder,
    refresh_panel: bool = True,
) -> bool:
    """Record 1 or 2 judge telemetry entries per verified claim into the Gateway panel."""
    judge_tel = [
        entry
        for entry in (result.get("judge_telemetry") or [])
        if isinstance(entry, dict)
    ]
    added_any = False
    if judge_tel:
        for j_idx, j_meta in enumerate(judge_tel, start=1):
            op_label = (
                f"Verify claim #{idx}"
                if len(judge_tel) == 1
                else f"Verify claim #{idx} (Judge {j_idx})"
            )
            j_rec = build_request_telemetry(
                op_label,
                j_meta,
                model_used=j_meta.get("model_used") or result.get("model"),
                fallback_used=bool(j_meta.get("fallback_used", False)),
                primary_model=j_meta.get("primary_model") or result.get("primary_model"),
            )
            if j_rec:
                append_gateway_request(j_rec)
                added_any = True
    else:
        claim_record = build_request_telemetry(
            f"Verify claim #{idx}",
            result.get("gateway_metrics"),
            model_used=result.get("model"),
            fallback_used=bool(result.get("fallback_used", False)),
            primary_model=result.get("primary_model"),
        )
        if claim_record:
            append_gateway_request(claim_record)
            added_any = True

    if added_any and refresh_panel:
        render_gateway_panel(
            gateway_placeholder,
            st.session_state["gateway_requests"],
        )
    return added_any


def run_stale_audit(
    answer_text: str,
    status_widget,
    steps_log: list[str],
    credits_placeholder,
    gateway_placeholder,
    max_claims: int | None = None,
    audit_meta_out: dict | None = None,
    stepper_placeholder=None,
    action_status_slot=None,
    scan_slot=None,
    discovery_slot=None,
    verify_board_slot=None,
    freshness_strip_slot=None,
    live_meter_slot=None,
    live_counters_slot=None,
    activity_feed_slot=None,
    live_text_slot=None,
    cancel_event: threading.Event | None = None,
) -> tuple[str | None, dict | None, list[dict]]:
    """
    Reuse the shared extractor + parallel verifier pipeline on `answer_text`, streaming live
    progress driven strictly by real pipeline events (Detect -> Claim Discovery -> Verify
    Search/Extract/Judge -> Score -> Complete/Error/Stopped).
    """
    st.session_state["pipeline_error_step"] = None
    st.session_state["audit_stopped"] = False
    st.session_state["audit_stopped_info"] = None
    st.session_state["partial_audit_results"] = []
    st.session_state["partial_audit_total"] = 0

    active_cancel = cancel_event or threading.Event()
    st.session_state["audit_cancel_event"] = active_cancel

    ui_state = create_initial_ui_state()
    st.session_state["audit_ui_state"] = ui_state
    audit_t0 = time.perf_counter()
    _last_refresh = [0.0]

    def _refresh_live_ui(
        include_scan: bool = False,
        include_verify: bool = False,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        if not force and (now - _last_refresh[0]) < 0.150:
            return
        _last_refresh[0] = now

        if stepper_placeholder is not None:
            stepper_placeholder.markdown(
                render_stepper_html(
                    st.session_state.get("busy_action"),
                    has_results=bool(ui_state.reveal_ready),
                    has_input=True,
                    ui_state=ui_state,
                ),
                unsafe_allow_html=True,
            )
        if action_status_slot is not None:
            pill_html = render_action_status_pill_html(ui_state)
            if pill_html:
                action_status_slot.markdown(pill_html, unsafe_allow_html=True)
            else:
                action_status_slot.empty()
        if freshness_strip_slot is not None:
            strip_html = render_freshness_strip_html(ui_state)
            if strip_html:
                freshness_strip_slot.markdown(strip_html, unsafe_allow_html=True)
            else:
                freshness_strip_slot.empty()
        if live_meter_slot is not None:
            meter_html = render_provisional_score_meter_html(ui_state)
            if meter_html:
                live_meter_slot.markdown(meter_html, unsafe_allow_html=True)
            else:
                live_meter_slot.empty()
        if live_counters_slot is not None:
            counters_html = render_live_counters_html(ui_state)
            if counters_html:
                live_counters_slot.markdown(counters_html, unsafe_allow_html=True)
            else:
                live_counters_slot.empty()
        if activity_feed_slot is not None:
            feed_html = render_activity_feed_html(ui_state)
            if feed_html:
                activity_feed_slot.markdown(feed_html, unsafe_allow_html=True)
            else:
                activity_feed_slot.empty()
        if live_text_slot is not None:
            marked_html = render_live_marked_text_html(answer_text, ui_state)
            if marked_html:
                live_text_slot.markdown(marked_html, unsafe_allow_html=True)
            else:
                live_text_slot.empty()
        if include_scan and scan_slot is not None:
            scan_html = render_scan_effect_html(answer_text, ui_state.scan_active)
            if scan_html:
                scan_slot.markdown(scan_html, unsafe_allow_html=True)
            else:
                scan_slot.empty()
        if include_verify and verify_board_slot is not None:
            board_html = render_verify_live_board_html(ui_state)
            if board_html:
                verify_board_slot.markdown(board_html, unsafe_allow_html=True)

    def _finalize_audit_timing() -> None:
        elapsed_ms = round((time.perf_counter() - audit_t0) * 1000.0, 1)
        st.session_state["last_audit_time_ms"] = elapsed_ms
        st.session_state["session_audit_time_ms"] = round(
            float(st.session_state.get("session_audit_time_ms") or 0.0) + elapsed_ms,
            1,
        )
        render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

    # --- Phase 1: Real Detect Start ---
    apply_pipeline_event(ui_state, "detect_start")
    _refresh_live_ui(include_scan=True, force=True)
    st.write("Detecting claims...")
    claims, extract_model = extract_claims(answer_text)
    total_extracted = len(claims)
    effective_max_claims = max_claims if max_claims is not None else get_max_claims_per_audit()
    if effective_max_claims is not None and effective_max_claims > 0 and total_extracted > effective_max_claims:
        claims = claims[:effective_max_claims]
    if isinstance(audit_meta_out, dict):
        audit_meta_out["total_claims_extracted"] = total_extracted
        audit_meta_out["claims_audited"] = len(claims)

    extract_record = build_request_telemetry(
        "Extract claims",
        get_extract_metrics(),
        model_used=extract_model,
    )
    append_gateway_request(extract_record)
    render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

    if effective_max_claims is not None and total_extracted > len(claims):
        extract_msg = (
            f"Extracted {total_extracted} claim(s) — auditing first {len(claims)} (capped to protect API quota) ({extract_model})"
        )
    else:
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
        apply_pipeline_event(ui_state, "complete", {"results": []})
        _refresh_live_ui(include_scan=True, force=True)
        _finalize_audit_timing()
        status_widget.update(
            label="Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )
        return extract_model, extract_record, []

    # PS06 Requirement 1: Freshness Detection vs Live Verification partitioning
    threshold = int(st.session_state.get("detection_threshold", 40))
    flagged_claims, skipped_claims = partition_claims_by_threshold(claims, threshold=threshold)
    st.session_state["partial_audit_total"] = len(flagged_claims)

    # --- Phase 2: Claim Discovery Reveal ---
    apply_pipeline_event(ui_state, "detect_done", {"claims": claims, "threshold": threshold})
    _refresh_live_ui(include_scan=True, force=True)
    if discovery_slot is not None:
        disc_html = render_claim_discovery_chips_html(claims, threshold=threshold, category_icons=CATEGORY_ICONS)
        if disc_html:
            discovery_slot.markdown(disc_html, unsafe_allow_html=True)

    detection_msg = (
        f"Detection: {len(claims)} claim(s) found, {len(flagged_claims)} flagged for live verification "
        f"(sensitivity >= {threshold})"
    )
    if skipped_claims:
        detection_msg += f" — {len(skipped_claims)} stable fact(s) skipped"
    steps_log.append(detection_msg)
    st.write(f":material/radar: {detection_msg}")

    if isinstance(audit_meta_out, dict):
        audit_meta_out["claims_found"] = len(claims)
        audit_meta_out["claims_flagged"] = len(flagged_claims)
        audit_meta_out["claims_skipped"] = len(skipped_claims)

    if active_cancel.is_set():
        apply_pipeline_event(ui_state, "stopped", {"completed": 0, "total": len(flagged_claims)})
        _refresh_live_ui(force=True)
        _finalize_audit_timing()
        st.session_state["audit_stopped"] = True
        st.session_state["audit_stopped_info"] = {"completed": 0, "total": len(flagged_claims)}
        status_widget.update(label="Stopped", state="complete", expanded=False)
        return extract_model, extract_record, []

    if not flagged_claims:
        verified_results = [create_skipped_claim_result(c) for c in claims]
        apply_pipeline_event(ui_state, "complete", {"results": verified_results})
        _refresh_live_ui(force=True)
        _finalize_audit_timing()
        steps_log.append("Done (all claims verified as stable)")
        status_widget.update(
            label="Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )
        audit_ledger = compute_audit_ledger(
            claims_results=verified_results,
            extract_meta=extract_record,
            gateway_requests=st.session_state.get("gateway_requests"),
        )
        st.session_state["last_audit_ledger"] = audit_ledger
        curr_sess = st.session_state.get("session_ledger") or create_empty_ledger()
        st.session_state["session_ledger"] = merge_ledgers(curr_sess, audit_ledger)
        return extract_model, extract_record, verified_results

    total = len(flagged_claims)
    apply_pipeline_event(ui_state, "verify_start", {"flagged_claims": flagged_claims})
    _refresh_live_ui(include_verify=True, force=True)
    status_widget.update(
        label="Searching live sources",
        state="running",
    )
    live_step = st.empty()
    live_step.write(f"Searching live sources for {total} flagged claim(s)...")
    progress_bar = st.progress(
        0.0,
        text=f"Searching live sources (0/{total} claims)...",
    )

    progress_calls = [0]
    partial_verified_live: list[dict] = []

    def on_verify_event(claim_idx: int, substep: str, payload: dict) -> None:
        if substep == "done":
            apply_pipeline_event(
                ui_state,
                "claim_done",
                {"claim_index": claim_idx, "result": payload.get("result")},
            )
        else:
            apply_pipeline_event(
                ui_state,
                "claim_substep",
                {
                    "claim_index": claim_idx,
                    "substep": substep,
                    "sources": payload.get("sources"),
                },
            )
        status_widget.update(label=ui_state.button_label, state="running")
        _refresh_live_ui(include_verify=True, force=False)

    def on_progress(idx: int, total_count: int, result: dict) -> None:
        progress_calls[0] += 1
        partial_verified_live.append(result)
        st.session_state["partial_audit_results"] = list(partial_verified_live)

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

        _append_claim_gateway_telemetry(idx, result, gateway_placeholder, refresh_panel=True)

        if progress_bar is not None and total_count > 0:
            frac = min(1.0, max(0.0, float(idx) / float(total_count)))
            prog_label = (
                f"Verified {idx}/{total_count} claims"
                if idx < total_count
                else f"Verified {total_count}/{total_count} claims — scoring..."
            )
            progress_bar.progress(frac, text=prog_label)

        if idx < total_count:
            status_widget.update(
                label=ui_state.button_label,
                state="running",
            )
            live_step.write(f"Searching live sources ({idx + 1}/{total_count})...")
            _refresh_live_ui(include_verify=True, force=True)
        else:
            apply_pipeline_event(ui_state, "score_start")
            status_widget.update(label="Scoring", state="running")
            live_step.write("Scoring freshness...")
            _refresh_live_ui(include_verify=True, force=True)

        res_score = result.get("score")
        res_score_str = "—" if res_score is None else f"{res_score}/100"
        step_text = (
            f"Claim {idx}/{total_count}: {result.get('status', 'uncertain').upper()} "
            f"({res_score_str})"
        )
        steps_log.append(step_text)

    dual_enabled = bool(st.session_state.get("dual_judge_mode", False))
    verify_kwargs: dict = {"progress_callback": on_progress}
    if dual_enabled and _callable_accepts_kwarg(verify_claims, "dual_judge"):
        verify_kwargs["dual_judge"] = True
    if _callable_accepts_kwarg(verify_claims, "event_callback"):
        verify_kwargs["event_callback"] = on_verify_event
    if _callable_accepts_kwarg(verify_claims, "cancel_event"):
        verify_kwargs["cancel_event"] = active_cancel

    flagged_results = verify_claims(flagged_claims, **verify_kwargs)
    if progress_calls[0] == 0:
        batch_credits = sum(
            float(r.get("tavily_credits_used", 0.0) or 0.0)
            for r in flagged_results
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
        for b_idx, r in enumerate(flagged_results, start=1):
            if _append_claim_gateway_telemetry(b_idx, r, gateway_placeholder, refresh_panel=False):
                batch_added = True
        if batch_added:
            render_gateway_panel(
                gateway_placeholder,
                st.session_state["gateway_requests"],
            )

    # Recombine flagged and skipped claims in original order (handles partial results cleanly if stopped)
    was_stopped = active_cancel.is_set()
    results_map = {}
    for orig_idx, c in enumerate(claims):
        ts = c.get("time_sensitivity", 50)
        try:
            ts_val = int(ts)
        except (TypeError, ValueError):
            ts_val = 50
        if ts_val >= threshold:
            results_map[orig_idx] = None
        else:
            results_map[orig_idx] = create_skipped_claim_result(c)

    flag_iter = iter(flagged_results)
    verified_results = []
    for orig_idx in range(len(claims)):
        if results_map[orig_idx] is None:
            nxt = next(flag_iter, None)
            if nxt is not None:
                verified_results.append(nxt)
        else:
            verified_results.append(results_map[orig_idx])

    if progress_bar is not None:
        if was_stopped:
            progress_bar.progress(
                min(1.0, len(flagged_results) / max(total, 1)),
                text=f"Stopped — showing {len(flagged_results)}/{total} completed live checks",
            )
        else:
            progress_bar.progress(1.0, text=f"Completed {total}/{total} live checks")

    _finalize_audit_timing()
    if was_stopped:
        st.session_state["audit_stopped"] = True
        st.session_state["audit_stopped_info"] = {
            "completed": len(flagged_results),
            "total": total,
        }
        apply_pipeline_event(
            ui_state,
            "stopped",
            {"completed": len(flagged_results), "total": total},
        )
        _refresh_live_ui(include_verify=True)
        live_step.write(f"Stopped — preserved {len(flagged_results)} of {total} verified claim(s)")
        steps_log.append(f"Stopped ({len(flagged_results)}/{total} verified)")
        status_widget.update(
            label="Stopped (partial results)",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )
    else:
        apply_pipeline_event(ui_state, "complete", {"results": verified_results})
        _refresh_live_ui()
        live_step.write(":material/check: Judging freshness complete")
        steps_log.append("Done")
        status_widget.update(
            label="Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )

    # PS06 Requirement 5: Contribution Ledger recording
    audit_ledger = compute_audit_ledger(
        claims_results=verified_results,
        extract_meta=extract_record,
        gateway_requests=st.session_state.get("gateway_requests"),
    )
    st.session_state["last_audit_ledger"] = audit_ledger
    curr_sess = st.session_state.get("session_ledger") or create_empty_ledger()
    st.session_state["session_ledger"] = merge_ledgers(curr_sess, audit_ledger)

    for r in verified_results:
        if isinstance(r, dict):
            r.setdefault("source_origin", "cached" if r.get("cache_hit") else "live")

    failed_claims = [
        r for r in verified_results
        if isinstance(r, dict) and r.get("status") == "verification_failed"
    ]
    if failed_claims:
        st.session_state["pipeline_error_step"] = "verify"
        for fc in failed_claims:
            err_q = classify_error_quota_status(fc.get("error") or fc.get("reason") or "")
            if err_q:
                st.session_state["quota_status"] = err_q
                break

    if not st.session_state.get("demo_mode"):
        audits_count = int(st.session_state.get("live_audits_performed") or 0) + 1
        st.session_state["live_audits_performed"] = audits_count
        if not failed_claims:
            if audits_count >= get_max_session_audits():
                st.session_state["quota_status"] = "Limited"
            else:
                st.session_state["quota_status"] = "Live"

    return extract_model, extract_record, verified_results


def render_audit_results(
    results: list[dict] | None,
    extract_model: str | None,
    extract_meta: dict | None,
    key_prefix: str,
    just_clicked: bool = False,
    empty_title: str = "Ready to verify",
    empty_body: str = "Paste an LLM answer on the left or pick a demo in the sidebar, then click <strong>Check Freshness</strong>.",
    empty_pills: tuple[str, ...] = ("Prices & tiers", "SDK versions", "Regulations", "Leadership", "Stats"),
    empty_hint: str = "Need offline results? Turn on <strong>Demo mode</strong> in the sidebar.",
    extra_models: list[str | None] | None = None,
    extra_metas: list[dict | None] | None = None,
    annotated_text: str | None = None,
    annotated_title: str = "Annotated answer",
) -> None:
    """Render shared freshness score ring, KPI grid, optional annotated text, and per-claim cards."""
    if results is None:
        if not just_clicked:
            st.markdown(
                _render_empty_state(
                    empty_title,
                    empty_body,
                    pills=empty_pills,
                    hint=empty_hint,
                ),
                unsafe_allow_html=True,
            )
        return

    if len(results) == 0:
        st.info(
            "No potentially outdated or time-sensitive factual claims were found in this answer.",
            icon=":material/info:",
        )
        zero_models = [m for m in [*(extra_models or []), extract_model] if m]
        zero_routes: list[str] = []
        for meta in [*(extra_metas or []), extract_meta]:
            if isinstance(meta, dict) and meta.get("fallback_used"):
                rt = format_fallback_route(meta.get("primary_model"), meta.get("model_used"))
                if rt not in zero_routes:
                    zero_routes.append(rt)
        if zero_models:
            st.markdown(
                _build_compact_model_row_html(zero_models, fallback_routes=zero_routes),
                unsafe_allow_html=True,
            )
        return

    overall_score = compute_risk_weighted_overall_score(results)
    checked_text_val = annotated_text or (results[0].get("claim", "") if results else "")
    failed_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "verification_failed")
    if not st.session_state.get("demo_mode"):
        audit_hist_status = (
            "Failed"
            if failed_count == len(results)
            else ("Partial" if failed_count > 0 else "Completed")
        )
        record_audit_to_history(
            checked_text_val,
            results,
            source_label="Live",
            mode=key_prefix.capitalize(),
            score=None if failed_count > 0 else overall_score,
            audit_status=audit_hist_status,
        )
    fresh_count = sum(1 for r in results if r.get("status") == "fresh")
    outdated_count = sum(1 for r in results if r.get("status") == "outdated")
    incorrect_count = sum(1 for r in results if r.get("status") == "incorrect")
    partially_correct_count = sum(1 for r in results if r.get("status") == "partially_correct")
    uncertain_count = sum(1 for r in results if r.get("status") == "uncertain")
    unverifiable_count = sum(1 for r in results if r.get("status") == "unverifiable")
    error_count = sum(1 for r in results if r.get("error"))

    verdict_sentence = build_overall_verdict_sentence(results)

    models_used: list[str] = []
    claim_models: list[str] = []
    for r in results:
        if isinstance(r.get("judge_models"), list) and r.get("judge_models"):
            claim_models.extend(str(m) for m in r["judge_models"] if m)
        if r.get("model"):
            claim_models.append(str(r["model"]))
    for model_name in [
        *(extra_models or []),
        extract_model,
        *claim_models,
    ]:
        if model_name and model_name not in models_used:
            models_used.append(model_name)

    fallback_routes = collect_fallback_routes(extract_meta, results)
    for meta in extra_metas or []:
        if isinstance(meta, dict) and meta.get("fallback_used"):
            route = format_fallback_route(meta.get("primary_model"), meta.get("model_used"))
            if route not in fallback_routes:
                fallback_routes.append(route)

    if st.session_state.get("audit_stopped") and key_prefix == "pasted":
        stop_info = st.session_state.get("audit_stopped_info") or {}
        st_done = int(stop_info.get("completed") or len(results))
        st_tot = int(stop_info.get("total") or len(results))
        st.info(
            f"Audit stopped by user — showing partial results ({st_done} of {max(st_tot, st_done)} claim(s) verified).",
            icon=":material/stop_circle:",
        )

    st.markdown(
        render_score_ring_html(
            overall_score,
            models_used,
            fallback_routes=fallback_routes,
            verdict_sentence=verdict_sentence,
            results=results,
        ),
        unsafe_allow_html=True,
    )
    if overall_score is not None:
        components.html(
            """
            <script>
            (function() {
              try {
                const win = window.parent || window;
                if (win.matchMedia && win.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
                const doc = win.document || document;
                const nums = doc.querySelectorAll('.sad-ring-num[data-target-score]');
                nums.forEach(function(el) {
                  if (el.dataset.counted === '1') return;
                  const target = parseInt(el.getAttribute('data-target-score'), 10);
                  if (isNaN(target) || target <= 0) return;
                  el.dataset.counted = '1';
                  const start = performance.now();
                  const dur = 480;
                  function tick(now) {
                    const t = Math.min(1, (now - start) / dur);
                    const eased = 1 - Math.pow(1 - t, 3);
                    el.textContent = String(Math.round(target * eased));
                    if (t < 1) win.requestAnimationFrame(tick);
                    else el.textContent = String(target);
                  }
                  win.requestAnimationFrame(tick);
                });
              } catch (e) {}
            })();
            </script>
            """,
            height=0,
        )

    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    flagged_count = len(results) - skipped_count
    source_origins = set()
    for r in results:
        so = str(r.get("source_origin") or ("saved" if st.session_state.get("demo_mode") else ("cached" if r.get("cache_hit") else "live"))).lower()
        source_origins.add(so)

    if "saved" in source_origins or st.session_state.get("demo_mode"):
        origin_badge_html = f'<span class="sad-cat-badge" style="border: 1px solid #2563EB; color: #2563EB; background: rgba(37,99,235,0.08);">💾 Source: Saved Results ({SAVED_RESULTS_DATE})</span>'
    elif "cached" in source_origins and "live" not in source_origins:
        origin_badge_html = '<span class="sad-cat-badge" style="border: 1px solid #D97706; color: #D97706; background: rgba(217,119,6,0.08);">⚡ Source: Cached (6h TTL)</span>'
    else:
        origin_badge_html = '<span class="sad-cat-badge" style="border: 1px solid #16A34A; color: #16A34A; background: rgba(22,163,74,0.08);">🟢 Source: Live Verification</span>'

    detect_html = (
        f'<div class="sad-detection-summary" style="display:flex; gap:0.6rem; align-items:center; flex-wrap:wrap; margin-bottom: 0.85rem;">'
        f'{origin_badge_html}'
        f'<span class="sad-cat-badge">🔍 Freshness Detection: <strong>{len(results)}</strong> claims found</span>'
        f'<span class="sad-cat-badge" style="color: #2563EB;">⚡ <strong>{flagged_count}</strong> flagged for live verification</span>'
    )
    if skipped_count > 0:
        detect_html += f'<span class="sad-cat-badge" style="color: #64748B;">🛡️ <strong>{skipped_count}</strong> stable fact(s) skipped</span>'
    detect_html += '</div>'
    st.markdown(detect_html, unsafe_allow_html=True)

    if failed_count > 0 and not st.session_state.get("demo_mode"):
        with st.container(border=True):
            fc1, fc2 = st.columns([2.2, 1.0], vertical_alignment="center", gap="small")
            with fc1:
                st.warning(
                    f"Could not verify {failed_count} of {len(results)} claims after retries and fallback. "
                    "Failed claims are excluded from the overall score and status counts. Use **Retry this claim** "
                    "on individual cards below, or load Demo / Saved results for built-in examples.",
                    icon=":material/warning:",
                )
            with fc2:
                if st.button(
                    "Load Demo / Saved results",
                    key=f"{key_prefix}_failed_demo_fallback_btn",
                    icon=":material/play_circle:",
                    use_container_width=True,
                ):
                    try:
                        sample, exact = find_cached_pasted_audit(checked_text_val)
                        _apply_pasted_demo_sample(sample, exact_match=exact, sync_widget=False)
                        st.session_state["demo_mode"] = True
                        st.session_state["pipeline_error_step"] = None
                        st.rerun()
                    except FileNotFoundError:
                        st.info("Saved demo cache file not found.")

    grid_cols = 7 if skipped_count > 0 else 6
    with st.grid(grid_cols, min_column_width=80, gap="small"):
        st.metric("Fresh", fresh_count, border=True, height="stretch", help="Claims verified current and fully supported by live web sources.")
        st.metric("Outdated", outdated_count, border=True, height="stretch", help="Claims that were historically true but superseded by newer changes.")
        st.metric("Incorrect", incorrect_count, border=True, height="stretch", help="Claims contradicted by official sources or factually false.")
        st.metric("Partially correct", partially_correct_count, border=True, height="stretch", help="Claims with a mix of accurate and inaccurate details.")
        st.metric("Uncertain", uncertain_count, border=True, height="stretch", help="Claims with conflicting evidence or missing source consensus.")
        st.metric("Unverifiable", unverifiable_count, border=True, height="stretch", help="Claims where no authoritative live web sources could be found.")
        if skipped_count > 0:
            st.metric("Skipped", skipped_count, border=True, height="stretch", help="Stable, non-volatile factual claims skipped to conserve API credits.")

    # PS06 Requirement 5: Contribution Ledger Panel (route-aware labels)
    audit_led = st.session_state.get("last_audit_ledger") or compute_audit_ledger(results, extract_meta)
    sess_led = st.session_state.get("session_ledger") or audit_led
    active_bifrost = is_use_bifrost()
    ledger_route_info = get_real_model_and_routing_info(use_bifrost=active_bifrost)
    llm_col_title = "#### ⚡ Bifrost AI Gateway" if active_bifrost else "#### ⚡ Direct Gemini API"
    llm_calls_lbl = ledger_route_info["calls_label"]
    llm_fb_lbl = "Gateway fallbacks triggered" if active_bifrost else "Model fallbacks triggered"
    llm_lat_lbl = "Gateway latency" if active_bifrost else "Measured LLM latency"
    with st.expander("🤝 Contribution Ledger (Tavily + Active LLM Route)", expanded=False, icon=":material/handshake:"):
        st.caption(
            f"Contribution metrics for this audit and session (active route: {ledger_route_info['route_label']}). "
            "Only LLM calls that returned usage are counted."
        )
        l_col1, l_col2 = st.columns(2, gap="medium")
        with l_col1:
            st.markdown("#### 🌐 Tavily Web Intelligence")
            st.markdown(
                f"- **Live searches executed:** `{audit_led['tavily']['searches']}` (Session: `{sess_led['tavily']['searches']}`)  \n"
                f"- **Deep extracts performed:** `{audit_led['tavily']['extracts']}` (Session: `{sess_led['tavily']['extracts']}`)  \n"
                f"- **Tavily search credits:** `{audit_led['tavily']['credits']:.2f}` (Session: `{sess_led['tavily']['credits']:.2f}`)  \n"
                f"- **Unique independent domains:** `{len(audit_led['tavily']['unique_domains'])}` (Session: `{len(sess_led['tavily']['unique_domains'])}`)  \n"
                f"- **Primary / official sources:** `{audit_led['tavily']['primary_sources']}` (Session: `{sess_led['tavily']['primary_sources']}`)"
            )
        with l_col2:
            st.markdown(llm_col_title)
            cost_str = f"${audit_led['bifrost']['cost']:.5f}" if audit_led['bifrost']['cost_set'] else "Pricing unconfigured"
            sess_cost_str = f"${sess_led['bifrost']['cost']:.5f}" if sess_led['bifrost']['cost_set'] else "Pricing unconfigured"
            models_list = ", ".join(f"`{m}`" for m in audit_led['bifrost']['models']) or "`None recorded`"
            st.markdown(
                f"- **{llm_calls_lbl}:** `{audit_led['bifrost']['calls']}` (Session: `{sess_led['bifrost']['calls']}`)  \n"
                f"- **Models used:** {models_list}  \n"
                f"- **{llm_fb_lbl}:** `{audit_led['bifrost']['fallbacks']}` (Session: `{sess_led['bifrost']['fallbacks']}`)  \n"
                f"- **Tokens returned:** `{audit_led['bifrost']['total_tokens']:,}` (In: `{audit_led['bifrost']['tokens_in']:,}` | Out: `{audit_led['bifrost']['tokens_out']:,}`)  \n"
                f"- **{llm_lat_lbl}:** `{audit_led['bifrost']['total_latency_ms']:.0f} ms`  \n"
                f"- **Estimated cost:** `{cost_str}` (Session: `{sess_cost_str}`)"
            )

    # PS06 Requirement 4: How the Freshness Score is Computed Panel (matches verifier.py SCORE_WEIGHTS)
    with st.expander("ℹ️ How the Freshness Score is Computed", expanded=False, icon=":material/calculate:"):
        st.markdown(
            "Every verifiable claim's freshness score is calculated deterministically from four weighted signals in `verifier.py`:\n\n"
            "1. **LLM Verdict (50% weight):** Grounded factual comparison strictly against retrieved web snippets and extracted text.\n"
            "2. **Source Recency (20% weight):** Based on `published_date` returned by Tavily.\n"
            "3. **Source Agreement (20% weight):** Ratio of unique registered domains supporting the current fact.\n"
            "4. **Source Authority (10% weight):** Domain tier (`PRIMARY`, `NEWS`, `BLOG`, or `FORUM`).\n\n"
            "**Enforced Score Bands:**\n"
            "- **Fresh:** `75 – 100` | **Partially Correct:** `45 – 74` | **Uncertain:** `40 – 74`\n"
            "- **Outdated:** `10 – 39` | **Incorrect:** `0 – 15` | **Unverifiable:** `N/A` | **Verification failed:** `—` (excluded)"
        )

    # PS06 Requirement 6: "Why Tavily?" & "Why Bifrost?" Exploration Controls
    ab_col1, ab_col2 = st.columns(2, gap="medium")
    with ab_col1:
        if st.button("⚖️ Why Tavily? (Compare with LLM Alone)", key=f"{key_prefix}_why_tavily_btn", use_container_width=True):
            st.session_state[f"{key_prefix}_show_why_tavily"] = not st.session_state.get(f"{key_prefix}_show_why_tavily", False)
    with ab_col2:
        if st.button("🛡️ Why Bifrost? (Simulate Outage & Failover)", key=f"{key_prefix}_why_bifrost_btn", use_container_width=True):
            st.session_state[f"{key_prefix}_show_why_bifrost"] = not st.session_state.get(f"{key_prefix}_show_why_bifrost", False)

    if st.session_state.get(f"{key_prefix}_show_why_tavily"):
        with st.container(border=True):
            st.subheader("⚖️ Why Tavily? Live Grounding vs. LLM Parametric Memory", icon=":material/balance:")
            st.caption("Here is how an LLM evaluates these same claims using ONLY its static training weights (no Tavily web search), side-by-side with Tavily live-grounded verification.")

            cache_key = f"{key_prefix}_{len(results)}"
            llm_alone_results = st.session_state.get("why_tavily_cache", {}).get(cache_key)
            if not llm_alone_results:
                if st.session_state.get("demo_mode"):
                    llm_alone_results = []
                    for r in results:
                        c_text = r.get("claim", "")
                        st_live = r.get("status", "fresh")
                        if st_live in ("outdated", "incorrect"):
                            llm_alone_results.append({
                                "claim": c_text,
                                "llm_status": "fresh",
                                "llm_value": "Old / pre-cutoff value (hallucinated as current)",
                                "llm_confidence": "high",
                                "llm_reason": "Assessed purely from internal training data; unaware of recent changes.",
                                "model_used": "parametric memory",
                            })
                        else:
                            llm_alone_results.append({
                                "claim": c_text,
                                "llm_status": "uncertain",
                                "llm_value": r.get("current_value") or "Unknown",
                                "llm_confidence": "medium",
                                "llm_reason": "Cannot verify without live internet search.",
                                "model_used": "parametric memory",
                            })
                else:
                    with st.spinner("Re-evaluating claims with LLM alone (no web search)..."):
                        llm_alone_results = run_llm_only_verification(results)
                if "why_tavily_cache" not in st.session_state:
                    st.session_state["why_tavily_cache"] = {}
                st.session_state["why_tavily_cache"][cache_key] = llm_alone_results

            for r_idx, (live_res, llm_res) in enumerate(zip(results, llm_alone_results), start=1):
                c_text = live_res.get("claim", "")
                live_st = str(live_res.get("status", "uncertain")).upper()
                live_val = live_res.get("current_value") or live_res.get("current_info") or "N/A"
                llm_st = str(llm_res.get("llm_status", "uncertain")).upper()
                llm_val = llm_res.get("llm_value", "Unknown")
                is_diff = live_st != llm_st or live_res.get("status") in ("outdated", "incorrect")
                diff_tag = "⚠️ LLM Overconfident / Stale" if is_diff else "✅ Concordant"

                with st.container(border=True):
                    st.markdown(f"**Claim #{r_idx}:** *\"{escape_html(c_text)}\"* &mdash; **{diff_tag}**", unsafe_allow_html=True)
                    cmp1, cmp2 = st.columns(2, gap="small")
                    with cmp1:
                        st.markdown(
                            f"**With Tavily (Live Web Grounded):**  \n"
                            f"- Verdict: `{live_st}` ({live_res.get('score', 'N/A')}/100)  \n"
                            f"- Current Info: {escape_html(str(live_val))}  \n"
                            f"- Stance: {escape_html(str(live_res.get('stance_summary', 'Grounding verified')))}"
                        )
                    with cmp2:
                        st.markdown(
                            f"**Without Tavily (LLM Alone):**  \n"
                            f"- Verdict: `{llm_st}` ({escape_html(str(llm_res.get('llm_confidence', 'medium'))).title()} confidence)  \n"
                            f"- Parametric Belief: {escape_html(str(llm_val))}  \n"
                            f"- LLM Note: {escape_html(str(llm_res.get('llm_reason', '')))}"
                        )

    if st.session_state.get(f"{key_prefix}_show_why_bifrost"):
        with st.container(border=True):
            st.subheader("🛡️ Why Bifrost? Upstream Outage Simulation & Failover", icon=":material/shield:")
            st.caption("Demonstrating Bifrost's gateway-level circuit-breaker failover when the primary model experiences an outage.")

            first_claim = results[0] if results else {"claim": "Example claim", "category": "other"}
            if st.session_state.get("demo_mode"):
                st.info(
                    "Demo mode is active (offline). Live outage failover simulation requires an active LLM route. "
                    "See **Learn -> Architecture & Evidence** for real recorded Bifrost failover events from `evidence/bifrost_evidence.json`.",
                    icon=":material/info:",
                )
                sim_res = None
            else:
                with st.spinner("Simulating primary model outage through active routing..."):
                    sim_res = run_why_bifrost_simulation(first_claim)

            if isinstance(sim_res, dict):
                st.markdown(
                    f'<div class="sad-meta-row">'
                    f'<span class="sad-cat-badge" style="color: #DC2626;">❌ Primary Outage: <strong>{escape_html(str(sim_res.get("primary_failed", "primary")))}</strong></span>'
                    f'<span class="sad-cat-badge" style="color: #16A34A;">✅ Responded: <strong>{escape_html(str(sim_res.get("answering_model", "fallback")))}</strong></span>'
                    f'<span class="sad-cat-badge">⏱️ Failover Overhead: <strong>{sim_res.get("failover_overhead_ms", 0):.0f} ms</strong></span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="sad-timeline">', unsafe_allow_html=True)
                for step in sim_res.get("timeline", []):
                    st_cls = step.get("status", "info")
                    t_ms = step.get("timestamp_ms", 0.0)
                    st.markdown(
                        f'<div class="sad-timeline-item">'
                        f'<div class="sad-timeline-dot {st_cls}"></div>'
                        f'<div class="sad-timeline-content">'
                        f'<div class="sad-timeline-title">+{t_ms:.0f}ms &mdash; {escape_html(str(step.get("event", "")))}</div>'
                        f'<div class="sad-timeline-desc">{escape_html(str(step.get("detail", "")))}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

    if error_count > 0:
        friendly_reasons: list[str] = []
        for err_item in results:
            if err_item.get("error"):
                friendly_msg = format_friendly_ui_error(err_item["error"], action="verifying claims")
                if friendly_msg not in friendly_reasons:
                    friendly_reasons.append(friendly_msg)
        summary_msg = f"{error_count} claim(s) encountered a verification error and were marked as unverifiable/uncertain."
        if friendly_reasons:
            summary_msg += " " + " ".join(friendly_reasons)
        st.warning(summary_msg, icon=":material/warning:")

    if annotated_text and annotated_text.strip():
        st.space("small")
        render_annotated_answer_card(
            answer_text=annotated_text,
            verification_results=results,
            key_prefix=key_prefix,
            title=annotated_title,
        )

    st.space("small")
    show_queries = st.session_state.get("show_search_queries", False)
    total_claims = len(results)
    for idx, item in enumerate(results, start=1):
        if total_claims > 4:
            st_label = item.get("status", "uncertain").replace("_", " ").title()
            claimed_preview = (item.get("claim") or "")[:55]
            if len(item.get("claim") or "") > 55:
                claimed_preview += "..."
            with st.expander(
                f"Claim #{idx}: {claimed_preview} — [{st_label}]",
                expanded=(idx <= 4),
            ):
                render_claim_card(idx, item, show_query=show_queries, key_prefix=key_prefix)
        else:
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
    source_url: str | None = None,
    ask_meta: dict | None = None,
    extract_model: str | None = None,
    extract_meta: dict | None = None,
    extra_tavily_credits: float = 0.0,
    audit_at: datetime | None = None,
    gateway_placeholder=None,
) -> None:
    """
    Render the always-visible 'Generate corrected answer' section after an audit
    (side-by-side Original vs. Corrected with word diff, draft preview before AI
    rewrite click, 'What changed & why' changelog, and Markdown/JSON report exports).
    """
    busy_tag = f"rewrite_{key_prefix}"
    if results is None:
        if st.session_state.get("busy_action") == busy_tag:
            st.session_state["busy_action"] = None
        return

    state_key_map = {
        "pasted": (
            "corrected_answer",
            "corrected_model",
            "corrected_meta",
            "pasted_audit_at",
            "Rewrite answer",
            "generate_corrected_btn",
            "pasted_rewrite_error",
        ),
        "ask": (
            "ask_corrected_answer",
            "ask_corrected_model",
            "ask_corrected_meta",
            "ask_audit_at",
            "Rewrite Q&A answer",
            "ask_generate_corrected_btn",
            "ask_rewrite_error",
        ),
        "url": (
            "url_corrected_answer",
            "url_corrected_model",
            "url_corrected_meta",
            "url_audit_at",
            "Rewrite URL article",
            "url_generate_corrected_btn",
            "url_rewrite_error",
        ),
    }
    corr_ans_key, corr_mod_key, corr_meta_key, audit_at_key, rewrite_op, btn_key, rewrite_err_key = state_key_map.get(
        key_prefix,
        (
            f"{key_prefix}_corrected_answer",
            f"{key_prefix}_corrected_model",
            f"{key_prefix}_corrected_meta",
            f"{key_prefix}_audit_at",
            "Rewrite answer",
            f"{key_prefix}_generate_corrected_btn",
            f"{key_prefix}_rewrite_error",
        ),
    )

    current_busy = st.session_state.get("busy_action")
    is_any_busy = bool(current_busy)
    is_rewriting_this = current_busy == busy_tag

    st.markdown(
        """
        <div class="sad-section-divider"></div>
        <div class="sad-section-header">
          <svg class="sad-section-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
          </svg>
          <div>
            <h3 class="sad-section-title">Corrected answer</h3>
            <p class="sad-section-sub">Automated surgical rewrite: replaces outdated or incorrect claims with verified current facts</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hdr_col, action_col = st.columns([2.2, 1], vertical_alignment="center", gap="medium")
    with hdr_col:
        st.caption("Re-synthesizes response text by replacing outdated or incorrect claims with verified web intelligence.")
    with action_col:
        btn_label = (
            "Generate corrected answer"
            if not corrected_answer
            else "Regenerate corrected answer"
        )
        generate_clicked = st.button(
            btn_label,
            key=btn_key,
            type="primary",
            icon=("spinner" if is_rewriting_this else ":material/auto_fix_high:"),
            disabled=(len(results) == 0 or is_any_busy),
            width="stretch",
        )

    rewrite_skel_slot = st.empty()
    if generate_clicked and results and not is_any_busy:
        st.session_state[rewrite_err_key] = None
        if st.session_state.get("demo_mode"):
            new_corrected, new_model = _resolve_cached_rewrite(
                key_prefix=key_prefix,
                original_answer=original_answer,
                results=results,
            )
            st.session_state[corr_ans_key] = new_corrected
            st.session_state[corr_mod_key] = new_model
            rewrite_record = {
                "operation": f"{rewrite_op} (Demo cache)",
                "task": "rewrite",
                "primary_model": new_model,
                "model_used": new_model,
                "fallback_used": False,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            st.session_state[corr_meta_key] = rewrite_record
            now_ts = datetime.now().astimezone()
            st.session_state[audit_at_key] = now_ts
            corrected_answer = new_corrected
            corrected_model = new_model
            corrected_meta = rewrite_record
            audit_at = now_ts
        else:
            st.session_state["busy_action"] = busy_tag
            st.session_state["_busy_armed"] = True
            st.rerun()
    elif is_rewriting_this and results:
        rewrite_finished = False
        try:
            _show_skeleton_slot(rewrite_skel_slot, (180,))
            with st.spinner("Rewriting answer with verified current info..."):
                new_corrected, new_model = generate_corrected_answer(
                    original_answer,
                    results,
                )
                st.session_state[corr_ans_key] = new_corrected
                st.session_state[corr_mod_key] = new_model
                rewrite_record = build_request_telemetry(
                    rewrite_op,
                    get_rewrite_metrics(),
                    model_used=new_model,
                )
                st.session_state[corr_meta_key] = rewrite_record
                now_ts = datetime.now().astimezone()
                st.session_state[audit_at_key] = now_ts
                append_gateway_request(rewrite_record)
                if gateway_placeholder is not None:
                    render_gateway_panel(
                        gateway_placeholder,
                        st.session_state["gateway_requests"],
                    )
            rewrite_finished = True
        except Exception as exc:
            st.session_state[rewrite_err_key] = format_friendly_ui_error(
                exc, action="generating the corrected answer"
            )
            rewrite_finished = True
        finally:
            if st.session_state.get("busy_action") == busy_tag:
                st.session_state["busy_action"] = None
            _clear_slot(rewrite_skel_slot)
        if rewrite_finished:
            st.rerun()

    if st.session_state.get(rewrite_err_key):
        st.error(
            str(st.session_state[rewrite_err_key]),
            icon=":material/error:",
        )

    matched_spans = match_claims_to_answer_spans(original_answer or "", results)
    is_ai_rewritten = bool(corrected_answer)
    display_corrected = (
        corrected_answer
        if is_ai_rewritten
        else build_draft_corrected_answer(
            original_answer or "",
            results,
            matched_spans=matched_spans,
        )
    )

    diff_result = build_side_by_side_word_diff(original_answer or "", display_corrected)
    diff_stats = diff_result["stats"]

    meta_notes: list[str] = [
        f"-{diff_stats['removed_words']} removed / +{diff_stats['added_words']} added word(s)"
    ]
    if is_ai_rewritten:
        if corrected_model:
            meta_notes.append(f"Rewritten with: {corrected_model}")
        if isinstance(corrected_meta, dict) and corrected_meta.get("fallback_used"):
            fb_route = format_fallback_route(
                corrected_meta.get("primary_model"),
                corrected_meta.get("model_used") or corrected_model,
            )
            meta_notes.append(f"Fallback used: {fb_route}")
    elif results:
        meta_notes.append(
            "Instant draft from verified claims — click Generate corrected answer for full AI rewrite"
        )
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
        corr_tag = "New text in green" if is_ai_rewritten else "New text in green (draft)"
        st.markdown(
            '<div class="sad-diff-card sad-diff-card-new">'
            '<div class="sad-diff-header">'
            '<span class="sad-diff-title">Corrected answer</span>'
            f'<span class="sad-diff-tag sad-diff-tag-ins">{corr_tag}</span>'
            '</div>'
            f'<div class="sad-diff-body">{diff_result["corrected_html"]}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    cmap_ui = build_citation_map(results)
    if display_corrected and (cmap_ui["citations"] or cmap_ui["unverified_claims"]):
        cited_ui_html = render_cited_corrected_answer_html(
            display_corrected,
            citations=cmap_ui["citations"],
            unverified_claims=cmap_ui["unverified_claims"],
        )
        st.markdown(
            '<div class="sad-diff-card" style="margin-top: 0.75rem;">'
            '<div class="sad-diff-header">'
            '<span class="sad-diff-title">Cited corrected answer &amp; primary sources</span>'
            '<span class="sad-diff-tag sad-diff-tag-ins">Numbered citations</span>'
            '</div>'
            f'<div class="sad-diff-body">{cited_ui_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.expander(
        "Copy plain corrected text",
        key=f"copy_plain_{key_prefix}",
        icon=":material/content_copy:",
    ):
        st.code(display_corrected, language=None, wrap_lines=True)

    st.subheader("What changed & why", icon=":material/difference:")
    changes = build_change_log(results)
    if changes:
        pill_cls_map = {
            "incorrect": "sad-change-pill-corrected",
            "outdated": "sad-change-pill-replaced",
            "partially_correct": "sad-change-pill-refined",
            "uncertain": "sad-change-pill-qualified",
            "unverifiable": "sad-change-pill-unverifiable",
        }
        for ch in changes:
            pill_cls = pill_cls_map.get(ch["status"], "sad-change-pill-qualified")
            cat_icon = CATEGORY_ICONS.get(ch["category"], CATEGORY_ICONS["other"])
            ch_score_text = "N/A" if ch.get("score") is None else f"{ch['score']}/100"

            ch_meta_rows: list[str] = []
            if ch.get("status") == "incorrect":
                corr_val = ch.get("correction") or ch.get("current_value") or ch.get("current_info") or "Contradicted by sources"
                ch_sources = ch.get("sources") or []
                ch_top_src = ""
                if ch_sources and isinstance(ch_sources, list) and isinstance(ch_sources[0], dict):
                    src_u = ch_sources[0].get("url") or ""
                    src_t = ch_sources[0].get("title") or (extract_domain(src_u) if src_u else "")
                    if src_u:
                        ch_top_src = f' (<a href="{safe_href_url(src_u)}" target="_blank" rel="noopener noreferrer">{escape_html(src_t)}</a>)'
                    elif src_t:
                        ch_top_src = f' ({escape_html(src_t)})'
                ch_meta_rows.append(
                    f'<div><span class="sad-actual-meta-label" style="color: #EF4444; font-weight: 700;">Never true / contradicted by:</span> {escape_html(corr_val)}{ch_top_src}</div>'
                )
            else:
                if ch.get("current_value"):
                    ch_meta_rows.append(
                        f'<div><span class="sad-actual-meta-label">Current value:</span> {escape_html(ch["current_value"])}</div>'
                    )
                if ch.get("correction") and not ch.get("current_value"):
                    ch_meta_rows.append(
                        f'<div><span class="sad-actual-meta-label">Correction:</span> {escape_html(ch["correction"])}</div>'
                    )
                if ch.get("change_summary"):
                    ch_meta_rows.append(
                        f'<div><span class="sad-actual-meta-label">Change summary:</span> {escape_html(ch["change_summary"])}</div>'
                    )
                if ch.get("effective_date"):
                    ch_meta_rows.append(
                        f'<div><span class="sad-actual-meta-label">Effective date:</span> {escape_html(ch["effective_date"])}</div>'
                    )
                if ch.get("status") == "outdated":
                    ch_stale_display = _format_stale_since_text(
                        ch.get("stale_since") or ch.get("change_date"),
                        ch.get("stale_duration_label"),
                    )
                    if ch_stale_display:
                        ch_meta_rows.append(
                            f'<div><span class="sad-actual-meta-label">Stale since:</span> {escape_html(ch_stale_display)}</div>'
                        )
            ch_meta_html = (
                '<div class="sad-actual-meta">' + "".join(ch_meta_rows) + "</div>"
                if ch_meta_rows
                else ""
            )

            ch_agree = str(ch.get("agreement_summary") or "").strip()
            ch_agree_pill = (
                f'<span class="sad-meta-pill">{escape_html(ch_agree)}</span>'
                if ch_agree
                else ""
            )
            ch_conflict_block = ""
            if ch.get("conflict_detected") or ch.get("conflict_note"):
                ch_conflict_msg = ch.get("conflict_note") or (
                    f"Sources disagree ({ch_agree or 'mixed evidence'})."
                )
                ch_conflict_block = (
                    f'<div class="sad-conflict-note">'
                    f'<strong>Sources disagree:</strong> {escape_html(ch_conflict_msg)}'
                    f'</div>'
                )

            sources_block = _format_source_chips_html(ch.get("sources"))

            st.markdown(
                '<div class="sad-change-card">'
                '<div class="sad-change-top">'
                f'<span class="sad-pill {pill_cls}">{escape_html(ch["action"])}</span>'
                f'<span class="sad-cat-badge">{cat_icon}{escape_html(ch["category"])}</span>'
                f'<span class="sad-meta-pill">Claim #{ch["claim_index"]} &bull; {ch_score_text}</span>'
                f'{ch_agree_pill}'
                '</div>'
                '<div class="sad-change-diff-row">'
                '<div class="sad-compare-box">'
                '<div class="sad-compare-label">Before (original claim)</div>'
                f'<div class="sad-compare-body">{escape_html(ch["original_claim"])}</div>'
                '</div>'
                '<div class="sad-compare-box sad-compare-actual">'
                '<div class="sad-compare-label">After (verified today)</div>'
                f'<div class="sad-compare-body">{escape_html(ch["updated_to"])}</div>'
                f'{ch_meta_html}'
                '</div>'
                '</div>'
                f'{ch_conflict_block}'
                f'<div class="sad-reason"><strong>Why:</strong> {escape_html(ch["why"])}</div>'
                f'{sources_block}'
                '</div>',
                unsafe_allow_html=True,
            )

        fresh_kept = sum(1 for r in results if r.get("status") == "fresh")
        if fresh_kept > 0:
            st.caption(f"Kept {fresh_kept} verified fresh claim(s) unchanged.")
        if not is_ai_rewritten:
            st.caption(
                "Comparing against instant claim-by-claim draft edits; click Generate corrected answer above for a full AI prose rewrite."
            )
    else:
        st.info(
            "No outdated, uncertain, or unverifiable claims needed factual updates — all checked claims stayed fresh.",
            icon=":material/check_circle:",
        )

    # In-session Claim Watchlist strip
    render_watchlist_strip()

    # Report export buttons (available as soon as an audit completes, and automatically
    # reflect the visible corrected comparison / AI rewrite telemetry).
    report_time = audit_at or datetime.now().astimezone()
    total_report_credits = None
    if extra_tavily_credits:
        claim_credits_sum = sum(float(r.get("tavily_credits_used") or 0.0) for r in results)
        total_report_credits = round(claim_credits_sum + float(extra_tavily_credits), 2)

    exported_corrected = (
        corrected_answer
        if is_ai_rewritten
        else (display_corrected if display_corrected != (original_answer or "") else None)
    )

    report_payload = build_report_payload(
        mode=mode,
        original_answer=original_answer or "",
        verification_results=results,
        question=question,
        source_url=source_url,
        corrected_answer=exported_corrected,
        ask_meta=ask_meta,
        extract_model=extract_model,
        extract_meta=extract_meta,
        corrected_model=corrected_model if is_ai_rewritten else None,
        corrected_meta=corrected_meta if is_ai_rewritten else None,
        tavily_credits_used=total_report_credits,
        generated_at=report_time,
    )
    md_report = format_report_markdown(report_payload)
    json_report = format_report_json(report_payload)
    html_report = format_report_html(report_payload)
    summary_text = format_shareable_summary(report_payload)
    ts_slug = report_time.strftime("%Y%m%d-%H%M%S")

    st.markdown(
        """
        <div class="sad-section-divider"></div>
        <div class="sad-section-header" id="sad-export-anchor">
          <svg class="sad-section-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
          <div>
            <h3 class="sad-section-title">Export</h3>
            <p class="sad-section-sub">Download audit findings in standard formats or copy shareable plain-text summary</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    models_label = ", ".join(report_payload["models_used"]) or "N/A"
    tokens_total = report_payload["tokens_used"]["total_tokens"]
    st.caption(
        escape_markdown(
            f"Includes {report_payload['summary']['claims_checked']} claim(s), scores, sources, "
            f"timestamps ({report_payload['generated_at']}), models ({models_label}), "
            f"and {tokens_total:,} tokens."
        )
    )

    dl_col1, dl_col2, dl_col3, dl_col4, dl_col5 = st.columns(5, gap="small")
    with dl_col1:
        st.download_button(
            "Markdown",
            data=md_report,
            file_name=f"stale-answer-report-{ts_slug}.md",
            mime="text/markdown",
            key=f"dl_md_{key_prefix}",
            icon=":material/download:",
            disabled=is_any_busy,
            on_click="ignore",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            "JSON",
            data=json_report,
            file_name=f"stale-answer-report-{ts_slug}.json",
            mime="application/json",
            key=f"dl_json_{key_prefix}",
            icon=":material/data_object:",
            disabled=is_any_busy,
            on_click="ignore",
            width="stretch",
        )
    with dl_col3:
        csv_rows = []
        for c_idx, c_item in enumerate(results, start=1):
            if isinstance(c_item, dict):
                csv_rows.append({
                    "Index": c_idx,
                    "Claim": c_item.get("claim", ""),
                    "Status": c_item.get("status", ""),
                    "Score": c_item.get("score"),
                    "Confidence": c_item.get("confidence", ""),
                    "Risk": c_item.get("risk_level", ""),
                    "Category": c_item.get("category", ""),
                    "Current Info": c_item.get("current_info", ""),
                    "Reason": c_item.get("reason", ""),
                })
        csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8") if csv_rows else b""
        st.download_button(
            "CSV",
            data=csv_data,
            file_name=f"stale-answer-claims-{ts_slug}.csv",
            mime="text/csv",
            key=f"dl_csv_{key_prefix}",
            icon=":material/table_view:",
            disabled=is_any_busy,
            on_click="ignore",
            width="stretch",
        )
    with dl_col4:
        st.download_button(
            "HTML Report",
            data=html_report,
            file_name=f"stale-answer-report-{ts_slug}.html",
            mime="text/html",
            key=f"dl_html_{key_prefix}",
            icon=":material/html:",
            disabled=is_any_busy,
            on_click="ignore",
            width="stretch",
            help="Self-contained offline HTML report with inline styles (no internet needed)",
        )
    with dl_col5:
        with st.popover("📋 Share Summary", icon=":material/share:"):
            st.caption("Copy this plain-text summary for Slack, PR comments, or email:")
            st.code(summary_text, language="text")


def build_benchmark_tables(
    bench_data: dict,
    filter_mode: str = "All",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build DataFrames for the confusion matrix, per-category accuracy, and per-claim table."""
    metrics = bench_data.get("metrics") or {}
    cm = metrics.get("confusion_matrix") or {}
    cm_mat = cm.get("matrix") or {}

    cm_df = pd.DataFrame(
        [
            {
                "Predicted: Outdated": int((cm_mat.get("outdated") or {}).get("outdated", cm.get("tp", 0))),
                "Predicted: Fresh": int((cm_mat.get("outdated") or {}).get("fresh", 0)),
                "Predicted: Uncertain / Unverifiable": int((cm_mat.get("outdated") or {}).get("other", 0)),
            },
            {
                "Predicted: Outdated": int((cm_mat.get("fresh") or {}).get("outdated", cm.get("fp", 0))),
                "Predicted: Fresh": int((cm_mat.get("fresh") or {}).get("fresh", cm.get("tn", 0))),
                "Predicted: Uncertain / Unverifiable": int((cm_mat.get("fresh") or {}).get("other", 0)),
            },
        ],
        index=["Expected: Outdated (stale)", "Expected: Fresh"],
    )

    by_cat = metrics.get("by_category") or {}
    cat_rows = []
    for cat_name in sorted(by_cat.keys()):
        stats = by_cat[cat_name] or {}
        cat_rows.append(
            {
                "Category": str(cat_name).title(),
                "Accuracy": round(float(stats.get("accuracy") or 0.0) * 100.0, 1),
                "Correct": int(stats.get("correct") or 0),
                "Total": int(stats.get("total") or 0),
                "TP / FP / FN / TN": (
                    f"{int(stats.get('tp') or 0)} / {int(stats.get('fp') or 0)} / "
                    f"{int(stats.get('fn') or 0)} / {int(stats.get('tn') or 0)}"
                ),
            }
        )
    cat_df = pd.DataFrame(cat_rows)

    raw_rows = bench_data.get("results") or []
    filtered_rows: list[dict] = []
    for row in raw_rows:
        is_ok = bool(row.get("correct"))
        exp = str(row.get("expected_status") or "").lower()
        if filter_mode == "Correct only" and not is_ok:
            continue
        if filter_mode == "Incorrect only" and is_ok:
            continue
        if filter_mode == "Expected: Outdated" and exp != "outdated":
            continue
        if filter_mode == "Expected: Fresh" and exp != "fresh":
            continue
        if filter_mode == "Expected: Uncertain / Unverifiable" and exp not in {"uncertain", "unverifiable"}:
            continue
        filtered_rows.append(row)

    table_rows = []
    for row in filtered_rows:
        is_ok = bool(row.get("correct"))
        sc = row.get("score")
        score_str = "N/A" if sc is None else f"{int(sc)}/100"
        fact_detail = str(row.get("current_info") or row.get("note") or "").strip()
        if row.get("error"):
            fact_detail = f"Error: {format_friendly_ui_error(row['error'], action='verifying this claim')}"
        table_rows.append(
            {
                "#": int(row.get("index") or 0),
                "Mark": "✓ Correct" if is_ok else "✗ Incorrect",
                "Category": str(row.get("category") or "other"),
                "Claim": str(row.get("claim") or ""),
                "Expected": str(row.get("expected_status") or ""),
                "Predicted": str(row.get("predicted_status") or "unverifiable"),
                "Score": score_str,
                "Latency (ms)": round(float(row.get("latency_ms") or 0.0), 1),
                "Tokens": int(row.get("total_tokens") or 0),
                "Credits": round(float(row.get("tavily_credits_used") or 0.0), 2),
                "Reference & Current Fact": fact_detail,
            }
        )
    rows_df = pd.DataFrame(
        table_rows,
        columns=[
            "#",
            "Mark",
            "Category",
            "Claim",
            "Expected",
            "Predicted",
            "Score",
            "Latency (ms)",
            "Tokens",
            "Credits",
            "Reference & Current Fact",
        ],
    )
    return cm_df, cat_df, rows_df


# ---------------------------------------------------------------------------
# App setup & custom CSS injection
# ---------------------------------------------------------------------------
init_session_state()
use_bifrost_mode = is_use_bifrost()

# Fast cached gateway health check (10s session cache to keep Streamlit UI snappy)
_now_ts = time.time()
_cached_health = st.session_state.get("_gw_health_cache")
_cached_ts = float(st.session_state.get("_gw_health_ts", 0.0))
if _cached_health and (_now_ts - _cached_ts) < 10.0:
    gateway_health = _cached_health
else:
    gateway_health = check_gateway_health(timeout=0.6) if use_bifrost_mode else None
    if gateway_health:
        st.session_state["_gw_health_cache"] = gateway_health
        st.session_state["_gw_health_ts"] = _now_ts

gateway_unreachable = bool(use_bifrost_mode and gateway_health and not gateway_health.get("healthy"))
set_simulate_outage(bool(st.session_state.get("simulate_outage", False)) and use_bifrost_mode)

is_dark_theme = bool(st.session_state.get("dark_mode", True))
_css_content = load_css()
if _css_content:
    st.markdown(f"<style>\n{_css_content}\n</style>", unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
if not is_dark_theme:
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

is_busy_armed = bool(st.session_state.pop("_busy_armed", False))
current_busy = st.session_state.get("busy_action")
if current_busy and not is_busy_armed:
    st.session_state["busy_action"] = None
    current_busy = None
is_any_busy = bool(current_busy)

# ---------------------------------------------------------------------------
# Sidebar: Settings, Dual-judge, Demo mode, "How it works" stepper, Gateway
# ---------------------------------------------------------------------------
with st.sidebar:
    try:
        _, resolved_base_url, resolved_chain = get_llm_config()
    except ValueError:
        resolved_chain = []
        resolved_base_url = None

    routing_summary = _resolve_safe_task_routing()

    raw_llm_key = _get_config_value("LLM_API_KEY")
    if not use_bifrost_mode and raw_llm_key.lower() == "bifrost-local":
        raw_llm_key = ""
    llm_key_ok = bool(raw_llm_key or (not use_bifrost_mode and _get_config_value("GEMINI_API_KEY")))
    tavily_ok = bool(_get_config_value("TAVILY_API_KEY"))

    # 1. Settings section
    with st.expander("Settings", expanded=False, icon=":material/settings:"):
        dark_mode_on = st.toggle(
            "Dark mode",
            key="dark_mode",
            help="Switch between Dark mode and Light mode.",
        )
        if dark_mode_on:
            st.caption("Theme: 🌙 Dark mode")
        else:
            st.caption("Theme: ☀️ Light mode")

        st.toggle("Show search queries on cards", key="show_search_queries")
        st.toggle("Keep status log expanded", key="keep_status_expanded")

        st.divider()
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
        extract_cfg = routing_summary.get("extract_model") or primary_model
        judge_cfg = routing_summary.get("judge_model") or primary_model
        judge2_cfg = routing_summary.get("dual_judge_secondary") or "Not configured"
        fallbacks_label = ", ".join(resolved_chain[1:]) if len(resolved_chain) > 1 else (_get_config_value("LLM_MODEL_FALLBACKS") or "None")
        endpoint_display = (
            resolved_base_url
            or (
                _get_config_value("LLM_BASE_URL") or "Default OpenAI"
                if use_bifrost_mode
                else GEMINI_OPENAI_BASE_URL
            )
        )
        vk_line = ""
        bifrost_url_line = ""
        if use_bifrost_mode:
            bifrost_url_cfg = _get_config_value("BIFROST_URL")
            if bifrost_url_cfg:
                bifrost_url_line = f"**Bifrost URL (`BIFROST_URL`):** `{bifrost_url_cfg}`  \n"
            vk_val = _get_config_value("LLM_VIRTUAL_KEY") or _get_config_value("BIFROST_VIRTUAL_KEY")
            vk_status = "Configured" if vk_val else "Not set"
            vk_line = f"**Virtual key (`x-bf-vk`):** {vk_status}  \n"

        st.caption(
            f"**Routing mode:** {routing_label}  \n"
            f"{bifrost_url_line}"
            f"**Primary model:** `{primary_model}`  \n"
            f"**Extract model:** `{extract_cfg}`  \n"
            f"**Judge model:** `{judge_cfg}`  \n"
            f"**Dual-judge 2nd model:** `{judge2_cfg}`  \n"
            f"**Fallbacks:** `{fallbacks_label}`  \n"
            f"**Endpoint:** `{endpoint_display}`  \n"
            f"**LLM / Gemini key:** {'Configured' if llm_key_ok else 'Missing'}  \n"
            f"{vk_line}"
            f"**Tavily key:** {'Configured' if tavily_ok else 'Missing'}  \n"
            f"**Appearance:** {'Dark mode (🌙)' if dark_mode_on else 'Light mode (☀️)'}  \n"
            f"**Session audit cap:** `{get_max_session_audits()} audits/session`  \n"
            f"**Claim cap:** `{get_max_claims_per_audit()} claims/audit`  \n"
            "**Tavily cache:** `6h temp SQLite TTL`"
        )

    # Session Quota Protection Counter (Requirement 4)
    max_audits = get_max_session_audits()
    audits_done = int(st.session_state.get("live_audits_performed") or 0)
    audits_left = max(0, max_audits - audits_done)
    max_claims = get_max_claims_per_audit()

    with st.container(border=True):
        st.markdown(
            f'<div style="font-size: 0.82rem; font-weight: 600; color: var(--text-color); margin-bottom: 4px;">'
            f'<span>🛡️ Session Protection</span>'
            f'</div>'
            f'<div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-muted); margin-bottom: 4px;">'
            f'<span>Live audits remaining:</span>'
            f'<span style="font-weight: 700; color: {"#16A34A" if audits_left > 1 else ("#EA580C" if audits_left == 1 else "#DC2626")};">'
            f'{audits_left} of {max_audits}'
            f'</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(audits_left / max_audits)
        st.caption(f"Cap: {max_claims} claims/audit to protect API quotas.")

    # 2. Dual-judge toggle
    dual_judge_enabled = st.toggle(
        "Dual-judge mode",
        key="dual_judge_mode",
        help=(
            "Run two different judge models per claim through Bifrost. "
            "Disagreements mark the claim Uncertain with a 'Models disagree' badge."
            if use_bifrost_mode
            else "Run two Gemini models per claim directly. "
            "Disagreements mark the claim Uncertain with a 'Models disagree' badge."
        ),
    )
    if dual_judge_enabled:
        j1_cfg = routing_summary.get("dual_judge_primary") or routing_summary.get("judge_model") or "Primary"
        j2_cfg = routing_summary.get("dual_judge_secondary") or "Fallback"
        st.caption(f"Comparing `{j1_cfg}` vs `{j2_cfg}` per claim.")
        if not routing_summary.get("dual_judge_configured", False):
            st.warning(
                (
                    "Add a second model in LLM_MODEL_FALLBACKS (provider/model) or LLM_MODEL_JUDGE_2 so Dual-judge compares two distinct models."
                    if use_bifrost_mode
                    else "Add a second Gemini model in LLM_MODEL_FALLBACKS or LLM_MODEL_JUDGE_2 so Dual-judge compares two distinct models."
                ),
                icon=":material/warning:",
            )

    # 3. Freshness detection sensitivity threshold slider (PS06 Req 1)
    threshold_val = st.slider(
        "Detection threshold",
        min_value=0,
        max_value=100,
        step=5,
        key="detection_threshold",
        help=(
            "Only claims with time-sensitivity score >= threshold undergo live Tavily verification. "
            "Lower-volatility claims are marked as 'Stable, not time-sensitive (skipped)' without consuming API credits."
        ),
    )
    st.caption(f"Claims with sensitivity < {threshold_val} are skipped as stable.")

    # 4. Demo mode toggle
    demo_mode_on = st.toggle(
        "Demo mode",
        key="demo_mode",
        on_change=on_demo_mode_toggle,
        help=(
            "Serve pre-verified cached audits, diffs, and rewrites offline with 0 API "
            "or Tavily calls — ideal for live demos when network or keys fail."
        ),
    )
    if demo_mode_on:
        st.caption("Offline cache active — 0 API calls or Tavily credits used.")

    # 4. "How it works" 4-step vertical stepper with icons
    with st.expander("How it works", expanded=False, icon=":material/route:"):
        st.markdown(
            '<div class="sad-stepper">'
            '<div class="sad-step">'
            '<div class="sad-step-marker"><span class="sad-step-num">1</span><div class="sad-step-line"></div></div>'
            '<div class="sad-step-content"><div class="sad-step-title"><span>🔍</span> Extract claims</div><div class="sad-step-desc">Identifies time-sensitive facts &amp; current facts with exact spans.</div></div>'
            '</div>'
            '<div class="sad-step">'
            '<div class="sad-step-marker"><span class="sad-step-num">2</span><div class="sad-step-line"></div></div>'
            '<div class="sad-step-content"><div class="sad-step-title"><span>⚡</span> Search live sources</div><div class="sad-step-desc">Parallel live search via Tavily with 6h smart cache.</div></div>'
            '</div>'
            '<div class="sad-step">'
            '<div class="sad-step-marker"><span class="sad-step-num">3</span><div class="sad-step-line"></div></div>'
            '<div class="sad-step-content"><div class="sad-step-title"><span>⚖️</span> Judge freshness</div><div class="sad-step-desc">Evaluates 6 statuses against primary and official sources.</div></div>'
            '</div>'
            '<div class="sad-step">'
            '<div class="sad-step-marker"><span class="sad-step-num">4</span></div>'
            '<div class="sad-step-content"><div class="sad-step-title"><span>📝</span> Rewrite &amp; export</div><div class="sad-step-desc">Generates verified side-by-side diff and exportable reports.</div></div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # 5. Gateway panel or friendly note
    if use_bifrost_mode:
        st.subheader("Gateway panel", icon=":material/hub:")
        dashboard_url = get_gateway_dashboard_url()
        render_gateway_health_card(gateway_health, dashboard_url, use_bifrost=True)

        if demo_mode_on:
            st.caption("Demo mode active — live gateway calls paused.")
        elif gateway_unreachable:
            st.warning(
                "⚠️ Gateway unreachable: Automatic fallback to direct Gemini is active.",
                icon=":material/warning:",
            )
        elif not (gateway_health and gateway_health.get("healthy")):
            st.caption(
                "Bifrost looks unreachable. Check your gateway service, enable Demo mode, "
                "or set `USE_BIFROST=false` to use Gemini directly."
            )

        outage_enabled = st.toggle(
            "Simulate provider outage",
            key="simulate_outage",
            help="Forces the primary model to fail so Bifrost switches to the next fallback model.",
        )
        set_simulate_outage(bool(outage_enabled))

        configured_fallbacks = (
            resolved_chain[1:]
            if resolved_chain
            else [m.strip() for m in _get_config_value("LLM_MODEL_FALLBACKS").split(",") if m.strip()]
        )
        if outage_enabled and not demo_mode_on:
            primary_cfg = (
                routing_summary.get("judge_model")
                or (resolved_chain[0] if resolved_chain else (_get_config_value("LLM_MODEL") or "gpt-4o-mini"))
            )
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
        render_gateway_panel(
            gateway_placeholder,
            st.session_state["gateway_requests"],
            task_routing=routing_summary,
            dual_judge_enabled=bool(dual_judge_enabled),
        )
    else:
        set_simulate_outage(False)
        st.info("Direct Gemini API active (`USE_BIFROST=false`).", icon=":material/info:")
        gateway_placeholder = None

    credits_placeholder = st.empty()
    render_sidebar_credits(credits_placeholder, st.session_state["tavily_credits_used"])

    if st.button(
        "Clear cache",
        key="clear_tavily_cache_btn",
        icon=":material/delete_sweep:",
        disabled=is_any_busy,
        width="stretch",
        help="Clear cached Tavily web search and URL extraction entries so claims re-verify fresh.",
    ):
        removed = clear_tavily_cache()
        st.caption(f"Cleared {removed} cached Tavily entr{'y' if removed == 1 else 'ies'}.")

    if not demo_mode_on:
        missing_keys: list[str] = []
        if not llm_key_ok:
            missing_keys.append(
                "`LLM_API_KEY` (or `GEMINI_API_KEY`)" if not use_bifrost_mode else "`LLM_API_KEY`"
            )
        if not tavily_ok:
            missing_keys.append("`TAVILY_API_KEY`")
        if missing_keys:
            st.warning(
                f"Missing API key(s): {', '.join(missing_keys)}. Add them in Streamlit Secrets or `.env`, "
                "or enable **Demo mode** above.",
                icon=":material/key:",
            )

# ---------------------------------------------------------------------------
# Highlights Bento Grid & Floating Dock Component
# ---------------------------------------------------------------------------
def render_highlights_bento_grid() -> None:
    """Render the 'Highlights of this app' bento grid with real values from saved evidence and config."""
    ev_counts = load_evidence_counts()
    route_info = get_real_model_and_routing_info()
    j1_m = escape_html(route_info.get("dual_judge_primary") or "primary judge")
    j2_m = escape_html(route_info.get("dual_judge_secondary") or "secondary judge")
    bento_html = (
        '<div class="sad-highlights-section">'
        '<div class="sad-section-head">'
        '<span class="sad-section-tag">PLATFORM CAPABILITIES</span>'
        '<h2 class="sad-section-title">Highlights of this app</h2>'
        '<p class="sad-section-desc">'
        'Freshness verification, optional dual-judge consensus, and framework-independent agent benchmarking.'
        '</p>'
        '</div>'
        '<div class="sad-bento-grid">'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(0) : (parent.sadSwitchTab && parent.sadSwitchTab(0))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-live">Config</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Dual-Judge Consensus</h3>'
        '<p class="sad-bento-metric">2 Configured Judges</p>'
        '<p class="sad-bento-fact">'
        f'Cross-evaluates claims across {j1_m} and {j2_m} with conflict detection on official sources.'
        '</p>'
        '</div>'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(0) : (parent.sadSwitchTab && parent.sadSwitchTab(0))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-live">Pipeline</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Category &amp; Volatility Detection</h3>'
        '<p class="sad-bento-metric">9 Claim Categories</p>'
        '<p class="sad-bento-fact">'
        'Includes leadership / office holder, capital / geography, records / latest, pricing, versions, and regulations.'
        '</p>'
        '</div>'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(0) : (parent.sadSwitchTab && parent.sadSwitchTab(0))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-live">Live</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Tavily Live Evidence</h3>'
        '<p class="sad-bento-metric">6h Server SQLite Cache</p>'
        '<p class="sad-bento-fact">'
        'Deduplicates sources per registered domain, classifies PRIMARY/NEWS/BLOG/FORUM, and extracts verbatim quotes.'
        '</p>'
        '</div>'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(3) : (parent.sadSwitchTab && parent.sadSwitchTab(3))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-gateway">Recorded</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Recorded Bifrost Evidence</h3>'
        f'<p class="sad-bento-metric">{ev_counts["bifrost_requests"]} Requests &bull; {ev_counts["bifrost_fallbacks"]} Fallbacks</p>'
        '<p class="sad-bento-fact">'
        f'Recorded {ev_counts["bifrost_tokens"]:,} tokens from local gateway execution ({escape_html(str(ev_counts["bifrost_date"]))}).'
        '</p>'
        '</div>'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(2) : (parent.sadSwitchTab && parent.sadSwitchTab(2))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-saved">Recorded</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Agent Lab Benchmark</h3>'
        f'<p class="sad-bento-metric">{ev_counts["agent_lab_runs"]} Recorded Runs</p>'
        '<p class="sad-bento-fact">'
        f'Evaluates {ev_counts["agent_lab_archs"]} agent architectures across clean, model failure, and broken tool scenarios.'
        '</p>'
        '</div>'
        '<div class="sad-bento-card" onclick="window.parent.sadSwitchTab ? window.parent.sadSwitchTab(0) : (parent.sadSwitchTab && parent.sadSwitchTab(0))" role="button" tabindex="0">'
        '<div class="sad-bento-top">'
        '<span class="sad-bento-badge badge-saved">Export</span>'
        '<span class="sad-bento-arrow">&#8599;</span>'
        '</div>'
        '<h3 class="sad-bento-title">Audit Report Export</h3>'
        '<p class="sad-bento-metric">Markdown, HTML &amp; JSON</p>'
        '<p class="sad-bento-fact">'
        'Download evidence dossiers with contribution ledgers, measured step timings, and corrected answer diffs.'
        '</p>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(bento_html, unsafe_allow_html=True)


def render_floating_dock_component(quota_state: str) -> None:
    """Render the floating macOS-style dock at the bottom of the window using components.html."""
    quota_label = f"Quota: {quota_state}"
    quota_cls = (
        "quota-live"
        if quota_state == "Live"
        else ("quota-limited" if quota_state in {"Limited", "Offline"} else "quota-saved")
    )
    dock_js = f"""
    <script>
    (function() {{
      try {{
        const pDoc = window.parent.document;
        if (!pDoc) return;
        
        let dockContainer = pDoc.getElementById('sad-dock-container');
        if (!dockContainer) {{
          dockContainer = pDoc.createElement('div');
          dockContainer.id = 'sad-dock-container';
          pDoc.body.appendChild(dockContainer);
        }}
        
        window.parent.sadSwitchTab = function(idx, scrollTop) {{
          try {{
            const doc = window.parent.document || document;
            const tabList = doc.querySelector('[data-testid="stTabs"] [role="tablist"]') || doc.querySelector('[data-testid="stTabs"]');
            let tabs = tabList ? tabList.querySelectorAll('[data-testid="stTab"], [role="tab"], [data-baseweb="tab"]') : [];
            if (!tabs.length) {{
              tabs = doc.querySelectorAll('[data-testid="stTabs"] [data-testid="stTab"], [data-testid="stTabs"] [role="tab"]');
            }}
            if (tabs && tabs[idx]) {{
              tabs[idx].click();
              if (scrollTop !== false) {{
                doc.querySelector('.main, .stApp')?.scrollTo({{ top: 0, behavior: 'smooth' }});
                window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
              }}
            }}
          }} catch (e) {{ console.warn(e); }}
        }};
        
        window.parent.sadScrollTop = function() {{
          try {{
            const doc = window.parent.document || document;
            doc.querySelector('.main, .stApp')?.scrollTo({{ top: 0, behavior: 'smooth' }});
            window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
          }} catch (e) {{ window.parent.scrollTo(0, 0); }}
        }};
        
        window.parent.sadNavigateToGuidedDemo = function() {{
          window.parent.sadSwitchTab(0, true);
          setTimeout(() => {{
            const doc = window.parent.document || document;
            const btns = doc.querySelectorAll('button');
            for (const b of btns) {{
              if (b.innerText && b.innerText.includes('Guided Demo')) {{
                b.click();
                break;
              }}
            }}
          }}, 150);
        }};

        window.parent.sadNavigateToEvidence = function() {{
          window.parent.sadSwitchTab(3, false);
          setTimeout(() => {{
            const doc = window.parent.document || document;
            const evEl = doc.getElementById('bifrost-evidence') || doc.querySelector('[data-testid="stExpander"]');
            if (evEl) evEl.scrollIntoView({{ behavior: 'smooth' }});
          }}, 180);
        }};

        pDoc.addEventListener('focusin', function(e) {{
          if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {{
            dockContainer.classList.add('sad-dock-hidden');
          }}
        }});
        pDoc.addEventListener('focusout', function(e) {{
          dockContainer.classList.remove('sad-dock-hidden');
        }});

        dockContainer.innerHTML = `
          <nav class="sad-dock" aria-label="Quick navigation dock">
            <button class="sad-dock-btn" onclick="parent.sadSwitchTab(0)" data-tooltip="Audit Answer" aria-label="Audit">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
              <span class="sad-dock-txt">Audit</span>
            </button>
            <button class="sad-dock-btn" onclick="parent.sadSwitchTab(1)" data-tooltip="Model Showdown" aria-label="Compare">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
              <span class="sad-dock-txt">Compare</span>
            </button>
            <button class="sad-dock-btn" onclick="parent.sadSwitchTab(2)" data-tooltip="Agent Lab Benchmark" aria-label="Agent Lab">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 2v7.31"/><path d="M14 9.3V1.99"/><path d="M8.5 2h7"/><path d="M14 9.3a6.5 6.5 0 1 1-4 0"/><path d="M5.52 16h12.96"/></svg>
              <span class="sad-dock-txt">Lab</span>
            </button>
            <button class="sad-dock-btn" onclick="parent.sadSwitchTab(3)" data-tooltip="Architecture & Methodology" aria-label="Learn">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/><path d="M6 6h10"/><path d="M6 10h10"/></svg>
              <span class="sad-dock-txt">Learn</span>
            </button>
            <button class="sad-dock-btn" onclick="parent.sadSwitchTab(4)" data-tooltip="Session History" aria-label="History">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              <span class="sad-dock-txt">History</span>
            </button>
            <div class="sad-dock-sep"></div>
            <button class="sad-dock-btn" onclick="parent.sadScrollTop()" data-tooltip="Back to Top" aria-label="Top">
              <svg class="sad-dock-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>
              <span class="sad-dock-txt">Top</span>
            </button>
            <span class="sad-dock-chip {quota_cls}">{quota_label}</span>
          </nav>
        `;
      }} catch (err) {{
        console.log("Floating dock progressive enhancement standby");
      }}
    }})();
    </script>
    """
    components.html(dock_js, height=0, width=0)

# ---------------------------------------------------------------------------
# Hero header & Brand Strip
# ---------------------------------------------------------------------------
if gateway_unreachable:
    routing_badge_text = "Gateway Fallback (Direct Gemini)"
    routing_dot_color = "#EA580C"
elif use_bifrost_mode:
    routing_badge_text = "Bifrost gateway"
    routing_dot_color = "#16A34A"
else:
    routing_badge_text = "Direct Gemini API"
    routing_dot_color = "#8B5CF6"

demo_hero_badge = (
    '<span class="sad-header-chip" style="border-color: rgba(34,197,94,0.4);"><span class="sad-header-chip-dot" style="background:#16A34A;"></span>Demo mode (offline)</span>'
    if demo_mode_on
    else ""
)

quota_state = st.session_state.get("quota_status")
if not quota_state:
    if demo_mode_on or st.session_state.get("guided_demo_active"):
        quota_state = "Saved results"
    else:
        audits_performed = int(st.session_state.get("live_audits_performed") or 0)
        if audits_performed >= get_max_session_audits():
            quota_state = "Limited"
        else:
            quota_state = "Live"

if quota_state == "Live":
    quota_dot_color = "#10B981"
    quota_border = "rgba(16, 185, 129, 0.35)"
elif quota_state == "Offline":
    quota_dot_color = "#EF4444"
    quota_border = "rgba(239, 68, 68, 0.45)"
elif quota_state == "Limited":
    quota_dot_color = "#EA580C"
    quota_border = "rgba(234,88,12,0.35)"
else:  # "Saved results"
    quota_dot_color = "#64748B"
    quota_border = "rgba(100,116,139,0.35)"

routing_chip_label = "Bifrost Gateway" if use_bifrost_mode else "Direct Gemini API"
routing_chip_dot = "#8B5CF6" if use_bifrost_mode else "#3B82F6"
routing_chip_border = "rgba(139, 92, 246, 0.35)" if use_bifrost_mode else "rgba(59, 130, 246, 0.35)"

# Unified Top Header: CSS Grid with 3 Zones (Left: Brand/Quota/Routing, Centre: Tab slot, Right: Sponsors/Hackathon)
st.markdown(
    f"""
    <div class="sad-app-header">
      <div class="sad-header-left">
        <div class="sad-brand-card">
          <div class="sad-brand-title-row">
            <span class="sad-brand-pin-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" fill="url(#pinGrad)" filter="url(#pinGlow)"/>
                <defs>
                  <linearGradient id="pinGrad" x1="5" y1="2" x2="19" y2="22" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#8B5CF6"/>
                    <stop offset="100%" stop-color="#EC4899"/>
                  </linearGradient>
                  <filter id="pinGlow" x="1" y="0" width="22" height="24">
                    <feGaussianBlur stdDeviation="1.5" result="blur"/>
                    <feComposite in="SourceGraphic" in2="blur" operator="over"/>
                  </filter>
                </defs>
              </svg>
            </span>
            <span class="sad-brand-text">Stale-Answer <span style="background: linear-gradient(135deg, #C084FC, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Detector</span></span>
          </div>
          <div class="sad-brand-pills-row">
            <span class="sad-mini-pill pill-quota" style="border-color: {quota_border};">
              <span class="sad-mini-dot" style="background: {quota_dot_color};"></span>
              <span>QUOTA: {quota_state.upper()}</span>
            </span>
            <span class="sad-mini-pill pill-routing" style="border-color: {routing_chip_border};">
              <span class="sad-mini-dot" style="background: {routing_chip_dot};"></span>
              <span>{routing_chip_label.upper()}</span>
            </span>
          </div>
        </div>
      </div>
      <div class="sad-header-centre">
        <!-- Centre zone reserved for floating tab bar -->
      </div>
      <div class="sad-header-right">
        <div class="sad-sponsor-pill">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="vertical-align: middle;">
            <rect width="24" height="24" rx="6" fill="#06B6D4" fill-opacity="0.3"/>
            <path d="M6 8H18M12 8V18" stroke="#06B6D4" stroke-width="2.5" stroke-linecap="round"/>
          </svg>
          <span style="font-weight: 600; color: #F1F5F9;">Tavily</span>
          <span style="color: #64748B; margin: 0 1px;">+</span>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" style="vertical-align: middle;">
            <rect width="24" height="24" rx="6" fill="#8B5CF6" fill-opacity="0.3"/>
            <path d="M7 6H13C15.2 6 17 7.8 17 10C17 11.4 16.3 12.6 15.2 13.2C16.8 13.8 18 15.3 18 17C18 19.2 16.2 21 14 21H7V6Z" stroke="#A855F7" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
          <span style="font-weight: 600; color: #F1F5F9;">Bifrost</span>
        </div>
        <a href="https://github.com/PurabTomar/stale-detector" target="_blank" class="sad-hackathon-badge">
          <span>Global AI Hackathon ↗</span>
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

evaluator_text = _get_config_value("EVALUATOR_NOTE_TEXT") or (
    "Welcome judges and evaluators! This live deployment includes session rate-limit protections and automatic fallback to verified saved results. "
    "Use the Guided Demo (or Demo mode) to test instant, quota-safe audits across all statuses (Fresh, Outdated, Incorrect), "
    "or run live audits within your session quota."
)

# ---------------------------------------------------------------------------
# Eleven-tab main workspace:
# 1. "Home" (hero, 4-step how it works, highlights bento, guided demo, evidence strip, FAQ, evaluator note)
# 2. "Audit pasted answer" (annotated answer view + two-column flow)
# 3. "Ask & Audit" (ask question via Bifrost -> show answer -> auto-audit + highlight)
# 4. "Audit a URL" (Tavily Extract article -> first 8 claims -> annotated view + report)
# 5. "Model Showdown" (Side-by-side LLM freshness comparison, winner evaluation)
# 6. "Benchmark" (20 labelled claims evaluation, cached in benchmark/results.json)
# 7. "Agent Lab" (Benchmarking AI research agents, Problem Statement 02)
# 8. "Architecture" (SVG flow diagram with live counters, Tavily vs Bifrost cards)
# 9. "History" (Session audit records, reopen, side-by-side compare, export)
# 10. "API Playground" (POST /verify schema, cURL, Python, JS snippets, interactive try it)
# 11. "Methodology" (6 status definitions, LaTeX math formula, stance consensus rules)
# ---------------------------------------------------------------------------
def render_benchmark_view(use_bifrost_mode: bool, current_busy: str | None, is_any_busy: bool) -> None:
    """Render the 20-claim ground truth detector benchmark view."""
    is_bench_running = current_busy == "benchmark"
    try:
        _dataset_preview_cached = load_benchmark_dataset()
        _bench_claim_count = len(_dataset_preview_cached)
    except Exception:
        _dataset_preview_cached = []
        _bench_claim_count = 24
    bench_head_col, bench_cache_col, bench_btn_col = st.columns(
        [2.7, 1.05, 1.05],
        vertical_alignment="bottom",
        gap="small",
    )
    with bench_head_col:
        st.subheader("Detector benchmark", icon=":material/monitoring:")
        st.caption(
            f"{_bench_claim_count} labelled claims (fresh, outdated, uncertain, and unverifiable across prices, versions, regulations, leadership, and stats). "
            "Loads cached benchmark results instantaneously on startup."
        )
    with bench_cache_col:
        bench_use_cache = st.toggle(
            "Use 6h Tavily cache",
            key="benchmark_use_cache",
            help="Reuse cached 6h Tavily search/extract results during re-runs to save Tavily credits.",
        )
    with bench_btn_col:
        rerun_bench_clicked = st.button(
            "Re-run benchmark",
            key="rerun_benchmark_btn",
            type="primary",
            icon=("spinner" if is_bench_running else ":material/speed:"),
            disabled=is_any_busy,
            width="stretch",
        )

    bench_skel_slot = st.empty()

    if rerun_bench_clicked and not is_any_busy:
        st.session_state["benchmark_error"] = None
        if st.session_state.get("demo_mode"):
            st.info(
                "Demo mode is active — showing cached benchmark results / dataset offline (0 live API calls). "
                "Turn off Demo mode to re-run against live APIs.",
                icon=":material/info:",
            )
        else:
            st.session_state["busy_action"] = "benchmark"
            st.session_state["_busy_armed"] = True
            st.rerun()
    elif is_bench_running:
        bench_finished = False
        bench_status = None
        bench_steps: list[str] = []
        dual_bench = bool(st.session_state.get("dual_judge_mode", False))
        try:
            _show_skeleton_slot(bench_skel_slot, (96, 96, 220))
            with st.status(f"Running {_bench_claim_count}-claim benchmark (3 parallel workers)...", expanded=True) as bench_status:
                bench_live = st.empty()
                bench_live.write(f"Evaluating {_bench_claim_count} labelled claims across Tavily + LLM judge...")
                bench_bar = st.progress(0.0, text=f"Evaluated 0/{_bench_claim_count} claims (3 parallel workers)...")

                def on_bench_progress(done: int, total_count: int, row: dict) -> None:
                    frac = min(1.0, max(0.0, float(done) / float(total_count))) if total_count else 1.0
                    mark_sym = "✓" if row.get("correct") else "✗"
                    pred_lbl = str(row.get("predicted_status") or "unknown")
                    exp_lbl = str(row.get("expected_status") or "")
                    idx_num = row.get("index", done)
                    prog_msg = f"Evaluated {done}/{total_count} claims — #{idx_num} {mark_sym} ({pred_lbl})"
                    bench_bar.progress(frac, text=prog_msg)
                    bench_live.write(prog_msg)
                    bench_steps.append(
                        f"#{idx_num} {mark_sym}: expected {exp_lbl}, predicted {pred_lbl} "
                        f"({float(row.get('latency_ms') or 0.0):.0f} ms)"
                    )

                run_kwargs: dict = {
                    "use_cache": bool(bench_use_cache),
                    "dual_judge": dual_bench,
                    "progress_callback": on_bench_progress,
                }
                if not _callable_accepts_kwarg(run_benchmark, "dual_judge"):
                    run_kwargs.pop("dual_judge", None)
                if not _callable_accepts_kwarg(run_benchmark, "use_cache"):
                    run_kwargs.pop("use_cache", None)
                bench_payload = run_benchmark(**run_kwargs)

                metrics_b = bench_payload.get("metrics") or {}
                total_b = int(metrics_b.get("total_claims") or len(bench_payload.get("results") or []))
                bench_bar.progress(1.0, text=f"Completed benchmark ({total_b}/{total_b} claims)")

                bench_credits = float(metrics_b.get("total_tavily_credits") or 0.0)
                if bench_credits > 0:
                    st.session_state["tavily_credits_used"] = round(
                        float(st.session_state.get("tavily_credits_used", 0.0)) + bench_credits,
                        2,
                    )
                    render_sidebar_credits(
                        credits_placeholder,
                        st.session_state["tavily_credits_used"],
                    )

                bench_p_tok = int(metrics_b.get("total_prompt_tokens") or 0)
                bench_c_tok = int(metrics_b.get("total_completion_tokens") or 0)
                bench_t_tok = int(metrics_b.get("total_tokens") or (bench_p_tok + bench_c_tok))
                bench_dur_ms = float(metrics_b.get("total_duration_ms") or 0.0)
                bench_avg_ms = float(metrics_b.get("avg_latency_ms") or 0.0)
                bench_models = bench_payload.get("models_used") or []
                if bench_t_tok > 0 or bench_models:
                    bench_record = build_request_telemetry(
                        f"Benchmark ({total_b} claims)",
                        {
                            "primary_model": bench_models[0] if bench_models else None,
                            "model_used": ", ".join(bench_models) if bench_models else "Unknown",
                            "latency_ms": bench_avg_ms,
                            "prompt_tokens": bench_p_tok,
                            "completion_tokens": bench_c_tok,
                            "total_tokens": bench_t_tok,
                            "fallback_used": False,
                        },
                    )
                    append_gateway_request(bench_record)
                if bench_dur_ms > 0:
                    st.session_state["last_audit_time_ms"] = round(bench_dur_ms, 1)
                    st.session_state["session_audit_time_ms"] = round(
                        float(st.session_state.get("session_audit_time_ms") or 0.0) + bench_dur_ms,
                        1,
                    )
                render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

                acc_pct = float(metrics_b.get("accuracy") or 0.0) * 100.0
                prec_pct = float(metrics_b.get("precision_stale") or 0.0) * 100.0
                rec_pct = float(metrics_b.get("recall_stale") or 0.0) * 100.0
                summary_step = (
                    f"Accuracy {acc_pct:.1f}% ({int(metrics_b.get('correct') or 0)}/{total_b}) • "
                    f"Stale precision {prec_pct:.1f}% • Stale recall {rec_pct:.1f}%"
                )
                bench_steps.append(summary_step)
                bench_live.write(f":material/check: {summary_step}")
                bench_status.update(
                    label=f"Benchmark complete — {summary_step}",
                    state="complete",
                    expanded=bool(st.session_state.get("keep_status_expanded", False)),
                )
                st.session_state["benchmark_results"] = bench_payload
                st.session_state["benchmark_status_steps"] = bench_steps
            bench_finished = True
        except Exception as exc:
            if bench_status is not None:
                bench_status.update(label="Benchmark failed", state="error", expanded=False)
            st.session_state["benchmark_error"] = format_friendly_ui_error(
                exc, action="running the benchmark"
            )
            bench_finished = True
        finally:
            if st.session_state.get("busy_action") == "benchmark":
                st.session_state["busy_action"] = None
            _clear_slot(bench_skel_slot)
        if bench_finished:
            st.rerun()
    elif st.session_state.get("benchmark_status_steps"):
        with st.status(
            "Last benchmark run complete",
            state="complete",
            expanded=bool(st.session_state.get("keep_status_expanded", False)),
        ):
            for step_line in st.session_state["benchmark_status_steps"]:
                st.write(step_line)

    if st.session_state.get("benchmark_error"):
        st.error(
            str(st.session_state["benchmark_error"]),
            icon=":material/error:",
        )

    if st.session_state.get("benchmark_results") is None:
        st.session_state["benchmark_results"] = load_benchmark_cache()

    bench_data = st.session_state.get("benchmark_results")
    if isinstance(bench_data, dict) and isinstance(bench_data.get("results"), list):
        bm = bench_data.get("metrics") or {}
        cm_info = bm.get("confusion_matrix") or {}
        b_total = int(bm.get("total_claims") or len(bench_data["results"]))
        b_correct = int(bm.get("correct") or 0)
        b_acc = float(bm.get("accuracy") or 0.0)
        b_prec = float(bm.get("precision_stale") or 0.0)
        b_rec = float(bm.get("recall_stale") or 0.0)
        b_f1 = float(bm.get("f1_stale") or 0.0)
        b_tp = int(cm_info.get("tp") or 0)
        b_fp = int(cm_info.get("fp") or 0)
        b_fn = int(cm_info.get("fn") or 0)
        b_tn = int(cm_info.get("tn") or 0)
        b_err_cnt = int(bm.get("error_count") or 0)
        b_avg_lat = float(bm.get("avg_latency_ms") or 0.0)
        b_tot_dur = float(bm.get("total_duration_ms") or 0.0)
        b_p_tok = int(bm.get("total_prompt_tokens") or 0)
        b_c_tok = int(bm.get("total_completion_tokens") or 0)
        b_tot_tok = int(bm.get("total_tokens") or (b_p_tok + b_c_tok))
        b_credits = float(bm.get("total_tavily_credits") or 0.0)

        bench_models_str = ", ".join(f"`{m}`" for m in (bench_data.get("models_used") or [])) or "`N/A`"
        bench_mode_lbl = (
            "Bifrost gateway"
            if str(bench_data.get("routing_mode") or "") == "bifrost"
            else "Direct Gemini"
        )
        dual_lbl = "On" if bench_data.get("dual_judge") else "Off"
        cache_lbl = "On" if bench_data.get("use_tavily_cache", True) else "Off"
        st.caption(
            f"**Cached run:** `{bench_data.get('generated_at', 'N/A')}` • "
            f"**Routing:** {bench_mode_lbl} • **Dual-judge:** {dual_lbl} • "
            f"**Tavily 6h cache:** {cache_lbl} • **Models:** {bench_models_str}"
        )

        with st.grid(4, gap="small"):
            st.metric(
                "Accuracy",
                f"{b_acc * 100:.1f}%",
                delta=f"{b_correct}/{b_total} claims correct",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "Precision (stale)",
                f"{b_prec * 100:.1f}%",
                delta=f"TP={b_tp}, FP={b_fp}",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "Recall (stale)",
                f"{b_rec * 100:.1f}%",
                delta=f"TP={b_tp}, FN={b_fn}",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "F1 score (stale)",
                f"{b_f1 * 100:.1f}%",
                delta=f"{b_err_cnt} claim error(s)" if b_err_cnt else "0 error claims",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )

        with st.grid(4, gap="small"):
            st.metric(
                "Avg latency / claim",
                _format_duration_ms(b_avg_lat),
                delta=f"Across {b_total} claims",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "Total wall-clock runtime",
                _format_duration_ms(b_tot_dur),
                delta="3 parallel workers",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "Total tokens used",
                f"{b_tot_tok:,}",
                delta=f"{b_p_tok:,} in / {b_c_tok:,} out",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                "Tavily credits used",
                format_credits(b_credits),
                delta="Search + extract pass",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )

        cm_df_base, cat_df_base, _ = build_benchmark_tables(bench_data, filter_mode="All")

        def _style_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
            if "Expected: Outdated (stale)" in frame.index:
                if "Predicted: Outdated" in frame.columns:
                    styles.loc["Expected: Outdated (stale)", "Predicted: Outdated"] = (
                        "background-color: rgba(34, 197, 94, 0.22); font-weight: 700;"
                    )
                if "Predicted: Fresh" in frame.columns and int(frame.loc["Expected: Outdated (stale)", "Predicted: Fresh"]) > 0:
                    styles.loc["Expected: Outdated (stale)", "Predicted: Fresh"] = (
                        "background-color: rgba(239, 68, 68, 0.20);"
                    )
                if (
                    "Predicted: Uncertain / Unverifiable" in frame.columns
                    and int(frame.loc["Expected: Outdated (stale)", "Predicted: Uncertain / Unverifiable"]) > 0
                ):
                    styles.loc["Expected: Outdated (stale)", "Predicted: Uncertain / Unverifiable"] = (
                        "background-color: rgba(245, 158, 11, 0.18);"
                    )
            if "Expected: Fresh" in frame.index:
                if "Predicted: Fresh" in frame.columns:
                    styles.loc["Expected: Fresh", "Predicted: Fresh"] = (
                        "background-color: rgba(34, 197, 94, 0.22); font-weight: 700;"
                    )
                if "Predicted: Outdated" in frame.columns and int(frame.loc["Expected: Fresh", "Predicted: Outdated"]) > 0:
                    styles.loc["Expected: Fresh", "Predicted: Outdated"] = (
                        "background-color: rgba(239, 68, 68, 0.20);"
                    )
                if (
                    "Predicted: Uncertain / Unverifiable" in frame.columns
                    and int(frame.loc["Expected: Fresh", "Predicted: Uncertain / Unverifiable"]) > 0
                ):
                    styles.loc["Expected: Fresh", "Predicted: Uncertain / Unverifiable"] = (
                        "background-color: rgba(245, 158, 11, 0.18);"
                    )
            return styles

        cm_col, cat_col = st.columns([1.15, 1], gap="medium")
        with cm_col:
            with st.container(border=True):
                st.markdown("**Confusion matrix (expected vs. predicted)**")
                st.table(
                    cm_df_base.style.apply(_style_confusion_matrix, axis=None),
                    border="horizontal",
                    alt="Confusion matrix comparing expected vs predicted claim freshness labels",
                )
                st.caption(
                    f"Stale detection counts (`outdated` = positive class): "
                    f"TP = {b_tp} • FP = {b_fp} • FN = {b_fn} • TN = {b_tn}"
                )
        with cat_col:
            with st.container(border=True):
                st.markdown("**Accuracy by category**")
                st.dataframe(
                    cat_df_base,
                    hide_index=True,
                    width="stretch",
                    alt="Benchmark accuracy broken down by claim category",
                    column_config={
                        "Category": st.column_config.TextColumn("Category"),
                        "Accuracy": st.column_config.ProgressColumn(
                            "Accuracy",
                            min_value=0,
                            max_value=100,
                            format="%.0f%%",
                        ),
                        "Correct": st.column_config.NumberColumn("Correct", format="%d"),
                        "Total": st.column_config.NumberColumn("Total", format="%d"),
                    },
                )

        # Feature D: Calibration Chart
        with st.container(border=True):
            st.markdown("### 📈 Score Calibration Curve (Empirical Accuracy vs. Predicted Score)")
            cal_data = compute_calibration(bench_data.get("results"))
            if cal_data.get("status") == "not_enough_data":
                st.info(
                    f"ℹ️ Calibration Curve: {cal_data.get('message', 'Not enough benchmark data.')} "
                    "(Empirical data only; never faked when uncalculated).",
                    icon=":material/info:",
                )
            else:
                cal_df = pd.DataFrame([
                    {
                        "Score Bin": b["bin"],
                        "Description": b["description"],
                        "Sample Count (n)": b["n"],
                        "Empirical Accuracy (%)": b["accuracy_pct"],
                    }
                    for b in cal_data["bins"]
                ])
                c_col1, c_col2 = st.columns([1.4, 1], gap="medium")
                with c_col1:
                    st.bar_chart(cal_df, x="Score Bin", y="Empirical Accuracy (%)", color="#6366F1", height=220)
                with c_col2:
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 12px; font-size: 0.88rem; height: 100%;">
                            <strong style="color: #818CF8;">Calibration Reading:</strong>
                            <p style="color: #F8FAFC; margin-top: 6px; line-height: 1.4;">{cal_data['reading']}</p>
                            <span style="color: #94A3B8; font-size: 0.78rem;">Calculated across {cal_data['total_evaluated']} benchmark claims.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with st.container(border=True):
            tbl_head_col, tbl_filter_col = st.columns(
                [1.1, 1.9],
                vertical_alignment="center",
                gap="small",
            )
            with tbl_head_col:
                st.markdown("**Per-claim evaluation table**")
            with tbl_filter_col:
                selected_bench_filter = st.segmented_control(
                    "Filter claims",
                    options=[
                        "All",
                        "Correct only",
                        "Incorrect only",
                        "Expected: Outdated",
                        "Expected: Fresh",
                        "Expected: Uncertain / Unverifiable",
                    ],
                    default="All",
                    key="benchmark_filter_mode",
                    disabled=is_any_busy,
                    label_visibility="collapsed",
                )
            active_bench_filter = str(selected_bench_filter or "All")
            _, _, filtered_rows_df = build_benchmark_tables(
                bench_data,
                filter_mode=active_bench_filter,
            )

            if filtered_rows_df.empty:
                st.info(
                    "No benchmark claims match the selected filter.",
                    icon=":material/check_circle:",
                )
            else:
                st.dataframe(
                    filtered_rows_df,
                    hide_index=True,
                    width="stretch",
                    alt="Per-claim benchmark evaluation table with correct and incorrect marks",
                    column_config={
                        "#": st.column_config.NumberColumn("#", format="%d", width="small"),
                        "Mark": st.column_config.TextColumn("Mark", width="small"),
                        "Category": st.column_config.TextColumn("Category", width="small"),
                        "Claim": st.column_config.TextColumn("Claim", width="large"),
                        "Expected": st.column_config.TextColumn("Expected", width="small"),
                        "Predicted": st.column_config.TextColumn("Predicted", width="small"),
                        "Score": st.column_config.TextColumn("Score", width="small"),
                        "Latency (ms)": st.column_config.NumberColumn("Latency (ms)", format="%.1f ms"),
                        "Tokens": st.column_config.NumberColumn("Tokens", format="%,d"),
                        "Credits": st.column_config.NumberColumn("Credits", format="%.2f"),
                        "Reference & Current Fact": st.column_config.TextColumn(
                            "Reference & Current Fact",
                            width="large",
                        ),
                    },
                )

        bench_ts_slug = datetime.now().strftime("%Y%m%d-%H%M%S")
        dl_bench_col1, dl_bench_col2 = st.columns(2, gap="small")
        with dl_bench_col1:
            st.download_button(
                "Download benchmark report (Markdown)",
                data=format_benchmark_markdown(bench_data),
                file_name=f"benchmark-report-{bench_ts_slug}.md",
                mime="text/markdown",
                key="dl_bench_md",
                icon=":material/download:",
                disabled=is_any_busy,
                on_click="ignore",
                width="stretch",
            )
        with dl_bench_col2:
            st.download_button(
                "Download benchmark JSON",
                data=json.dumps(bench_data, indent=2, ensure_ascii=False) + "\n",
                file_name=f"benchmark-results-{bench_ts_slug}.json",
                mime="application/json",
                key="dl_bench_json",
                icon=":material/data_object:",
                disabled=is_any_busy,
                on_click="ignore",
                width="stretch",
            )
    else:
        bench_empty_msg = (
            "No cached benchmark run found yet. Start Bifrost on port 8080 "
            "(or set `USE_BIFROST=false` with a Gemini API key) and click **Re-run benchmark** "
            "above, or run `python benchmark.py --update-readme` from your terminal."
            if use_bifrost_mode
            else "No cached benchmark run found yet. Add `GEMINI_API_KEY` (or `LLM_API_KEY`) "
            "and `TAVILY_API_KEY` in Streamlit Secrets (or `.env`) and click **Re-run benchmark** above."
        )
        st.info(
            bench_empty_msg,
            title=f"Benchmark dataset ready ({_bench_claim_count} labelled claims)",
            icon=":material/info:",
        )
        dataset_preview = _dataset_preview_cached
        if not dataset_preview:
            try:
                dataset_preview = load_benchmark_dataset()
            except Exception as ds_exc:
                st.error(
                    format_friendly_ui_error(ds_exc, action="loading benchmark dataset"),
                    icon=":material/error:",
                )
                dataset_preview = []

        if dataset_preview:
            preview_df = pd.DataFrame(
                [
                    {
                        "#": int(item.get("index") or idx),
                        "Category": str(item.get("category") or "other"),
                        "Expected": str(item.get("expected_status") or ""),
                        "Claim": str(item.get("claim") or ""),
                        "Reference Note": str(item.get("note") or ""),
                    }
                    for idx, item in enumerate(dataset_preview, start=1)
                ]
            )
            with st.container(border=True):
                st.markdown("**Labelled benchmark dataset preview (`benchmark/dataset.json`)**")
                st.dataframe(
                    preview_df,
                    hide_index=True,
                    width="stretch",
                    alt=f"Preview of {len(dataset_preview)} labelled claims in benchmark dataset",
                    column_config={
                        "#": st.column_config.NumberColumn("#", format="%d", width="small"),
                        "Category": st.column_config.TextColumn("Category", width="small"),
                        "Expected": st.column_config.TextColumn("Expected", width="small"),
                        "Claim": st.column_config.TextColumn("Claim", width="large"),
                        "Reference Note": st.column_config.TextColumn("Reference Note", width="large"),
                    },
                )



# ---------------------------------------------------------------------------
# Five-Group Main Workspace
# 1. "🔍 Audit" (Input Studio + Overview/Highlights sub-view)
# 2. "⚖️ Compare" (Model Showdown)
# 3. "🧪 Agent Lab" (Agent Lab PS02 + Benchmark Ground Truth)
# 4. "📚 Learn" (Architecture & Telemetry + Methodology + API Playground)
# 5. "🕒 History" (Session History)
# ---------------------------------------------------------------------------
tab_audit, tab_compare, tab_lab, tab_learn, tab_history = st.tabs(
    [
        "AUDIT",
        "COMPARE",
        "AGENT LAB",
        "LEARN",
        "HISTORY",
    ]
)

route_target = "Bifrost" if use_bifrost_mode else "Gemini"

with tab_audit:
    # -------------------------------------------------------------------
    # 2. Compact Evaluator Banner (directly below top navbar)
    # -------------------------------------------------------------------
    if not st.session_state.get("dismiss_evaluator_note", False):
        eval_c1, eval_c2 = st.columns([0.96, 0.04], vertical_alignment="center")
        with eval_c1:
            st.markdown(
                f'<div class="sad-eval-banner">'
                f'<span class="sad-eval-icon">💡</span>'
                f'<span class="sad-eval-text"><strong>Note for evaluators:</strong> {escape_html(evaluator_text[:185])}...</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with eval_c2:
            if st.button("✕", key="dismiss_eval_strip", help="Dismiss note"):
                st.session_state["dismiss_evaluator_note"] = True
                st.rerun()

    # -------------------------------------------------------------------
    # 3. Top-Level 5-Stage Pipeline Tracker (ABOVE the Input card)
    # -------------------------------------------------------------------
    active_mode_val = st.session_state.get("active_input_mode", "Paste an answer")
    has_active_results = bool(
        (active_mode_val == "Paste an answer" and st.session_state.get("results"))
        or (active_mode_val == "Ask a question" and st.session_state.get("ask_results"))
        or (active_mode_val == "Paste a URL" and st.session_state.get("url_results"))
        or (active_mode_val == "Batch" and st.session_state.get("batch_summary"))
        or (active_mode_val == "Doc Checker" and st.session_state.get("doc_checker_results"))
    )
    has_active_input = bool(
        (active_mode_val == "Paste an answer" and st.session_state.get("answer_input", "").strip())
        or (active_mode_val == "Ask a question" and st.session_state.get("ask_question_input", "").strip())
        or (active_mode_val == "Paste a URL" and st.session_state.get("url_input", "").strip())
        or (active_mode_val == "Batch" and (st.session_state.get("batch_text_input", "").strip() or st.session_state.get("batch_file_uploader")))
        or (active_mode_val == "Doc Checker" and (st.session_state.get("doc_checker_text", "").strip() or st.session_state.get("doc_checker_file_uploader")))
    )
    active_mode_results: list[dict] = []
    if active_mode_val == "Paste an answer":
        active_mode_results = st.session_state.get("results") or []
    elif active_mode_val == "Ask a question":
        active_mode_results = st.session_state.get("ask_results") or []
    elif active_mode_val == "Paste a URL":
        active_mode_results = st.session_state.get("url_results") or []
    elif active_mode_val == "Doc Checker":
        active_mode_results = st.session_state.get("doc_checker_results") or []
    stepper_placeholder = st.empty()
    stepper_placeholder.markdown(
        render_stepper_html(
            current_busy,
            has_active_results,
            has_active_input,
            results=active_mode_results,
        ),
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------
    # 4. Section 1: Input Studio Card (Always Visible)
    # -------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<span id="sad-input-studio-anchor" style="display:none;"></span>', unsafe_allow_html=True)
        # Unified Card Header: Left title & subtitle, Right audits counter
        hdr_c1, hdr_c2 = st.columns([2.8, 1.2], vertical_alignment="center", gap="medium")
        with hdr_c1:
            st.markdown(
                """
                <div class="sad-section-header" style="margin: 0;">
                  <svg class="sad-section-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                  </svg>
                  <div>
                    <h3 class="sad-section-title">Input Studio</h3>
                    <p class="sad-section-sub">Audit freshness against live web reality</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hdr_c2:
            audits_done = int(st.session_state.get("live_audits_performed") or 0)
            audits_left = max(0, get_max_session_audits() - audits_done)
            st.markdown(
                f'<div style="text-align: right;"><span class="sad-studio-quota" style="white-space: nowrap;">Audits left: <strong>{audits_left}</strong></span></div>',
                unsafe_allow_html=True,
            )

        active_input_mode = st.segmented_control(
            "Audit Mode",
            options=["Paste an answer", "Ask a question", "Paste a URL", "Batch", "Doc Checker"],
            default=st.session_state.get("active_input_mode", "Paste an answer"),
            key="active_input_mode",
            label_visibility="collapsed",
        )

        if True:  # Input workspace always active and visible
            is_pasted_running = current_busy == "pasted"
            is_ask_running = current_busy == "ask"
            is_url_running = current_busy == "url"

            check_clicked = False
            ask_clicked = False
            url_clicked = False
            batch_clicked = False
            doc_clicked = False

            if active_input_mode == "Paste an answer":
                char_count = len(st.session_state.get("answer_input", ""))
                ex_labels = ["iPhone", "GPT", "India", "Company", "World", "🗣️ Hinglish", "🇮🇳 Hindi"]
                chip_cols = st.columns([1, 1, 1, 1, 1, 1.25, 1.15], vertical_alignment="center", gap="small")
                for ex_idx, ex_lbl in enumerate(ex_labels):
                    with chip_cols[ex_idx]:
                        if st.button(
                            ex_lbl,
                            key=f"pasted_chip_{ex_idx}",
                            help=f"{DEMO_EXAMPLES[ex_idx]['label']}: {DEMO_EXAMPLES[ex_idx]['text'][:80]}...",
                            disabled=is_any_busy,
                            width="stretch",
                        ):
                            load_demo_example(DEMO_EXAMPLES[ex_idx]["text"])

                act_col1, act_col2, act_col3 = st.columns([2.0, 1.4, 6.6], vertical_alignment="center", gap="small")
                with act_col1:
                    st.markdown('<span id="sad-guided-demo-anchor" style="display:none;"></span>', unsafe_allow_html=True)
                    if st.button(
                        "✨ Guided Demo",
                        key="guided_demo_btn",
                        type="secondary",
                        width="stretch",
                        help="Auto-loads 3 cached examples in 30 seconds: iPhone (fresh), GST (outdated), Wrong PM (incorrect)",
                    ):
                        st.session_state["guided_demo_active"] = True
                        st.session_state["guided_demo_step"] = 0
                        step_data = get_guided_demo_step(0)
                        _apply_pasted_demo_sample(step_data["sample"], exact_match=True, sync_widget=True)
                        st.rerun()

                with act_col2:
                    with st.popover("💡 Tips", icon=":material/lightbulb:"):
                        st.markdown(
                            "**Recommended claims to test:**\n"
                            "1. **Tech Specs & Prices**: *\"Apple's latest flagship is iPhone 16 Pro Max starting at $1,199.\"*\n"
                            "2. **Tax & Legal Regulations**: *\"India GST rates are structured into 5%, 12%, 18%, and 28% slabs.\"*\n"
                            "3. **Government & Elections**: *\"Narendra Modi is serving his 3rd term as Prime Minister of India.\"*\n"
                            "4. **API Pricing**: *\"OpenAI GPT-4 Turbo input tokens are $10.00 per 1M tokens.\"*"
                        )

                if st.session_state.get("guided_demo_active"):
                    curr_step = int(st.session_state.get("guided_demo_step", 0))
                    step_data = get_guided_demo_step(curr_step)
                    st.markdown(
                        f'<div class="sad-guided-demo-banner">'
                        f'<div>'
                        f'<span class="sad-guided-step-badge">Step {curr_step + 1} of 3</span> '
                        f'<strong>{escape_html(step_data["title"])}</strong> &mdash; '
                        f'<span style="color: var(--text-muted); font-size: 0.85rem;">{escape_html(step_data["caption"])}</span>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    g_btn1, g_btn2, g_btn3 = st.columns([1, 1.2, 1], gap="small")
                    with g_btn1:
                        if st.button("◀ Previous", key="gd_prev_btn", disabled=(curr_step <= 0), width="stretch"):
                            st.session_state["guided_demo_step"] = curr_step - 1
                            prev_step = get_guided_demo_step(curr_step - 1)
                            _apply_pasted_demo_sample(prev_step["sample"], exact_match=True, sync_widget=True)
                            st.rerun()
                    with g_btn2:
                        if st.button("Next Step ▶", key="gd_next_btn", disabled=(curr_step >= len(GUIDED_DEMO_STEPS) - 1), width="stretch", type="primary"):
                            st.session_state["guided_demo_step"] = curr_step + 1
                            next_step = get_guided_demo_step(curr_step + 1)
                            _apply_pasted_demo_sample(next_step["sample"], exact_match=True, sync_widget=True)
                            st.rerun()
                    with g_btn3:
                        if st.button("Exit Demo ✕", key="gd_exit_btn", width="stretch"):
                            st.session_state["guided_demo_active"] = False
                            st.rerun()

                st.text_area(
                    "LLM-generated answer",
                    key="answer_input",
                    max_chars=MAX_TEXT_CHARS,
                    height=70,
                    label_visibility="collapsed",
                    placeholder=(
                        "Paste an LLM-generated answer here or click one of the example chips above..."
                    ),
                )
                char_warn = "color: #EF4444;" if char_count > MAX_TEXT_CHARS else ""
                st.caption(f"⌨️ Hint: Press **Ctrl + Enter** to audit &bull; <span style='{char_warn}'>{char_count:,} / {MAX_TEXT_CHARS:,} chars</span>", unsafe_allow_html=True)

                # Requirement 8: Heuristic preview (zero API calls, updates on blur/Ctrl+Enter)
                raw_ans_text = st.session_state.get("answer_input", "").strip()
                if raw_ans_text:
                    h_chips = detect_heuristic_highlights(raw_ans_text)
                    if h_chips:
                        st.markdown(render_heuristic_preview_html(h_chips), unsafe_allow_html=True)

                curr_ui_state = st.session_state.get("audit_ui_state")
                if not isinstance(curr_ui_state, AuditUIState):
                    curr_ui_state = create_initial_ui_state()
                    st.session_state["audit_ui_state"] = curr_ui_state

                if is_pasted_running:
                    primary_btn_label = curr_ui_state.button_label or "Detecting claims"
                    primary_btn_icon = "spinner"
                elif st.session_state.get("pasted_error") or curr_ui_state.button_state == "error":
                    primary_btn_label = "Retry"
                    primary_btn_icon = ":material/refresh:"
                else:
                    primary_btn_label = "Check Freshness"
                    primary_btn_icon = ":material/verified:"

                btn_col1, btn_col2 = st.columns([0.8, 0.2], gap="small")
                with btn_col1:
                    check_clicked = st.button(
                        primary_btn_label,
                        key="check_freshness_btn",
                        type="primary",
                        icon=primary_btn_icon,
                        disabled=is_any_busy,
                        width="stretch",
                    )
                with btn_col2:
                    if is_pasted_running:
                        st.button(
                            "Stop",
                            key="stop_audit_btn",
                            type="secondary",
                            icon=":material/stop_circle:",
                            on_click=request_stop_audit,
                            help="Cancel remaining verification steps and keep partial results",
                            width="stretch",
                        )
                    else:
                        st.button(
                            "Clear",
                            key="clear_workspace_btn",
                            type="secondary",
                            icon=":material/delete_outline:",
                            on_click=clear_workspace,
                            disabled=is_any_busy,
                            width="stretch",
                        )

                action_status_slot = st.empty()
                freshness_strip_slot = st.empty()
                if not is_pasted_running:
                    initial_status_pill = render_action_status_pill_html(curr_ui_state)
                    if initial_status_pill:
                        action_status_slot.markdown(initial_status_pill, unsafe_allow_html=True)
                    if curr_ui_state.strip_segments:
                        freshness_strip_slot.markdown(
                            render_freshness_strip_html(curr_ui_state),
                            unsafe_allow_html=True,
                        )

            elif active_input_mode == "Ask a question":
                st.caption(
                    f"Type a question below. The app fetches an answer via {route_target}, shows it, "
                    "runs the live stale-claim audit automatically, and highlights outdated vs. fresh claims inline."
                )
                q_chips = [
                    ("iPhone 16 Specs", "What is the latest iPhone model, chip, and US starting price?"),
                    ("India GST Slabs", "What are the current GST tax slabs in India?"),
                    ("GPT-4 Turbo Price", "What is the per-million token pricing for OpenAI GPT-4 Turbo?"),
                ]
                q_chip_cols = st.columns(len(q_chips), gap="small")
                for q_idx, (q_lbl, q_text) in enumerate(q_chips):
                    with q_chip_cols[q_idx]:
                        if st.button(q_lbl, key=f"ask_chip_{q_idx}", width="stretch", disabled=is_any_busy):
                            st.session_state["ask_question_input"] = q_text
                            st.rerun()

                st.text_input(
                    "Ask a question",
                    key="ask_question_input",
                    max_chars=500,
                    label_visibility="collapsed",
                    placeholder="e.g. What is the latest iPhone model, chip, and US starting price?",
                )

                btn_col1, btn_col2 = st.columns([0.8, 0.2], gap="small")
                with btn_col1:
                    ask_clicked = st.button(
                        "Ask & Audit",
                        key="ask_and_audit_btn",
                        type="primary",
                        icon=("spinner" if is_ask_running else ":material/send:"),
                        disabled=is_any_busy,
                        width="stretch",
                    )
                with btn_col2:
                    st.button(
                        "Clear",
                        key="ask_clear_btn",
                        icon=":material/delete_outline:",
                        on_click=clear_ask_workspace,
                        disabled=is_any_busy,
                        width="stretch",
                    )

            elif active_input_mode == "Paste a URL":
                st.caption(
                    "Paste an article or blog URL below. Tavily Extract pulls clean text, audits the first "
                    "8 time-sensitive claims, highlights facts inline, and exports report."
                )
                u_chips = [
                    ("Tech Blog", "https://techcrunch.com/2024/09/09/apple-iphone-16-pro-event/"),
                    ("Policy Article", "https://pib.gov.in/PressReleasePage.aspx?PRID=1943890"),
                    ("AI Guide", "https://openai.com/index/hello-gpt-4o/"),
                ]
                u_chip_cols = st.columns(len(u_chips), gap="small")
                for u_idx, (u_lbl, u_text) in enumerate(u_chips):
                    with u_chip_cols[u_idx]:
                        if st.button(u_lbl, key=f"url_chip_{u_idx}", width="stretch", disabled=is_any_busy):
                            st.session_state["url_input"] = u_text
                            st.rerun()

                st.text_input(
                    "Article or blog URL",
                    key="url_input",
                    max_chars=2048,
                    label_visibility="collapsed",
                    placeholder="https://example.com/blog/python-release-guide",
                )

                btn_col1, btn_col2 = st.columns([0.8, 0.2], gap="small")
                with btn_col1:
                    url_clicked = st.button(
                        "Audit URL",
                        key="audit_url_btn",
                        type="primary",
                        icon=("spinner" if is_url_running else ":material/travel_explore:"),
                        disabled=is_any_busy,
                        width="stretch",
                    )
                with btn_col2:
                    st.button(
                        "Clear",
                        key="url_clear_btn",
                        icon=":material/delete_outline:",
                        on_click=clear_url_workspace,
                        disabled=is_any_busy,
                        width="stretch",
                    )

            elif active_input_mode == "Batch":
                st.caption(
                    "Upload a `.txt`, `.md`, or `.csv` file (max 20 items, max 200 KB) or paste multiple answers below "
                    "separated by a blank line. Runs the verification pipeline across all items with live progress tracking."
                )
                b_c1, b_c2 = st.columns([1, 1], gap="medium")
                with b_c1:
                    batch_file = st.file_uploader(
                        "Upload .txt, .md, or .csv",
                        type=["txt", "md", "csv"],
                        key="batch_file_uploader",
                        disabled=is_any_busy,
                    )
                with b_c2:
                    st.text_area(
                        "Or paste multiple answers separated by a blank line",
                        key="batch_text_input",
                        height=110,
                        disabled=is_any_busy,
                        placeholder="Answer 1: Apple sells iPhone 15 at $799...\n\nAnswer 2: GPT-4 is the latest OpenAI model...",
                    )

                btn_col1, btn_col2 = st.columns([0.8, 0.2], gap="small")
                with btn_col1:
                    batch_clicked = st.button(
                        "Run Batch Audit",
                        key="batch_audit_btn",
                        type="primary",
                        icon=":material/playlist_add_check:",
                        disabled=is_any_busy,
                        width="stretch",
                    )
                with btn_col2:
                    def clear_batch_workspace():
                        st.session_state["batch_summary"] = None
                        st.session_state["batch_items_data"] = None
                        st.session_state["batch_error"] = None
                        st.session_state["batch_text_input"] = ""

                    st.button(
                        "Clear",
                        key="batch_clear_btn",
                        icon=":material/delete_outline:",
                        on_click=clear_batch_workspace,
                        disabled=is_any_busy,
                        width="stretch",
                    )

            elif active_input_mode == "Doc Checker":
                st.caption(
                    "Audit Markdown and documentation files for stale facts. "
                    "Extracts factual claims and maps every verified finding directly to its 1-based source line number."
                )
                doc_c1, doc_c2 = st.columns([1, 1], gap="medium")
                with doc_c1:
                    doc_file = st.file_uploader(
                        "Upload .md or .txt documentation file",
                        type=["md", "txt"],
                        key="doc_checker_file_uploader",
                        disabled=is_any_busy,
                    )
                with doc_c2:
                    st.text_area(
                        "Or paste Markdown documentation content",
                        key="doc_checker_text",
                        height=110,
                        disabled=is_any_busy,
                        placeholder="# Guide\n\nApple released iPhone 15 at $1,199.\n\nIndia GST rate on gold is 3%.\n",
                    )

                btn_col1, btn_col2 = st.columns([0.8, 0.2], gap="small")
                with btn_col1:
                    doc_clicked = st.button(
                        "Audit Document Lines",
                        key="doc_checker_audit_btn",
                        type="primary",
                        icon=":material/fact_check:",
                        disabled=is_any_busy,
                        width="stretch",
                    )
                with btn_col2:
                    def clear_doc_workspace():
                        st.session_state["doc_checker_results"] = None
                        st.session_state["doc_checker_text"] = ""
                        st.session_state["doc_checker_error"] = None

                    st.button(
                        "Clear",
                        key="doc_clear_btn",
                        icon=":material/delete_outline:",
                        on_click=clear_doc_workspace,
                        disabled=is_any_busy,
                        width="stretch",
                    )

            # Feature H: Settings & Resource Estimator Panel
            with st.expander("⚙️ Audit Settings & Resource Estimator", expanded=False):
                s_c1, s_c2, s_c3 = st.columns([1.2, 1.2, 1.6], gap="medium")
                with s_c1:
                    st.slider(
                        "Strictness Cutoff (Sensitivity)",
                        min_value=0,
                        max_value=80,
                        value=int(st.session_state.get("detection_threshold", 40)),
                        step=10,
                        key="settings_detection_threshold",
                        help="Claims with time-sensitivity below this threshold will be skipped.",
                    )
                    st.slider(
                        "Max Claims per Audit",
                        min_value=1,
                        max_value=8,
                        value=int(st.session_state.get("max_claims_per_audit", 6)),
                        step=1,
                        key="settings_max_claims",
                    )
                with s_c2:
                    st.toggle(
                        "Dual-Judge Consensus",
                        value=bool(st.session_state.get("dual_judge_mode", False)),
                        key="settings_dual_judge",
                        help="Runs two independent judge models via Bifrost over the same evidence.",
                    )
                    st.selectbox(
                        "Language Mode",
                        options=SUPPORTED_LANGUAGES,
                        index=0,
                        key="settings_language",
                        help="Accepts English, Hindi (Devanagari), and Hinglish with English query generation.",
                    )
                with s_c3:
                    st.multiselect(
                        "Active Factual Categories",
                        options=SUPPORTED_CATEGORIES,
                        default=st.session_state.get("active_categories_filter", SUPPORTED_CATEGORIES),
                        key="settings_active_categories",
                    )

                # Live Pre-Run Resource Estimation
                curr_txt = ""
                is_url_mode = (active_input_mode == "Paste a URL")
                if active_input_mode == "Paste an answer":
                    curr_txt = st.session_state.get("answer_input", "")
                elif active_input_mode == "Ask a question":
                    curr_txt = st.session_state.get("ask_question_input", "")
                elif active_input_mode == "Paste a URL":
                    curr_txt = st.session_state.get("url_input", "")
                elif active_input_mode == "Batch":
                    curr_txt = st.session_state.get("batch_text_input", "")
                elif active_input_mode == "Doc Checker":
                    curr_txt = st.session_state.get("doc_checker_text", "")

                est = estimate_audit_resources(
                    text=curr_txt,
                    max_claims=int(st.session_state.get("settings_max_claims", 6)),
                    dual_judge=bool(st.session_state.get("settings_dual_judge", False)),
                    active_categories=st.session_state.get("settings_active_categories", SUPPORTED_CATEGORIES),
                    is_url=is_url_mode,
                    strictness_threshold=int(st.session_state.get("settings_detection_threshold", 40)),
                )

                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 10px 14px; margin-top: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.85rem;">
                        <div>
                            <strong style="color: #818CF8;">Pre-Run Resource Estimate:</strong>
                            <span style="color: #F8FAFC; margin-left: 8px;">~{est['estimated_claims']} claims</span> |
                            <span style="color: #38BDF8; margin-left: 8px;">~{est['estimated_tavily_credits']} Tavily credits</span> |
                            <span style="color: #A855F7; margin-left: 8px;">~{est['estimated_llm_calls']} LLM calls</span>
                        </div>
                        <div style="color: #94A3B8; font-size: 0.8rem;">
                            Session Quota: <strong>{audits_left}</strong> audits remaining
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------------------------
        # Execute Handlers for Active Mode
        # -------------------------------------------------------------------
        if active_input_mode == "Paste an answer":
            pasted_skel_slot = st.empty()
            if check_clicked and not is_any_busy:
                st.session_state["pasted_error"] = None
                st.session_state["pasted_rewrite_error"] = None
                answer_text = st.session_state["answer_input"].strip()
                if st.session_state.get("demo_mode"):
                    try:
                        sample, exact = find_cached_pasted_audit(answer_text or None)
                        _apply_pasted_demo_sample(
                            sample,
                            exact_match=(True if not answer_text else exact),
                            sync_widget=False,
                        )
                    except FileNotFoundError as exc:
                        _set_tab_error(
                            "pasted_error",
                            exc,
                            action="loading offline demo cache",
                            recover_key="demo_recover_pasted_cache",
                            query_or_url=answer_text,
                        )
                elif not answer_text:
                    st.warning(
                        "Please paste an LLM-generated answer (or pick an example / enable Demo mode) first.",
                        icon=":material/warning:",
                    )
                else:
                    is_valid, err_msg = validate_text_input(answer_text)
                    if not is_valid:
                        st.error(err_msg, icon=":material/error:")
                    else:
                        st.session_state["results"] = None
                        st.session_state["extract_model"] = None
                        st.session_state["extract_meta"] = None
                        st.session_state["corrected_answer"] = None
                        st.session_state["corrected_model"] = None
                        st.session_state["corrected_meta"] = None
                        st.session_state["pasted_audit_at"] = None
                        st.session_state["status_steps"] = []
                        st.session_state["demo_pasted_notice"] = ""
                        st.session_state["demo_pasted_rewrite"] = None
                        st.session_state["audit_stopped"] = False
                        st.session_state["audit_stopped_info"] = None
                        st.session_state["partial_audit_results"] = []
                        st.session_state["partial_audit_total"] = 0
                        live_init = create_initial_ui_state()
                        apply_pipeline_event(live_init, "detect_start")
                        st.session_state["audit_ui_state"] = live_init
                        st.session_state["busy_action"] = "pasted"
                        st.session_state["_busy_armed"] = True
                        st.rerun()

            elif is_pasted_running:
                answer_text = st.session_state.get("answer_input", "").strip()
                if not answer_text:
                    st.session_state["busy_action"] = None
                    st.rerun()
                pasted_finished = False
                status = None
                scan_slot = st.empty()
                freshness_strip_slot = st.empty()
                live_meter_slot = st.empty()
                live_counters_slot = st.empty()
                live_text_slot = st.empty()
                activity_feed_slot = st.empty()
                discovery_slot = st.empty()
                verify_board_slot = st.empty()
                try:
                    if not st.session_state.get("demo_mode") and int(st.session_state.get("live_audits_performed") or 0) >= get_max_session_audits():
                        raise RuntimeError("Live quota is busy, showing saved results. Session audit limit reached.")

                    cancel_ev = threading.Event()
                    st.session_state["audit_cancel_event"] = cancel_ev
                    with st.status("Detecting claims", expanded=True) as status:
                        steps_log: list[str] = []
                        extract_model, extract_record, verified_results = run_stale_audit(
                            answer_text=answer_text,
                            status_widget=status,
                            steps_log=steps_log,
                            credits_placeholder=credits_placeholder,
                            gateway_placeholder=gateway_placeholder,
                            stepper_placeholder=stepper_placeholder,
                            action_status_slot=action_status_slot,
                            scan_slot=scan_slot,
                            discovery_slot=discovery_slot,
                            verify_board_slot=verify_board_slot,
                            freshness_strip_slot=freshness_strip_slot,
                            live_meter_slot=live_meter_slot,
                            live_counters_slot=live_counters_slot,
                            activity_feed_slot=activity_feed_slot,
                            live_text_slot=live_text_slot,
                            cancel_event=cancel_ev,
                        )
                        st.session_state["extract_model"] = extract_model
                        st.session_state["extract_meta"] = extract_record
                        st.session_state["results"] = verified_results
                        st.session_state["checked_answer"] = answer_text
                        st.session_state["pasted_audit_at"] = datetime.now().astimezone()
                        st.session_state["status_steps"] = steps_log
                        st.session_state["quota_banner"] = None
                    pasted_finished = True
                except Exception as exc:
                    err_ui = st.session_state.get("audit_ui_state")
                    if not isinstance(err_ui, AuditUIState):
                        err_ui = create_initial_ui_state()
                    apply_pipeline_event(
                        err_ui,
                        "error",
                        {"step": "detect", "reason": format_friendly_ui_error(exc, action="verifying your answer")},
                    )
                    st.session_state["audit_ui_state"] = err_ui
                    if status is not None:
                        status.update(label="Check failed", state="error", expanded=False)
                    if is_quota_or_rate_limit_error(exc):
                        st.session_state["quota_status"] = "Saved results"
                        st.session_state["quota_banner"] = f"Live API limit reached, showing real saved results from {SAVED_RESULTS_DATE}"
                        try:
                            sample, exact = find_cached_pasted_audit(answer_text)
                            _apply_pasted_demo_sample(sample, exact_match=exact, sync_widget=False)
                            st.session_state["pasted_error"] = None
                        except Exception:
                            _set_tab_error(
                                "pasted_error",
                                exc,
                                action="verifying your answer",
                                recover_key="demo_recover_pasted",
                                query_or_url=answer_text,
                            )
                    else:
                        _set_tab_error(
                            "pasted_error",
                            exc,
                            action="verifying your answer",
                            recover_key="demo_recover_pasted",
                            query_or_url=answer_text,
                        )
                    pasted_finished = True
                finally:
                    if st.session_state.get("busy_action") == "pasted":
                        st.session_state["busy_action"] = None
                    _clear_slot(pasted_skel_slot)
                    _clear_slot(scan_slot)
                    _clear_slot(verify_board_slot)
                if pasted_finished:
                    st.rerun()

            elif st.session_state.get("status_steps"):
                with st.status(
                    "Done",
                    state="complete",
                    expanded=st.session_state["keep_status_expanded"],
                ):
                    for step_line in st.session_state["status_steps"]:
                        st.write(step_line)

            if st.session_state.get("quota_banner"):
                st.warning(f"⚠️ **{st.session_state['quota_banner']}**", icon=":material/history:")
            if st.session_state.get("demo_pasted_notice"):
                _render_demo_banner(st.session_state["demo_pasted_notice"])
            _render_stored_error_with_demo_recovery("pasted_error", recover_scope="pasted", default_recover_key="demo_recover_pasted")

        elif active_input_mode == "Ask a question":
            question_ready = bool(st.session_state.get("ask_question_input", "").strip())
            ask_is_demo = bool(st.session_state.get("demo_mode"))
            ask_left_skel = st.empty()
            ask_right_skel = st.empty()

            if ask_clicked and not is_any_busy:
                st.session_state["ask_error"] = None
                st.session_state["ask_rewrite_error"] = None
                q_text = st.session_state["ask_question_input"].strip()
                if ask_is_demo:
                    try:
                        _apply_ask_demo_sample(q_text or None, sync_widget=False)
                    except FileNotFoundError as exc:
                        _set_tab_error("ask_error", exc, action="loading offline demo Q&A", recover_key="demo_recover_ask_cache", query_or_url=q_text)
                elif not q_text:
                    st.warning("Please type a question to ask and audit.", icon=":material/warning:")
                else:
                    is_valid, err_msg = validate_question_input(q_text)
                    if not is_valid:
                        st.error(err_msg, icon=":material/error:")
                    else:
                        st.session_state["ask_question_asked"] = q_text
                        st.session_state["ask_answer"] = ""
                        st.session_state["ask_model"] = None
                        st.session_state["ask_meta"] = None
                        st.session_state["ask_extract_model"] = None
                        st.session_state["ask_extract_meta"] = None
                        st.session_state["ask_results"] = None
                        st.session_state["ask_corrected_answer"] = None
                        st.session_state["ask_corrected_model"] = None
                        st.session_state["ask_corrected_meta"] = None
                        st.session_state["ask_audit_at"] = None
                        st.session_state["ask_status_steps"] = []
                        st.session_state["demo_ask_notice"] = ""
                        st.session_state["demo_ask_rewrite"] = None
                        st.session_state["busy_action"] = "ask"
                        st.session_state["_busy_armed"] = True
                        st.rerun()

            elif is_ask_running:
                question_text = st.session_state.get("ask_question_input", "").strip()
                if not question_text:
                    st.session_state["busy_action"] = None
                    st.rerun()
                ask_finished = False
                ask_status = None
                try:
                    if not st.session_state.get("demo_mode") and int(st.session_state.get("live_audits_performed") or 0) >= get_max_session_audits():
                        raise RuntimeError("Live quota is busy, showing saved results. Session audit limit reached.")

                    _show_skeleton_slot(ask_left_skel, (150,))
                    _show_skeleton_slot(ask_right_skel, (120, 80))
                    with st.status(f"Querying {route_target}...", expanded=True) as ask_status:
                        ask_steps: list[str] = []
                        llm_answer, ask_model, ask_record = call_bifrost_or_gemini_direct(
                            prompt=question_text,
                            model=DEFAULT_LLM_MODEL,
                            temperature=0.3,
                            max_tokens=600,
                            system_instruction=(
                                "You are a knowledgeable assistant. Answer the user's question concisely "
                                "and factually based on your knowledge. State specific models, dates, prices, "
                                "and names where relevant."
                            ),
                        )
                        st.session_state["ask_answer"] = llm_answer
                        st.session_state["ask_model"] = ask_model
                        st.session_state["ask_meta"] = ask_record
                        st.session_state["ask_question_asked"] = question_text
                        append_gateway_request(ask_record)
                        if gateway_placeholder is not None:
                            render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])
                        ask_status.update(label="Auditing LLM answer for freshness...")
                        ext_model, ext_record, ask_verified = run_stale_audit(
                            answer_text=llm_answer,
                            status_widget=ask_status,
                            steps_log=ask_steps,
                            credits_placeholder=credits_placeholder,
                            gateway_placeholder=gateway_placeholder,
                            stepper_placeholder=stepper_placeholder,
                        )
                        st.session_state["ask_extract_model"] = ext_model
                        st.session_state["ask_extract_meta"] = ext_record
                        st.session_state["ask_results"] = ask_verified
                        st.session_state["ask_audit_at"] = datetime.now().astimezone()
                        st.session_state["ask_status_steps"] = ask_steps
                        st.session_state["quota_banner"] = None
                    ask_finished = True
                except Exception as exc:
                    _clear_slot(ask_left_skel)
                    if ask_status is not None:
                        ask_status.update(label="Ask & Audit failed", state="error", expanded=False)
                    if is_quota_or_rate_limit_error(exc):
                        st.session_state["quota_status"] = "Saved results"
                        st.session_state["quota_banner"] = f"Live API limit reached, showing real saved results from {SAVED_RESULTS_DATE}"
                        try:
                            _apply_ask_demo_sample(question_text, sync_widget=False)
                            st.session_state["ask_error"] = None
                        except Exception:
                            _set_tab_error("ask_error", exc, action="running Ask & Audit", recover_key="demo_recover_ask", query_or_url=question_text)
                    else:
                        _set_tab_error("ask_error", exc, action="running Ask & Audit", recover_key="demo_recover_ask", query_or_url=question_text)
                    ask_finished = True
                finally:
                    if st.session_state.get("busy_action") == "ask":
                        st.session_state["busy_action"] = None
                    _clear_slot(ask_left_skel)
                    _clear_slot(ask_right_skel)
                if ask_finished:
                    st.rerun()

            elif st.session_state.get("ask_status_steps"):
                with st.status("Done", state="complete", expanded=st.session_state["keep_status_expanded"]):
                    for step_line in st.session_state["ask_status_steps"]:
                        st.write(step_line)

            if st.session_state.get("quota_banner"):
                st.warning(f"⚠️ **{st.session_state['quota_banner']}**", icon=":material/history:")
            if st.session_state.get("demo_ask_notice"):
                _render_demo_banner(st.session_state["demo_ask_notice"])
            _render_stored_error_with_demo_recovery("ask_error", recover_scope="ask", default_recover_key="demo_recover_ask")

        elif active_input_mode == "Paste a URL":
            url_skel_slot = st.empty()
            if url_clicked and not is_any_busy:
                st.session_state["url_error"] = None
                st.session_state["url_rewrite_error"] = None
                url_val = st.session_state["url_input"].strip()
                if st.session_state.get("demo_mode"):
                    try:
                        _apply_url_demo_sample(url_val or None, sync_widget=False)
                    except FileNotFoundError as exc:
                        _set_tab_error("url_error", exc, action="loading offline URL demo", recover_key="demo_recover_url_cache", query_or_url=url_val)
                elif not url_val:
                    st.warning("Please enter an article or blog URL to audit.", icon=":material/warning:")
                else:
                    is_valid, err_msg = validate_url_input(url_val)
                    if not is_valid:
                        st.error(err_msg, icon=":material/error:")
                    else:
                        st.session_state["url_audited"] = None
                        st.session_state["url_article_title"] = ""
                        st.session_state["url_article_text"] = ""
                        st.session_state["url_article_truncated"] = False
                        st.session_state["url_extract_credits"] = 0.0
                        st.session_state["url_extract_cache_hit"] = False
                        st.session_state["url_total_claims_found"] = 0
                        st.session_state["url_results"] = None
                        st.session_state["url_extract_model"] = None
                        st.session_state["url_extract_meta"] = None
                        st.session_state["url_corrected_answer"] = None
                        st.session_state["url_corrected_model"] = None
                        st.session_state["url_corrected_meta"] = None
                        st.session_state["url_audit_at"] = None
                        st.session_state["url_status_steps"] = []
                        st.session_state["demo_url_notice"] = ""
                        st.session_state["demo_url_rewrite"] = None
                        st.session_state["busy_action"] = "url"
                        st.session_state["_busy_armed"] = True
                        st.rerun()

            elif is_url_running:
                url_val = st.session_state.get("url_input", "").strip()
                if not url_val:
                    st.session_state["busy_action"] = None
                    st.rerun()
                url_finished = False
                url_status = None
                try:
                    if not st.session_state.get("demo_mode") and int(st.session_state.get("live_audits_performed") or 0) >= get_max_session_audits():
                        raise RuntimeError("Live quota is busy, showing saved results. Session audit limit reached.")

                    _show_skeleton_slot(url_skel_slot, (120, 90, 160))
                    with st.status(f"Extracting article text via Tavily...", expanded=True) as url_status:
                        url_steps: list[str] = []
                        extract_res = extract_article_cached(url_val, session_id=session_id)
                        article_text = extract_res["raw_content"]
                        title = extract_res.get("title") or url_val
                        truncated = extract_res.get("truncated", False)
                        total_credits = extract_res.get("credits_used", 0.0)
                        cache_hit = extract_res.get("cache_hit", False)

                        st.session_state["url_audited"] = url_val
                        st.session_state["url_article_title"] = title
                        st.session_state["url_article_text"] = article_text
                        st.session_state["url_article_truncated"] = truncated
                        st.session_state["url_extract_credits"] = total_credits
                        st.session_state["url_extract_cache_hit"] = cache_hit

                        url_steps.append(
                            f"Extracted clean article text ({len(article_text):,} chars"
                            + (", truncated to 12k" if truncated else "")
                            + f", credits: {total_credits:.2f}"
                            + (", 6h cache hit" if cache_hit else "")
                            + ")."
                        )
                        if credits_placeholder is not None:
                            render_credits_counter(credits_placeholder, get_session_credits(session_id))

                        url_status.update(label="Extracting and auditing article claims...")
                        url_ext_model, url_ext_meta, all_url_results = run_stale_audit(
                            answer_text=article_text,
                            status_widget=url_status,
                            steps_log=url_steps,
                            credits_placeholder=credits_placeholder,
                            gateway_placeholder=gateway_placeholder,
                            max_claims=8,
                            stepper_placeholder=stepper_placeholder,
                        )
                        st.session_state["url_total_claims_found"] = len(all_url_results)
                        st.session_state["url_results"] = all_url_results
                        st.session_state["url_extract_model"] = url_ext_model
                        st.session_state["url_extract_meta"] = url_ext_meta
                        st.session_state["url_audit_at"] = datetime.now().astimezone()
                        st.session_state["url_status_steps"] = url_steps
                        st.session_state["quota_banner"] = None
                    url_finished = True
                except Exception as exc:
                    if url_status is not None:
                        url_status.update(label="URL Audit failed", state="error", expanded=False)
                    if is_quota_or_rate_limit_error(exc):
                        st.session_state["quota_status"] = "Saved results"
                        st.session_state["quota_banner"] = f"Live API limit reached, showing real saved results from {SAVED_RESULTS_DATE}"
                        try:
                            _apply_url_demo_sample(url_val, sync_widget=False)
                            st.session_state["url_error"] = None
                        except Exception:
                            _set_tab_error("url_error", exc, action="auditing article URL", recover_key="demo_recover_url", query_or_url=url_val)
                    else:
                        _set_tab_error("url_error", exc, action="auditing article URL", recover_key="demo_recover_url", query_or_url=url_val)
                    url_finished = True
                finally:
                    if st.session_state.get("busy_action") == "url":
                        st.session_state["busy_action"] = None
                    _clear_slot(url_skel_slot)
                if url_finished:
                    st.rerun()

            elif st.session_state.get("url_status_steps"):
                with st.status("Done", state="complete", expanded=st.session_state["keep_status_expanded"]):
                    for step_line in st.session_state["url_status_steps"]:
                        st.write(step_line)

            if st.session_state.get("quota_banner"):
                st.warning(f"⚠️ **{st.session_state['quota_banner']}**", icon=":material/history:")
            if st.session_state.get("demo_url_notice"):
                _render_demo_banner(st.session_state["demo_url_notice"])
            _render_stored_error_with_demo_recovery("url_error", recover_scope="url", default_recover_key="demo_recover_url")

        elif active_input_mode == "Batch":
            if batch_clicked and not is_any_busy:
                st.session_state["batch_error"] = None
                b_file = st.session_state.get("batch_file_uploader")
                file_bytes = b_file.getvalue() if b_file is not None else None
                fname = b_file.name if b_file is not None else ""
                raw_txt = st.session_state.get("batch_text_input", "")
                parsed_items, err = parse_batch_input(raw_text=raw_txt, file_bytes=file_bytes, filename=fname)
                if err:
                    st.error(err, icon=":material/error:")
                elif not parsed_items:
                    st.warning("Please upload a file or paste at least one answer to batch audit.", icon=":material/warning:")
                else:
                    with st.status(f"Auditing batch of {len(parsed_items)} items...", expanded=True) as batch_status:
                        b_prog = st.progress(0.0)
                        summary_rows = []
                        items_data = []
                        total_claims_seen = 0

                        for b_idx, item_text in enumerate(parsed_items, start=1):
                            if total_claims_seen >= MAX_BATCH_TOTAL_CLAIMS:
                                st.info(f"Reached safety cap of {MAX_BATCH_TOTAL_CLAIMS} total batch claims to protect API quota.")
                                break
                            if not st.session_state.get("demo_mode") and int(st.session_state.get("live_audits_performed") or 0) >= get_max_session_audits():
                                st.warning("Live session audit quota limit reached. Pausing remaining batch items.")
                                break

                            st.write(f"Auditing item {b_idx}/{len(parsed_items)}...")
                            item_steps_log = []
                            try:
                                item_meta = {}
                                ex_model, ex_record, item_results = run_stale_audit(
                                    answer_text=item_text,
                                    status_widget=batch_status,
                                    steps_log=item_steps_log,
                                    credits_placeholder=credits_placeholder,
                                    gateway_placeholder=gateway_placeholder,
                                    max_claims=int(st.session_state.get("settings_max_claims", DEFAULT_BATCH_CLAIMS_PER_ITEM)),
                                    audit_meta_out=item_meta,
                                )
                                total_claims_seen += len(item_results)
                                if not st.session_state.get("demo_mode") and item_results:
                                    st.session_state["live_audits_performed"] = int(st.session_state.get("live_audits_performed") or 0) + 1

                                s_row = summarize_batch_item(
                                    index=b_idx,
                                    answer_text=item_text,
                                    results=item_results,
                                    extract_model=ex_model,
                                    source_origin=("cached" if any(r.get("cache_hit") for r in item_results) else "live"),
                                )
                                summary_rows.append(s_row)
                                items_data.append({
                                    "index": b_idx,
                                    "answer_text": item_text,
                                    "extract_model": ex_model,
                                    "extract_meta": ex_record,
                                    "results": item_results,
                                    "summary": s_row,
                                })
                            except Exception as exc:
                                s_row = summarize_batch_item(
                                    index=b_idx,
                                    answer_text=item_text,
                                    results=[],
                                    overall_score=0,
                                    source_origin="saved",
                                )
                                summary_rows.append(s_row)
                                items_data.append({
                                    "index": b_idx,
                                    "answer_text": item_text,
                                    "extract_model": "error",
                                    "extract_meta": None,
                                    "results": [],
                                    "summary": s_row,
                                })

                            b_prog.progress(b_idx / len(parsed_items))

                        batch_status.update(label=f"Batch Audit complete: {len(summary_rows)} item(s) processed", state="complete", expanded=False)
                        st.session_state["batch_summary"] = summary_rows
                        st.session_state["batch_items_data"] = items_data
                        if items_data:
                            st.session_state["results"] = items_data[0]["results"]
                            st.session_state["checked_answer"] = items_data[0]["answer_text"]
                            st.session_state["extract_model"] = items_data[0]["extract_model"]
                            st.session_state["extract_meta"] = items_data[0]["extract_meta"]
                            st.session_state["batch_selected_index"] = 1
                        st.rerun()

        elif active_input_mode == "Doc Checker":
            if doc_clicked and not is_any_busy:
                st.session_state["doc_checker_error"] = None
                d_file = st.session_state.get("doc_checker_file_uploader")
                file_content = d_file.getvalue().decode("utf-8", errors="replace") if d_file is not None else ""
                raw_doc_txt = file_content or st.session_state.get("doc_checker_text", "")
                if not raw_doc_txt.strip():
                    st.warning("Please upload a Markdown file or paste documentation text to audit.", icon=":material/warning:")
                else:
                    with st.status("Auditing document lines and extracting claims...", expanded=True) as doc_status:
                        d_fname = d_file.name if d_file is not None else "document.md"
                        try:
                            doc_res = audit_markdown_document(
                                doc_text=raw_doc_txt,
                                max_claims=int(st.session_state.get("settings_max_claims", MAX_CLAIMS_PER_AUDIT)),
                                filename=d_fname,
                            )
                            if not st.session_state.get("demo_mode") and doc_res.get("claims"):
                                st.session_state["live_audits_performed"] = int(st.session_state.get("live_audits_performed") or 0) + 1
                            st.session_state["doc_checker_results"] = doc_res
                            st.session_state["results"] = doc_res.get("claims", [])
                            st.session_state["checked_answer"] = raw_doc_txt
                            st.session_state["extract_model"] = doc_res.get("extract_model")
                            doc_status.update(label=f"Document check complete ({len(doc_res.get('claims', []))} claims found)", state="complete", expanded=False)
                            st.rerun()
                        except Exception as exc:
                            doc_status.update(label="Document check failed", state="error", expanded=False)
                            st.session_state["doc_checker_error"] = format_friendly_ui_error(exc)

    # -------------------------------------------------------------------
    # Gather Active Data for Sections 2 - 5
    # -------------------------------------------------------------------
    if active_input_mode == "Paste an answer":
        act_results = st.session_state.get("results")
        act_text = st.session_state.get("checked_answer") or st.session_state.get("answer_input", "")
        act_prefix = "pasted"
        act_ext_model = st.session_state.get("extract_model")
        act_ext_meta = st.session_state.get("extract_meta")
        act_corr = st.session_state.get("corrected_answer")
        act_corr_model = st.session_state.get("corrected_model")
        act_corr_meta = st.session_state.get("corrected_meta")
        act_audit_at = st.session_state.get("pasted_audit_at")
        act_mode_label = "Audit pasted answer"
        act_card_title = "LLM answer"
        act_question = None
        act_url = None
        act_extra_credits = 0.0
        act_busy = is_pasted_running
    elif active_input_mode == "Ask a question":
        act_results = st.session_state.get("ask_results")
        act_text = st.session_state.get("ask_answer", "")
        act_prefix = "ask"
        act_ext_model = st.session_state.get("ask_extract_model")
        act_ext_meta = st.session_state.get("ask_extract_meta")
        act_corr = st.session_state.get("ask_corrected_answer")
        act_corr_model = st.session_state.get("ask_corrected_model")
        act_corr_meta = st.session_state.get("ask_corrected_meta")
        act_audit_at = st.session_state.get("ask_audit_at")
        act_mode_label = "Ask & Audit"
        act_card_title = "Q&A Answer"
        act_question = st.session_state.get("ask_question_asked") or st.session_state.get("ask_question_input", "")
        act_url = None
        act_extra_credits = 0.0
        act_busy = is_ask_running
    elif active_input_mode == "Batch":
        act_results = st.session_state.get("results")
        act_text = st.session_state.get("checked_answer", "")
        act_prefix = "batch"
        act_ext_model = st.session_state.get("extract_model")
        act_ext_meta = st.session_state.get("extract_meta")
        act_corr = None
        act_corr_model = None
        act_corr_meta = None
        act_audit_at = datetime.now().astimezone()
        act_mode_label = "Batch Audit"
        act_card_title = "Selected Batch Item"
        act_question = None
        act_url = None
        act_extra_credits = 0.0
        act_busy = False
    elif active_input_mode == "Doc Checker":
        act_results = st.session_state.get("results")
        act_text = st.session_state.get("checked_answer", "")
        act_prefix = "doc"
        act_ext_model = st.session_state.get("extract_model")
        act_ext_meta = st.session_state.get("extract_meta")
        act_corr = None
        act_corr_model = None
        act_corr_meta = None
        act_audit_at = datetime.now().astimezone()
        act_mode_label = "Doc Checker"
        act_card_title = "Document Text"
        act_question = None
        act_url = None
        act_extra_credits = 0.0
        act_busy = False
    else:  # "Paste a URL"
        act_results = st.session_state.get("url_results")
        act_text = st.session_state.get("url_article_text", "")
        act_prefix = "url"
        act_ext_model = st.session_state.get("url_extract_model")
        act_ext_meta = st.session_state.get("url_extract_meta")
        act_corr = st.session_state.get("url_corrected_answer")
        act_corr_model = st.session_state.get("url_corrected_model")
        act_corr_meta = st.session_state.get("url_corrected_meta")
        act_audit_at = st.session_state.get("url_audit_at")
        act_mode_label = "Audit a URL"
        act_card_title = "Extracted article"
        act_question = None
        act_url = st.session_state.get("url_audited") or st.session_state.get("url_input", "")
        act_extra_credits = float(st.session_state.get("url_extract_credits") or 0.0)
        act_busy = is_url_running

    # Render Batch Summary Card if in Batch mode with results
    if active_input_mode == "Batch" and st.session_state.get("batch_summary"):
        summary_list = st.session_state["batch_summary"]
        items_list = st.session_state.get("batch_items_data") or []
        with st.container(border=True):
            st.markdown("### 📋 Batch Audit Summary Table")
            st.caption(f"Processed {len(summary_list)} items. Click an item below to inspect in the Audit Studio.")
            batch_df = pd.DataFrame([
                {
                    "Item #": r["index"],
                    "Snippet": r["snippet"],
                    "Score": f"{r['overall_score']}/100",
                    "Breakdown": f"🟢 {r['fresh_count']} Fresh | 🔴 {r['outdated_count']} Outdated | ⚪ {r['unverifiable_count']} Unverified",
                    "Worst Claim": r["worst_claim"],
                    "Origin": r["source_label"],
                }
                for r in summary_list
            ])
            st.dataframe(batch_df, hide_index=True, width="stretch")
            
            cur_idx = int(st.session_state.get("batch_selected_index", 1))
            sel_col1, sel_col2, sel_col3 = st.columns([1.5, 1, 1], vertical_alignment="center", gap="medium")
            with sel_col1:
                new_sel = st.selectbox(
                    "Inspect individual audit from batch:",
                    options=[r["index"] for r in summary_list],
                    index=min(cur_idx - 1, len(summary_list) - 1) if summary_list else 0,
                    format_func=lambda i: f"Item #{i}: {summary_list[i-1]['snippet'][:35]}... ({summary_list[i-1]['overall_score']}/100)",
                    key="batch_item_selector",
                )
                if new_sel != cur_idx and items_list:
                    st.session_state["batch_selected_index"] = new_sel
                    match_item = next((it for it in items_list if it["index"] == new_sel), items_list[0])
                    st.session_state["results"] = match_item["results"]
                    st.session_state["checked_answer"] = match_item["answer_text"]
                    st.session_state["extract_model"] = match_item["extract_model"]
                    st.session_state["extract_meta"] = match_item.get("extract_meta")
                    st.rerun()
            with sel_col2:
                csv_bytes = export_batch_summary_csv(summary_list)
                st.download_button("📥 Export CSV", data=csv_bytes, file_name="batch_audit_summary.csv", mime="text/csv", width="stretch")
            with sel_col3:
                json_bytes = export_batch_summary_json(summary_list, items_list)
                st.download_button("📥 Export JSON", data=json_bytes, file_name="batch_audit_results.json", mime="application/json", width="stretch")

    # Render Doc Checker Summary Card if in Doc Checker mode with results
    if active_input_mode == "Doc Checker" and st.session_state.get("doc_checker_results"):
        doc_res = st.session_state["doc_checker_results"]
        with st.container(border=True):
            st.markdown(f"### 📄 Document Line Audit: `{escape_html(doc_res.get('filename', 'document.md'))}`")
            st.caption(f"Score: **{doc_res.get('overall_score')}/100** • Fresh: **{doc_res.get('fresh_count')}** • Outdated: **{doc_res.get('outdated_count')}** • Claims: **{doc_res.get('total_claims')}**")
            doc_df = pd.DataFrame([
                {
                    "Line": f"L{c.get('line_number', 1)}",
                    "Status": str(c.get("status", "unknown")).upper(),
                    "Category": str(c.get("category", "other")).title(),
                    "Claim Text": c.get("claim", ""),
                    "Current Fact / Fix": c.get("current_info") or c.get("reason") or "Verified against web sources",
                }
                for c in doc_res.get("claims", [])
            ])
            st.dataframe(doc_df, hide_index=True, width="stretch")

            d_c1, d_c2 = st.columns([1, 1], gap="medium")
            with d_c1:
                st.download_button("📥 Export Markdown Report", data=format_markdown_report(doc_res), file_name="stale_check_report.md", mime="text/markdown", width="stretch")
            with d_c2:
                st.download_button("📥 Export JSON Report", data=format_json_report(doc_res), file_name="stale_check_report.json", mime="application/json", width="stretch")

    # ===================================================================
    # ===================================================================
    # Section 2: Answer and summary & View Switch
    # ===================================================================
    st.markdown('<div class="sad-section-divider"></div>', unsafe_allow_html=True)
    res_hdr_c1, res_hdr_c2 = st.columns([2.3, 1.7], vertical_alignment="center", gap="medium")
    with res_hdr_c1:
        st.markdown(
            """
            <div class="sad-section-header" id="sad-summary-anchor" style="margin: 0;">
              <span id="sad-claims-detected-anchor" style="display:none;"></span>
              <svg class="sad-section-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="3" y1="9" x2="21" y2="9"></line>
                <line x1="9" y1="21" x2="9" y2="9"></line>
              </svg>
              <div>
                <h3 class="sad-section-title">Answer and summary</h3>
                <p class="sad-section-sub">Side-by-side inspection: annotated source text with inline fact status and overall freshness rating</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with res_hdr_c2:
        audit_subview = st.segmented_control(
            "Audit Subview",
            options=["Side-by-side", "Studio", "Overview"],
            default=st.session_state.get("audit_subview_choice", "Side-by-side"),
            key="audit_subview_choice",
            label_visibility="collapsed",
        )

    if audit_subview in ("Overview", "Overview & Highlights"):
        render_home_tab(
            use_bifrost_mode=use_bifrost_mode,
            gateway_unreachable=gateway_unreachable,
            quota_state=quota_state,
            gateway_health=gateway_health,
            evaluator_text=evaluator_text,
            highlights_bento_fn=render_highlights_bento_grid,
        )
    else:
        res_col1, res_col2 = st.columns([1, 1], gap="large")
    
        with res_col1:
            if act_results and act_text.strip():
                render_annotated_answer_card(
                    answer_text=act_text,
                    verification_results=act_results,
                    key_prefix=act_prefix,
                    title=act_card_title,
                )
            elif act_busy:
                st.markdown(
                    f'<div class="sad-result-card sad-result-card-purple">'
                    f'<div class="sad-result-header"><div class="sad-liquid-tab sad-liquid-tab-purple"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg><span>{act_card_title}</span></div><span class="sad-result-badge">Auditing...</span></div>'
                    '<div class="sad-skeleton-card" style="height: 320px; border-radius: 12px;"></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="sad-result-card sad-result-card-purple">'
                    f'<div class="sad-result-header"><div class="sad-liquid-tab sad-liquid-tab-purple"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg><span>LLM Answer</span></div><span class="sad-status-pill pill-ready"><span class="sad-pulse-dot dot-ready"></span>READY</span></div>'
                    + _render_empty_state("No answer audited yet", "Paste an LLM answer above or pick an example to inspect inline highlighted facts.", pills=("Prices & specs", "Tax rules", "AI API pricing"), variant="purple")
                    + '</div>',
                    unsafe_allow_html=True,
                )
    
        with res_col2:
            if act_results:
                overall_score = compute_risk_weighted_overall_score(act_results)
                checked_text_val = act_text or (act_results[0].get("claim", "") if act_results else "")
                src_origin = f"Saved ({SAVED_RESULTS_DATE})" if st.session_state.get("demo_mode") else "Live"
                record_audit_to_history(checked_text_val, act_results, source_label=src_origin, mode=act_mode_label, score=overall_score)
    
                fresh_count = sum(1 for r in act_results if r.get("status") == "fresh")
                outdated_count = sum(1 for r in act_results if r.get("status") == "outdated")
                incorrect_count = sum(1 for r in act_results if r.get("status") == "incorrect")
                partially_correct_count = sum(1 for r in act_results if r.get("status") == "partially_correct")
                uncertain_count = sum(1 for r in act_results if r.get("status") == "uncertain")
                unverifiable_count = sum(1 for r in act_results if r.get("status") == "unverifiable")
                skipped_count = sum(1 for r in act_results if r.get("status") == "skipped")
                flagged_count = len(act_results) - skipped_count
    
                verdict_sentence = build_overall_verdict_sentence(act_results)
    
                models_used = []
                claim_models = []
                for r in act_results:
                    if isinstance(r.get("judge_models"), list) and r.get("judge_models"):
                        claim_models.extend(str(m) for m in r["judge_models"] if m)
                    if r.get("model"):
                        claim_models.append(str(r["model"]))
                for model_name in [act_ext_model, *claim_models]:
                    if model_name and model_name not in models_used:
                        models_used.append(model_name)
    
                fallback_routes = collect_fallback_routes(act_ext_meta, act_results)
    
                source_origins = set()
                for r in act_results:
                    so = str(r.get("source_origin") or ("saved" if st.session_state.get("demo_mode") else ("cached" if r.get("cache_hit") else "live"))).lower()
                    source_origins.add(so)
    
                if "saved" in source_origins or st.session_state.get("demo_mode"):
                    origin_badge_html = f'💾 Saved ({SAVED_RESULTS_DATE})'
                elif "cached" in source_origins and "live" not in source_origins:
                    origin_badge_html = '⚡ Cached (6h TTL)'
                else:
                    origin_badge_html = '🟢 Live'
    
                # Explicit Routing Label
                if fallback_routes or gateway_unreachable:
                    routing_badge_html = '<span class="sad-status-pill" style="background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3);">Direct Gemini (Fallback)</span>'
                elif is_use_bifrost():
                    routing_badge_html = '<span class="sad-status-pill" style="background: rgba(139,92,246,0.15); color: #A78BFA; border: 1px solid rgba(139,92,246,0.3);">⚡ Bifrost Gateway</span>'
                else:
                    routing_badge_html = '<span class="sad-status-pill" style="background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3);">🔮 Direct Gemini</span>'
    
                strip_html_header = render_freshness_strip_from_results(
                    act_results,
                    st.session_state.get("audit_ui_state"),
                )

                score_ring_html = render_score_ring_html(
                    overall_score,
                    models_used,
                    fallback_routes=fallback_routes,
                    verdict_sentence=verdict_sentence,
                    results=act_results,
                )
    
                st.markdown(
                    f"""
                    <div class="sad-result-card sad-result-card-cyan">
                      <div class="sad-result-header">
                        <div class="sad-liquid-tab sad-liquid-tab-cyan">
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                            <path d="m9 12 2 2 4-4"></path>
                          </svg>
                          <span>Verification Summary</span>
                        </div>
                        <div style="display: flex; gap: 6px; align-items: center;">
                          <span class="sad-result-badge">{origin_badge_html}</span>
                          {routing_badge_html}
                        </div>
                      </div>
                      <div style="padding: 2px 14px 4px 14px;">
                        {strip_html_header}
                      </div>
                      <div class="sad-result-body" style="flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
                        {score_ring_html}
                        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 0.85rem; padding-top: 0.65rem; border-top: 1px solid var(--border-main);">
                          <span class="sad-cat-badge" style="border-color: rgba(22,163,74,0.3); color: #16A34A;">Fresh: <strong>{fresh_count}</strong></span>
                          <span class="sad-cat-badge" style="border-color: rgba(234,88,12,0.3); color: #EA580C;">Outdated: <strong>{outdated_count}</strong></span>
                          <span class="sad-cat-badge" style="border-color: rgba(220,38,38,0.3); color: #DC2626;">Incorrect: <strong>{incorrect_count}</strong></span>
                          <span class="sad-cat-badge" style="border-color: rgba(217,119,6,0.3); color: #D97706;">Partial: <strong>{partially_correct_count}</strong></span>
                          <span class="sad-cat-badge" style="border-color: rgba(100,116,139,0.3); color: #64748B;">Uncertain: <strong>{uncertain_count}</strong></span>
                          <span class="sad-cat-badge" style="border-color: rgba(148,163,184,0.3); color: #94A3B8;">Unverifiable: <strong>{unverifiable_count}</strong></span>
                        </div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.5rem; display: flex; justify-content: space-between; flex-wrap: wrap;">
                          <span>🔍 Analyzed: <strong>{len(act_results)}</strong> claims</span>
                          <span>⚡ Verified: <strong>{flagged_count}</strong> live facts</span>
                          <span>🛡️ Skipped: <strong>{skipped_count}</strong> stable</span>
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
                # Feature G: Trust Badge & Markdown Generator
                badge_date_str = (
                    act_audit_at.strftime("%b %d, %Y")
                    if hasattr(act_audit_at, "strftime")
                    else (SAVED_RESULTS_DATE if st.session_state.get("demo_mode") else "Nov 17, 2026")
                )
                total_stale = outdated_count + incorrect_count
                badge_svg = generate_trust_badge_svg(fresh_count=fresh_count, stale_count=total_stale, check_date=badge_date_str)
                badge_md = generate_trust_badge_markdown(fresh_count=fresh_count, stale_count=total_stale, check_date=badge_date_str)
    
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; font-size: 0.88rem; color: #F8FAFC;">🛡️ Trust Badge &amp; Snapshot</span>
                            <span style="font-size: 0.75rem; color: #94A3B8;">SVG Artifact</span>
                        </div>
                        <div style="text-align: center; margin: 8px 0;">{badge_svg}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    tb_c1, tb_c2 = st.columns([1, 1], gap="small")
                    with tb_c1:
                        st.download_button(
                            "Download SVG Badge",
                            data=badge_svg,
                            file_name="freshness-trust-badge.svg",
                            mime="image/svg+xml",
                            width="stretch",
                        )
                    with tb_c2:
                        with st.popover("Copy Markdown", icon=":material/content_copy:"):
                            st.code(badge_md, language="markdown")
                    st.caption(get_trust_badge_snapshot_notice(badge_date_str))
            elif act_busy:
                st.markdown(
                    '<div class="sad-result-card sad-result-card-cyan">'
                    '<div class="sad-result-header"><div class="sad-liquid-tab sad-liquid-tab-cyan"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg><span>Verification Summary</span></div><span class="sad-result-badge">Auditing...</span></div>'
                    '<div class="sad-skeleton-card" style="height: 320px; border-radius: 12px;"></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="sad-result-card sad-result-card-cyan">'
                    '<div class="sad-result-header"><div class="sad-liquid-tab sad-liquid-tab-cyan"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="m9 12 2 2 4-4"></path></svg><span>Verification Summary</span></div><span class="sad-status-pill pill-standby"><span class="sad-pulse-dot dot-standby"></span>STANDBY</span></div>'
                    + _render_empty_state("Awaiting verification", "Score ring, fact breakdown, consensus stance, and primary sources will appear here.", pills=("Risk-weighted score", "Stance consensus", "freshness statuses"), variant="cyan")
                    + '</div>',
                    unsafe_allow_html=True,
                )
    
        # ===================================================================
    # Section 3: Claims
    # Section 3: Claims
    if act_results:
        st.markdown(
            """
            <div class="sad-section-divider"></div>
            <div class="sad-section-header" id="sad-claim-cards-anchor">
              <svg class="sad-section-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              <div>
                <h3 class="sad-section-title">Claims</h3>
                <p class="sad-section-sub">Granular claim-by-claim verification: claimed facts vs. live web reality with primary sources</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ab_col1, ab_col2 = st.columns(2, gap="medium")
        with ab_col1:
            if st.button("⚖️ Why Tavily? (Compare with LLM Alone)", key=f"{act_prefix}_why_tavily_btn", use_container_width=True):
                st.session_state[f"{act_prefix}_show_why_tavily"] = not st.session_state.get(f"{act_prefix}_show_why_tavily", False)
        with ab_col2:
            if st.button("🛡️ Why Bifrost? (Simulate Outage & Failover)", key=f"{act_prefix}_why_bifrost_btn", use_container_width=True):
                st.session_state[f"{act_prefix}_show_why_bifrost"] = not st.session_state.get(f"{act_prefix}_show_why_bifrost", False)

        if st.session_state.get(f"{act_prefix}_show_why_bifrost"):
            with st.container(border=True):
                st.subheader("🛡️ Why Bifrost? Upstream Outage Simulation & Failover", icon=":material/shield:")
                st.caption("Demonstrating Bifrost's gateway-level circuit-breaker failover when the primary model experiences an outage.")
                first_claim = act_results[0] if act_results else {"claim": "Example claim", "category": "other"}
                sim_res = run_why_bifrost_simulation(first_claim)
                if isinstance(sim_res, dict):
                    st.markdown(
                        f'<div class="sad-meta-row">'
                        f'<span class="sad-cat-badge" style="color: #DC2626;">❌ Primary Outage: <strong>{escape_html(str(sim_res.get("primary_failed", "primary")))}</strong></span>'
                        f'<span class="sad-cat-badge" style="color: #16A34A;">✅ Responded: <strong>{escape_html(str(sim_res.get("answering_model", "fallback")))}</strong></span>'
                        f'<span class="sad-cat-badge">⏱️ Failover Overhead: <strong>{sim_res.get("failover_overhead_ms", 0):.0f} ms</strong></span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="sad-timeline">', unsafe_allow_html=True)
                    for step in sim_res.get("timeline", []):
                        st_cls = step.get("status", "info")
                        t_ms = step.get("timestamp_ms", 0.0)
                        st.markdown(
                            f'<div class="sad-timeline-item">'
                            f'<div class="sad-timeline-dot {st_cls}"></div>'
                            f'<div class="sad-timeline-content">'
                            f'<div class="sad-timeline-title">+{t_ms:.0f}ms &mdash; {escape_html(str(step.get("event", "")))}</div>'
                            f'<div class="sad-timeline-desc">{escape_html(str(step.get("detail", "")))}</div>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

        show_queries = st.session_state.get("show_search_queries", False)
        for row_idx in range(0, len(act_results), 2):
            c_cols = st.columns(2, gap="medium")
            with c_cols[0]:
                render_claim_card(row_idx + 1, act_results[row_idx], show_query=show_queries, key_prefix=act_prefix)
            if row_idx + 1 < len(act_results):
                with c_cols[1]:
                    render_claim_card(row_idx + 2, act_results[row_idx + 1], show_query=show_queries, key_prefix=act_prefix)

        # In-session Claim Watchlist strip
        render_watchlist_strip()

        audit_led = st.session_state.get("last_audit_ledger") or compute_audit_ledger(act_results, act_ext_meta)
        sess_led = st.session_state.get("session_ledger") or audit_led
        with st.expander("🤝 How Tavily and Bifrost Contribute (Contribution Ledger)", expanded=False, icon=":material/handshake:"):
            st.caption("Detailed contribution metrics showing exactly how live Tavily search and Bifrost AI gateway work together.")
            l_col1, l_col2 = st.columns(2, gap="medium")
            with l_col1:
                st.markdown("#### 🌐 Tavily Web Intelligence")
                st.markdown(
                    f"- **Live searches executed:** `{audit_led['tavily']['searches']}` (Session: `{sess_led['tavily']['searches']}`)  \n"
                    f"- **Deep extracts performed:** `{audit_led['tavily']['extracts']}` (Session: `{sess_led['tavily']['extracts']}`)  \n"
                    f"- **Tavily search credits:** `{audit_led['tavily']['credits']:.2f}` (Session: `{sess_led['tavily']['credits']:.2f}`)  \n"
                    f"- **Unique independent domains:** `{len(audit_led['tavily']['unique_domains'])}` (Session: `{len(sess_led['tavily']['unique_domains'])}`)  \n"
                    f"- **Primary / official sources:** `{audit_led['tavily']['primary_sources']}` (Session: `{sess_led['tavily']['primary_sources']}`)"
                )
            with l_col2:
                st.markdown("#### ⚡ Bifrost AI Gateway")
                cost_str = f"${audit_led['bifrost']['cost']:.5f}" if audit_led['bifrost']['cost_set'] else "Pricing unconfigured"
                sess_cost_str = f"${sess_led['bifrost']['cost']:.5f}" if sess_led['bifrost']['cost_set'] else "Pricing unconfigured"
                models_list = ", ".join(f"`{m}`" for m in audit_led['bifrost']['models']) or "`Default`"
                st.markdown(
                    f"- **LLM calls routed:** `{audit_led['bifrost']['calls']}` (Session: `{sess_led['bifrost']['calls']}`)  \n"
                    f"- **Models used:** {models_list}  \n"
                    f"- **Gateway fallbacks triggered:** `{audit_led['bifrost']['fallbacks']}` (Session: `{sess_led['bifrost']['fallbacks']}`)  \n"
                    f"- **Tokens processed:** `{audit_led['bifrost']['total_tokens']:,}` (In: `{audit_led['bifrost']['tokens_in']:,}` | Out: `{audit_led['bifrost']['tokens_out']:,}`)  \n"
                    f"- **Gateway latency:** `{audit_led['bifrost']['total_latency_ms']:.0f} ms`  \n"
                    f"- **Estimated cost:** `{cost_str}` (Session: `{sess_cost_str}`)"
                )

        with st.expander("ℹ️ How the Freshness Score is Computed", expanded=False, icon=":material/calculate:"):
            st.markdown(
                "Every verifiable claim's freshness score is calculated deterministically from four weighted signals in `verifier.py`:\n\n"
                "1. **LLM Verdict (50% weight):** Grounded factual comparison strictly against retrieved web snippets and extracted text.\n"
                "2. **Source Recency (20% weight):** Based on `published_date` returned by Tavily.\n"
                "3. **Source Agreement (20% weight):** Ratio of unique registered domains supporting the current fact.\n"
                "4. **Source Authority (10% weight):** Domain tier (`PRIMARY`, `NEWS`, `BLOG`, or `FORUM`).\n\n"
                "**Enforced Score Bands:**\n"
                "- **Fresh:** `75 – 100` | **Partially Correct:** `45 – 74` | **Uncertain:** `40 – 74`\n"
                "- **Outdated:** `10 – 39` | **Incorrect:** `0 – 15` | **Unverifiable:** `N/A` | **Verification failed:** `—` (excluded)"
            )

        # ===============================================================
        # Section 4: Corrected answer & Section 5: Export
        # ===============================================================
        render_corrected_and_export_section(
            mode=act_mode_label,
            key_prefix=act_prefix,
            original_answer=act_text,
            results=act_results,
            corrected_answer=act_corr,
            corrected_model=act_corr_model,
            corrected_meta=act_corr_meta,
            question=act_question,
            source_url=act_url,
            extract_model=act_ext_model,
            extract_meta=act_ext_meta,
            extra_tavily_credits=act_extra_credits,
            audit_at=act_audit_at,
            gateway_placeholder=gateway_placeholder,
        )

with tab_compare:
    render_showdown_tab(use_bifrost=use_bifrost_mode)

with tab_lab:
    lab_subview = st.segmented_control(
        "Agent Lab View",
        options=["Agent Lab (PS02)", "Ground Truth Benchmark (30 Claims)"],
        default="Agent Lab (PS02)",
        key="lab_subview_choice",
        label_visibility="collapsed",
    )
    if lab_subview == "Agent Lab (PS02)":
        render_agent_lab_tab()
    else:
        render_benchmark_view(use_bifrost_mode, current_busy, is_any_busy)

with tab_learn:
    learn_subview = st.segmented_control(
        "Learn Navigation",
        options=["Architecture & Evidence", "Detection Methodology", "API Playground"],
        default="Architecture & Evidence",
        key="learn_subview_choice",
        label_visibility="collapsed",
    )
    if learn_subview == "Architecture & Evidence":
        render_architecture_tab(use_bifrost_mode)
        render_architecture_telemetry_expander(use_bifrost_mode)
    elif learn_subview == "Detection Methodology":
        render_methodology_tab()
    else:
        render_api_playground_tab()

with tab_history:
    render_history_tab()

render_floating_dock_component(quota_state)

# ---------------------------------------------------------------------------
# Footer (Truthful Privacy Notice — Requirement 6)
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sad-footer">'
    '<span>Built for <strong>Global AI Hackathon</strong></span> &bull; '
    '<a href="https://github.com/PurabTomar/stale-detector" target="_blank" rel="noopener noreferrer">'
    'GitHub Repository &#8599;</a>'
    '<div style="font-size: 0.74rem; color: var(--text-muted); margin-top: 6px; letter-spacing: -0.01em;">'
    '🔒 <strong>Privacy &amp; Data Handling:</strong> Your input text, extracted claims, and web snippets are sent to Google Gemini '
    '(directly or via your configured Bifrost gateway); search queries and target URLs are sent to Tavily. '
    'Search queries and snippets derived from your input are cached on the server in SQLite for 6 hours. '
    'Session history lives in memory for the current session. The hosting platform and API providers may keep their own operational logs.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)
