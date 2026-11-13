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
import time
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from benchmark import (
    format_results_markdown as format_benchmark_markdown,
    load_cached_results as load_benchmark_cache,
    load_dataset as load_benchmark_dataset,
    run_benchmark,
)
from demo_data import (
    find_cached_pasted_audit,
    get_cached_ask_audit,
    get_cached_url_audit,
    load_demo_cache,
)
from extractor import (
    extract_claims,
    get_last_gateway_metrics as get_extract_metrics,
    locate_span_in_text,
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
    format_report_json,
    format_report_markdown,
)
from verifier import (
    classify_source_type,
    clear_tavily_cache,
    compute_risk_weighted_overall_score,
    extract_article_from_url,
    generate_corrected_answer,
    get_last_gateway_metrics as get_rewrite_metrics,
    truncate_snippet_words,
    verify_claims,
)

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
/* Dark Mode and Core Overrides */
[data-theme="dark"], body.dark-theme, .stApp.dark-theme {
    color-scheme: dark !important;
    --bg-page: #0B0F19;
    --bg-card: #131A2A;
    --text-main: #F1F5F9;
    --text-muted: #94A3B8;
    --border-main: #243047;
}
</style>
"""

LIGHT_THEME_CSS = """
<style>
/* Light Mode Theme Overrides */
:root {
    color-scheme: light !important;
    --bg-page: #F7F8FC;
    --bg-card: #FFFFFF;
    --text-main: #0F172A;
    --text-muted: #64748B;
    --border-main: #E5E7EF;
}
.stApp {
    background-color: var(--bg-page) !important;
    color: var(--text-main) !important;
}
section[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border-main) !important;
}
.sad-hero {
    background: linear-gradient(135deg, rgba(91, 91, 240, 0.05) 0%, rgba(139, 92, 246, 0.08) 100%), #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
}
.sad-score-card {
    background: #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
}
.sad-claim-card {
    background: #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
}
.sad-compare-box {
    background: #F8FAFC !important;
    border: 1px solid var(--border-main) !important;
}
.sad-source-chip {
    background: #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
    color: var(--text-main) !important;
}
.sad-snippet-card {
    background: #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
}
.sad-empty-state {
    background: #FFFFFF !important;
    border: 1px dashed var(--border-main) !important;
}
.sad-diff-card {
    background: #FFFFFF !important;
    border: 1px solid var(--border-main) !important;
}
.pill-fresh {
    background: #DCFCE7 !important;
    color: #16A34A !important;
}
.pill-outdated {
    background: #FFEDD5 !important;
    color: #EA580C !important;
}
.pill-uncertain {
    background: #E2E8F0 !important;
    color: #475569 !important;
}
.pill-unverifiable {
    background: #F3F4F6 !important;
    color: #6B7280 !important;
}
div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border-main) !important;
}
div.stDownloadButton > button {
    border-radius: 12px !important;
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


def get_overall_tier(score: int | None) -> tuple[str, str, str]:
    """
    Map a risk-weighted overall freshness score (0-100 or None) to
    (label, ring_color, description).

    Thresholds:
    - Fresh: 80-100
    - Mixed: 40-79
    - Stale: 0-39
    - Unverifiable: None (no scorable claims had reliable evidence)
    """
    if score is None:
        return (
            "Unverifiable",
            "#64748B",
            "No reliable web evidence was found to score these claims yet.",
        )
    if score >= 80:
        return "Fresh", "#22C55E", "Risk-weighted claims align closely with current live web sources."
    if score >= 40:
        return "Mixed", "#F59E0B", "Risk-weighted check shows some details are outdated or uncertain."
    return "Stale", "#EF4444", "Key high-impact facts are outdated compared to today's sources."


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
            "Invalid or missing Tavily API key. Please add TAVILY_API_KEY in "
            "Streamlit Secrets or your .env file."
        )

    if use_bifrost and ("llm_virtual_key" in lower or "x-bf-vk" in lower or "virtual key" in lower):
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
                "Invalid or missing Gemini API key. Please add LLM_API_KEY "
                "(or GEMINI_API_KEY) in Streamlit Secrets or your .env file."
            )
        return (
            "Invalid or missing LLM API key. Please check LLM_API_KEY in "
            "Streamlit Secrets or .env (and your provider keys in Bifrost)."
        )

    # 4. Bifrost gateway down / unreachable (only in local/gateway mode)
    if use_bifrost and (
        ("bifrost" in lower and any(tok in lower for tok in ("down", "unreachable", "offline", "refused", "timeout", "timed out")))
        or any(
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
    pills: tuple[str, ...] = (),
    hint: str = "",
    icon_svg: str = "",
) -> str:
    """Build polished empty-state HTML shared across tabs with inline SVG illustration and 3 prompt cards."""
    friendly_illustration = (
        '<div class="sad-empty-svg-wrap">'
        '<svg viewBox="0 0 160 160" width="130" height="130" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="25" y="25" width="110" height="110" rx="28" fill="#5B5BF0" fill-opacity="0.08"/>'
        '<circle cx="80" cy="80" r="44" fill="#FFFFFF" stroke="#E5E7EF" stroke-width="2"/>'
        '<path d="M62 80L74 92L98 68" stroke="#16A34A" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="114" cy="46" r="7" fill="#5B5BF0"/>'
        '<circle cx="46" cy="114" r="5" fill="#8B5CF6"/>'
        '</svg>'
        '</div>'
    )
    pills_html = (
        '<div class="sad-empty-pills">'
        + "".join(f'<span class="sad-empty-pill">{escape_html(p)}</span>' for p in pills)
        + "</div>"
        if pills
        else ""
    )
    hint_html = f'<div class="sad-empty-hint">{hint}</div>' if hint else ""
    prompts_html = (
        '<div class="sad-empty-prompts-grid">'
        '<div class="sad-empty-prompt-card">'
        '<div class="sad-empty-prompt-badge">Hardware &amp; Pricing</div>'
        '<div>"Apple\'s latest flagship is iPhone 15 Pro Max at $1,199 with A17 Pro..."</div>'
        '</div>'
        '<div class="sad-empty-prompt-card">'
        '<div class="sad-empty-prompt-badge">AI Models &amp; APIs</div>'
        '<div>"OpenAI\'s flagship model is GPT-4 Turbo with 128K context window..."</div>'
        '</div>'
        '<div class="sad-empty-prompt-card">'
        '<div class="sad-empty-prompt-badge">Facts &amp; Leadership</div>'
        '<div>"The Prime Minister of India is Rahul Gandhi, while Sydney is Australia\'s capital..."</div>'
        '</div>'
        '</div>'
    )
    return (
        '<div class="sad-empty-state">'
        f'{friendly_illustration}'
        f'<h4>{escape_html(title)}</h4>'
        f'<p>{body_html}</p>'
        f'{pills_html}'
        f'{hint_html}'
        f'{prompts_html}'
        '</div>'
    )


def _set_tab_error(
    error_state_key: str,
    exc: Exception | str,
    *,
    action: str,
    recover_key: str,
    query_or_url: str = "",
) -> None:
    """Persist a sanitized friendly error so recovery actions work across Streamlit reruns."""
    st.session_state[error_state_key] = {
        "message": format_friendly_ui_error(exc, action=action),
        "recover_key": recover_key,
        "query_or_url": str(query_or_url or ""),
    }


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
    is_test_env = "unittest" in sys.modules or any("test" in arg.lower() for arg in sys.argv)
    default_dark = True if is_test_env else False
    defaults = {
        "busy_action": None,
        "_busy_armed": False,
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
        "benchmark_error": None,
        "pasted_rewrite_error": None,
        "ask_rewrite_error": None,
        "url_rewrite_error": None,
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
        "dark_mode": default_dark,
        "tavily_credits_used": 0.0,
        "gateway_requests": [],
        "last_audit_time_ms": 0.0,
        "session_audit_time_ms": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
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
            '<span><span class="sad-hl-outdated">Orange</span> = Outdated</span>'
            '<span><span class="sad-hl-incorrect">Red</span> = Incorrect</span>'
            '<span><span class="sad-hl-partially-correct">Amber</span> = Partially correct</span>'
            '<span><span class="sad-hl-uncertain">Yellow</span> = Uncertain</span>'
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
        '<div class="sad-annotated-card">'
        '<div class="sad-annotated-head">'
        f'<span class="sad-annotated-title">{escape_html(title)}</span>'
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
        model_chips = "".join(
            f'<span class="sad-model-tag">{escape_html(m)}</span>'
            for m in unique_models
        )
    else:
        model_chips = '<span class="sad-model-tag">Unknown</span>'

    fallback_chip = ""
    if unique_routes:
        fb_text = unique_routes[0]
        if len(unique_routes) > 1:
            fb_text = f"{fb_text} (+{len(unique_routes) - 1} more)"
        fallback_chip = f'<span class="sad-fallback-pill">Fallback used: {escape_html(fb_text)}</span>'

    return (
        '<div class="sad-model-row">'
        f'<span class="sad-model-label">{label_text}</span>'
        f'{model_chips}'
        f'{fallback_chip}'
        '</div>'
    )


def build_overall_verdict_sentence(results: list[dict]) -> str:
    """Build a natural-language summary sentence of verified claim findings (e.g. '1 incorrect claim found')."""
    if not results:
        return "No factual claims verified."
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
) -> str:
    """Build the pure HTML/CSS circular score ring, verdict sentence, and compact model/fallback row."""
    tier_label, ring_color, tier_desc = get_overall_tier(overall_score)
    if overall_score is None:
        ring_fill = 0
        ring_num_text = "N/A"
        ring_denom_text = "no score"
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

    return (
        f'<div class="sad-score-card">'
        f'<div class="sad-ring-wrap" style="background: conic-gradient({ring_color} {ring_fill}%, var(--border-main, #E5E7EF) 0);">'
        f'<div class="sad-ring-inner">'
        f'<span class="sad-ring-num">{ring_num_text}</span>'
        f'<span class="sad-ring-denom">{ring_denom_text}</span>'
        f'<span class="sad-ring-tier" style="color: {ring_color};">{tier_label}</span>'
        f'</div></div>'
        f'<div class="sad-score-meta">'
        f'<h3>Overall freshness: {tier_label}</h3>'
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
    """Render clickable source chips labeled Primary/News/Blog/Forum with optional published dates."""
    chips_parts: list[str] = []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url", "")).strip()
        if not url:
            continue
        domain = escape_html(extract_domain(url))
        title_attr = escape_html(src.get("title") or url)
        safe_url = html.escape(url, quote=True)
        raw_type = str(src.get("source_type") or "").strip()
        src_type = (
            raw_type
            if raw_type in {"Primary", "News", "Blog", "Forum"}
            else classify_source_type(url, entity_domains=entity_domains)
        )
        type_cls = f"sad-src-{src_type.lower()}"
        pub_date = str(src.get("published_date") or "").strip()
        date_span = f'<span class="sad-src-date">&bull; {escape_html(pub_date)}</span>' if pub_date else ""
        chips_parts.append(
            f'<a class="sad-source-chip" href="{safe_url}" target="_blank" '
            f'rel="noopener noreferrer" title="{title_attr}">'
            f'<span class="sad-src-type {type_cls}">{escape_html(src_type)}</span>'
            f'{domain}{date_span} &#8599;</a>'
        )
    if not chips_parts:
        return '<div class="sad-card-footer">No source links available.</div>'
    return '<div class="sad-chips-row">' + "".join(chips_parts) + "</div>"


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
    is_unverifiable = status == "unverifiable" or raw_score is None
    score: int | None = None if is_unverifiable else max(0, min(100, int(raw_score)))
    bar_width = 0 if score is None else score
    score_display = "N/A" if score is None else f"{score}/100"

    category = str(item.get("category", "other")).lower()
    cat_icon = CATEGORY_ICONS.get(category, CATEGORY_ICONS["other"])
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
                top_src_str = f' (<a href="{html.escape(src_url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape_html(src_title)}</a>)'
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
    card_html = (
        f'<div id="{escape_html(card_id)}" class="sad-claim-card">'
        f'<div class="sad-claim-header">'
        f'<div class="sad-claim-badges">'
        f'<span class="sad-pill {meta["pill_class"]}">{meta["label"]}</span>'
        f'<span class="sad-cat-badge">{cat_icon}{escape_html(category)}</span>'
        f'<span class="sad-meta-pill">{escape_html(confidence.title())} confidence</span>'
        f'<span class="sad-meta-pill {risk_cls}">{escape_html(risk_level.title())} risk</span>'
        f'{agreement_badge_html}'
        f'{dual_judge_badge_html}'
        f'</div>'
        f'<div class="sad-score-bar-wrap">'
        f'<div class="sad-score-track">'
        f'<div class="sad-score-fill" style="width: {bar_width}%; background: {meta["color"]};"></div>'
        f'</div>'
        f'<span class="sad-score-text">{score_display}</span>'
        f'</div></div>'
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
                safe_s_url = html.escape(s_url, quote=True)
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
) -> tuple[str | None, dict | None, list[dict]]:
    """
    Reuse the shared extractor + parallel verifier pipeline on `answer_text`, streaming live
    progress (including an st.progress bar), sidebar Tavily credits, and the Bifrost Gateway panel.
    """
    audit_t0 = time.perf_counter()

    def _finalize_audit_timing() -> None:
        elapsed_ms = round((time.perf_counter() - audit_t0) * 1000.0, 1)
        st.session_state["last_audit_time_ms"] = elapsed_ms
        st.session_state["session_audit_time_ms"] = round(
            float(st.session_state.get("session_audit_time_ms") or 0.0) + elapsed_ms,
            1,
        )
        render_gateway_panel(gateway_placeholder, st.session_state["gateway_requests"])

    st.write("Extracting claims...")
    claims, extract_model = extract_claims(answer_text)
    total_extracted = len(claims)
    if max_claims is not None and max_claims > 0 and total_extracted > max_claims:
        claims = claims[:max_claims]
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

    if max_claims is not None and total_extracted > len(claims):
        extract_msg = (
            f"Extracted {total_extracted} claim(s) — auditing first {len(claims)} ({extract_model})"
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
        _finalize_audit_timing()
        status_widget.update(
            label="Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        )
        return extract_model, extract_record, []

    total = len(claims)
    status_widget.update(
        label="Searching live sources",
        state="running",
    )
    live_step = st.empty()
    live_step.write(f"Searching live sources for {total} claim(s)...")
    progress_bar = st.progress(
        0.0,
        text=f"Searching live sources (0/{total} claims)...",
    )

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

        _append_claim_gateway_telemetry(idx, result, gateway_placeholder, refresh_panel=True)

        if progress_bar is not None and total_count > 0:
            frac = min(1.0, max(0.0, float(idx) / float(total_count)))
            prog_label = (
                f"Verified {idx}/{total_count} claims"
                if idx < total_count
                else f"Verified {total_count}/{total_count} claims — finalizing..."
            )
            progress_bar.progress(frac, text=prog_label)

        if idx < total_count:
            status_widget.update(
                label="Searching live sources",
                state="running",
            )
            live_step.write(f"Searching live sources ({idx + 1}/{total_count})...")
        else:
            status_widget.update(label="Judging", state="running")
            live_step.write("Judging...")

        res_score = result.get("score")
        res_score_str = "N/A" if res_score is None else f"{res_score}/100"
        step_text = (
            f"Claim {idx}/{total_count}: {result.get('status', 'uncertain').upper()} "
            f"({res_score_str})"
        )
        steps_log.append(step_text)

    dual_enabled = bool(st.session_state.get("dual_judge_mode", False))
    verify_kwargs: dict = {"progress_callback": on_progress}
    if dual_enabled and _callable_accepts_kwarg(verify_claims, "dual_judge"):
        verify_kwargs["dual_judge"] = True

    verified_results = verify_claims(claims, **verify_kwargs)
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
            if _append_claim_gateway_telemetry(b_idx, r, gateway_placeholder, refresh_panel=False):
                batch_added = True
        if batch_added:
            render_gateway_panel(
                gateway_placeholder,
                st.session_state["gateway_requests"],
            )

    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"Completed {total}/{total} claim checks")

    _finalize_audit_timing()
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

    st.markdown(
        render_score_ring_html(
            overall_score,
            models_used,
            fallback_routes=fallback_routes,
            verdict_sentence=verdict_sentence,
        ),
        unsafe_allow_html=True,
    )

    with st.grid(6, min_column_width=85, gap="small"):
        st.metric("Fresh", fresh_count, border=True, height="stretch")
        st.metric("Outdated", outdated_count, border=True, height="stretch")
        st.metric("Incorrect", incorrect_count, border=True, height="stretch")
        st.metric("Partially correct", partially_correct_count, border=True, height="stretch")
        st.metric("Uncertain", uncertain_count, border=True, height="stretch")
        st.metric("Unverifiable", unverifiable_count, border=True, height="stretch")

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

    st.divider()
    hdr_col, action_col = st.columns([2.2, 1], vertical_alignment="center", gap="medium")
    with hdr_col:
        st.subheader("Generate corrected answer", icon=":material/auto_fix_high:")
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
                        ch_top_src = f' (<a href="{html.escape(src_u, quote=True)}" target="_blank" rel="noopener noreferrer">{escape_html(src_t)}</a>)'
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
            disabled=is_any_busy,
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
            disabled=is_any_busy,
            on_click="ignore",
            width="stretch",
        )


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
        if use_bifrost_mode:
            vk_ok = bool(_get_config_value("LLM_VIRTUAL_KEY"))
            vk_status = "Configured" if vk_ok else "Not set"
            vk_line = f"**Virtual key (`x-bf-vk`):** {vk_status}  \n"

        st.caption(
            f"**Routing mode:** {routing_label}  \n"
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
            "**Tavily cache:** `6h temp SQLite TTL`"
        )

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

    # 3. Demo mode toggle
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
        gateway_health = check_gateway_health(timeout=0.5)
        dashboard_url = get_gateway_dashboard_url()
        render_gateway_health_card(gateway_health, dashboard_url, use_bifrost=True)

        if demo_mode_on:
            st.caption("Demo mode active — live gateway calls paused.")
        elif not gateway_health.get("healthy"):
            st.caption(
                "Bifrost looks unreachable on port 8080. Start Bifrost locally, enable Demo mode, "
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
# Hero header
# ---------------------------------------------------------------------------
routing_badge_text = "Routed via Bifrost" if use_bifrost_mode else "Direct Gemini API"
demo_hero_badge = (
    '<span class="sad-tech-badge sad-tech-demo"><span class="sad-dot-green"></span>Demo mode (offline)</span>'
    if demo_mode_on
    else ""
)
bifrost_badge_html = (
    f'<span class="sad-tech-badge"><span class="sad-dot"></span>{routing_badge_text}</span>'
    if use_bifrost_mode
    else ""
)
st.markdown(
    '<div class="sad-hero">'
    '<div class="sad-hero-top">'
    '<div>'
    '<h1 class="sad-hero-title">Don&#39;t trust AI blindly. Check its expiry date.</h1>'
    '<p class="sad-hero-tagline">'
    'Verify time-sensitive AI answers against live web sources, compute explainable freshness scores, and rewrite stale facts.'
    '</p>'
    '</div>'
    '<div class="sad-hero-badges">'
    f'{demo_hero_badge}'
    '<span class="sad-tech-badge"><span class="sad-dot"></span>Powered by Tavily</span>'
    '<span class="sad-tech-badge"><span class="sad-dot"></span>Multi-model verified</span>'
    f'{bifrost_badge_html}'
    '</div>'
    '</div>'
    '<div class="sad-hero-stats">'
    '<span class="sad-stat-chip">🌐 Live web sources</span>'
    '<span class="sad-stat-chip">⚖️ Dual-judge verification</span>'
    '<span class="sad-stat-chip">⚡ 6h smart cache</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Four-tab main workspace:
# 1. "Audit pasted answer" (annotated answer view + two-column flow)
# 2. "Ask & Audit" (ask question via Bifrost -> show answer -> auto-audit + highlight)
# 3. "Audit a URL" (Tavily Extract article -> first 8 claims -> annotated view + report)
# 4. "Benchmark" (20 labelled claims evaluation, cached in benchmark/results.json)
# ---------------------------------------------------------------------------
tab_pasted, tab_ask, tab_url, tab_benchmark = st.tabs(
    ["Audit pasted answer", "Ask & Audit", "Audit a URL", "Benchmark"]
)

route_target = "Bifrost" if use_bifrost_mode else "Gemini"

with tab_pasted:
    is_pasted_running = current_busy == "pasted"
    left_col, right_col = st.columns([1, 1.35], gap="large")

    with left_col:
        char_count = len(st.session_state.get("answer_input", ""))
        st.markdown(
            f'<div class="sad-input-header">'
            f'<span class="sad-input-title">Input answer</span>'
            f'<span class="sad-char-counter">{char_count:,} chars</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sad-examples-label">Try an example:</div>', unsafe_allow_html=True)
        chip_cols = st.columns(len(DEMO_EXAMPLES), gap="small")
        for ex_idx, ex_item in enumerate(DEMO_EXAMPLES):
            with chip_cols[ex_idx]:
                short_name = ex_item["label"].split()[0]
                if st.button(
                    short_name,
                    key=f"pasted_chip_{ex_idx}",
                    help=f"{ex_item['label']}: {ex_item['text'][:80]}...",
                    disabled=is_any_busy,
                    width="stretch",
                ):
                    load_demo_example(ex_item["text"])

        st.text_area(
            "LLM-generated answer",
            key="answer_input",
            height=210,
            label_visibility="collapsed",
            placeholder=(
                "Paste an LLM-generated answer here or click one of the example chips above..."
            ),
        )

        btn_col1, btn_col2 = st.columns([3, 1], gap="small")
        with btn_col1:
            check_clicked = st.button(
                "Check Freshness",
                key="check_freshness_btn",
                type="primary",
                icon=("spinner" if is_pasted_running else ":material/verified:"),
                disabled=is_any_busy,
                width="stretch",
            )
        with btn_col2:
            st.button(
                "Clear",
                key="clear_workspace_btn",
                type="secondary",
                icon=":material/delete_outline:",
                on_click=clear_workspace,
                disabled=is_any_busy,
                width="stretch",
            )

    with right_col:
        st.subheader("Verification results", icon=":material/fact_check:")
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
                    "Please paste an LLM-generated answer (or pick a sidebar example / enable Demo mode) first.",
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
                st.session_state["demo_pasted_notice"] = ""
                st.session_state["demo_pasted_rewrite"] = None
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
            try:
                _show_skeleton_slot(pasted_skel_slot, (128, 82, 165))
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
                pasted_finished = True
            except Exception as exc:
                if status is not None:
                    status.update(label="Check failed", state="error", expanded=False)
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
            if pasted_finished:
                st.rerun()

        # Render collapsed status trail on subsequent reruns (e.g., after rewriting answer).
        elif st.session_state.get("status_steps"):
            with st.status(
                "Done",
                state="complete",
                expanded=st.session_state["keep_status_expanded"],
            ):
                for step_line in st.session_state["status_steps"]:
                    st.write(step_line)

        if st.session_state.get("demo_pasted_notice"):
            _render_demo_banner(st.session_state["demo_pasted_notice"])

        _render_stored_error_with_demo_recovery(
            "pasted_error",
            recover_scope="pasted",
            default_recover_key="demo_recover_pasted",
        )

        # Render stored verification results (score ring, KPI cards, annotated answer, claim cards).
        if st.session_state.get("results") is not None or not st.session_state.get("pasted_error"):
            render_audit_results(
                results=st.session_state.get("results"),
                extract_model=st.session_state.get("extract_model"),
                extract_meta=st.session_state.get("extract_meta"),
                key_prefix="pasted",
                just_clicked=bool(
                    is_pasted_running or (check_clicked and st.session_state.get("results") is not None)
                ),
                empty_body=(
                    f"Paste an LLM answer on the left or choose a sidebar sample, then click "
                    f"<strong>Check Freshness</strong> ({route_target})."
                ),
                annotated_text=st.session_state.get("checked_answer"),
                annotated_title="Annotated answer",
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
        gateway_placeholder=gateway_placeholder,
    )

with tab_ask:
    is_ask_running = current_busy == "ask"
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
            icon=("spinner" if is_ask_running else ":material/send:"),
            disabled=is_any_busy,
            width="stretch",
        )
    with ask_clr_col:
        st.button(
            "Clear",
            key="ask_clear_btn",
            icon=":material/delete_outline:",
            on_click=clear_ask_workspace,
            disabled=is_any_busy,
            width="stretch",
        )

    ask_left, ask_right = st.columns([1, 1.35], gap="large")
    question_ready = bool(st.session_state.get("ask_question_input", "").strip())
    ask_is_demo = bool(st.session_state.get("demo_mode"))

    with ask_left:
        st.subheader("LLM answer", icon=":material/chat_bubble_outline:")
        ask_left_skel = st.empty()
        ask_answer_slot = st.empty()

        if is_ask_running or (ask_clicked and question_ready and not ask_is_demo and not is_any_busy):
            _show_skeleton_slot(ask_left_skel, (150,))
        elif not (ask_clicked and ask_is_demo and not is_any_busy):
            saved_ask_answer = st.session_state.get("ask_answer", "")
            if saved_ask_answer:
                render_ask_answer_card(
                    placeholder=ask_answer_slot,
                    answer_text=saved_ask_answer,
                    verification_results=st.session_state.get("ask_results"),
                    question_text=st.session_state.get("ask_question_asked"),
                    answer_model=st.session_state.get("ask_model"),
                    answer_meta=st.session_state.get("ask_meta"),
                    audit_error=bool(
                        st.session_state.get("ask_error")
                        and st.session_state.get("ask_results") is None
                    ),
                )
            else:
                ask_answer_slot.markdown(
                    _render_empty_state(
                        "Ask any question",
                        f"Enter a question above and click <strong>Ask &amp; Audit</strong> "
                        f"to inspect live {route_target} output with inline claim highlights.",
                        pills=("Product lineups", "API rates", "Release versions", "Policy rules"),
                        hint="Or enable <strong>Demo mode</strong> to replay a pre-cached Q&amp;A audit.",
                    ),
                    unsafe_allow_html=True,
                )

    with ask_right:
        st.subheader("Verification audit", icon=":material/fact_check:")
        ask_right_skel = st.empty()

        if ask_clicked and not is_any_busy:
            st.session_state["ask_error"] = None
            st.session_state["ask_rewrite_error"] = None
            question_text = st.session_state["ask_question_input"].strip()
            if ask_is_demo:
                try:
                    demo_ask = _apply_ask_demo_sample(question_text or None, sync_widget=False)
                    render_ask_answer_card(
                        placeholder=ask_answer_slot,
                        answer_text=demo_ask["answer_text"],
                        verification_results=st.session_state.get("ask_results"),
                        question_text=st.session_state.get("ask_question_asked"),
                        answer_model=st.session_state.get("ask_model"),
                        answer_meta=st.session_state.get("ask_meta"),
                    )
                except FileNotFoundError as exc:
                    _set_tab_error(
                        "ask_error",
                        exc,
                        action="loading offline Ask & Audit demo",
                        recover_key="demo_recover_ask_cache",
                        query_or_url=question_text,
                    )
            elif not question_text:
                _clear_slot(ask_left_skel)
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
                st.session_state["demo_ask_notice"] = ""
                st.session_state["demo_ask_rewrite"] = None
                st.session_state["busy_action"] = "ask"
                st.session_state["_busy_armed"] = True
                st.rerun()

        elif is_ask_running:
            question_text = str(
                st.session_state.get("ask_question_asked")
                or st.session_state.get("ask_question_input")
                or ""
            ).strip()
            if not question_text:
                _clear_slot(ask_left_skel)
                st.session_state["busy_action"] = None
                st.rerun()

            ask_finished = False
            ask_status = None
            try:
                _show_skeleton_slot(ask_right_skel, (128, 82, 165))
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
                    _clear_slot(ask_left_skel)
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
                ask_finished = True

            except Exception as exc:
                _clear_slot(ask_left_skel)
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
                _set_tab_error(
                    "ask_error",
                    exc,
                    action="running Ask & Audit",
                    recover_key="demo_recover_ask",
                    query_or_url=question_text,
                )
                ask_finished = True
            finally:
                if st.session_state.get("busy_action") == "ask":
                    st.session_state["busy_action"] = None
                _clear_slot(ask_left_skel)
                _clear_slot(ask_right_skel)
            if ask_finished:
                st.rerun()

        elif st.session_state.get("ask_status_steps"):
            with st.status(
                "Done",
                state="complete",
                expanded=st.session_state["keep_status_expanded"],
            ):
                for step_line in st.session_state["ask_status_steps"]:
                    st.write(step_line)

        if st.session_state.get("demo_ask_notice"):
            _render_demo_banner(st.session_state["demo_ask_notice"])

        _render_stored_error_with_demo_recovery(
            "ask_error",
            recover_scope="ask",
            default_recover_key="demo_recover_ask",
        )

        if st.session_state.get("ask_results") is not None or not st.session_state.get("ask_error"):
            render_audit_results(
                results=st.session_state.get("ask_results"),
                extract_model=st.session_state.get("ask_extract_model"),
                extract_meta=st.session_state.get("ask_extract_meta"),
                key_prefix="ask",
                just_clicked=bool(
                    is_ask_running or (ask_clicked and st.session_state.get("ask_results") is not None)
                ),
                empty_title="Ready to ask & audit",
                empty_body=(
                    f"Type any question above and click <strong>Ask &amp; Audit</strong> to fetch "
                    f"an LLM response via {route_target} and highlight stale facts automatically."
                ),
                empty_pills=("Live LLM answer", "Auto-extract claims", "Inline highlights", "Diff & export"),
                extra_models=[st.session_state.get("ask_model")],
                extra_metas=[st.session_state.get("ask_meta")],
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
        gateway_placeholder=gateway_placeholder,
    )

with tab_url:
    is_url_running = current_busy == "url"
    st.caption(
        "Paste an article or blog URL below. Tavily Extract pulls the clean text, audits the first "
        "8 time-sensitive claims in parallel, highlights facts inline, and exports the full report."
    )

    u_col, u_btn_col, u_clr_col = st.columns(
        [4, 1.3, 0.75],
        vertical_alignment="bottom",
        gap="small",
    )
    with u_col:
        st.text_input(
            "Article or blog URL",
            key="url_input",
            label_visibility="collapsed",
            placeholder="https://example.com/blog/python-release-guide",
        )
    with u_btn_col:
        url_clicked = st.button(
            "Audit URL",
            key="audit_url_btn",
            type="primary",
            icon=("spinner" if is_url_running else ":material/travel_explore:"),
            disabled=is_any_busy,
            width="stretch",
        )
    with u_clr_col:
        st.button(
            "Clear",
            key="url_clear_btn",
            icon=":material/delete_outline:",
            on_click=clear_url_workspace,
            disabled=is_any_busy,
            width="stretch",
        )

    url_skel_slot = st.empty()

    if url_clicked and not is_any_busy:
        st.session_state["url_error"] = None
        st.session_state["url_rewrite_error"] = None
        url_val = st.session_state["url_input"].strip()
        if st.session_state.get("demo_mode"):
            try:
                _apply_url_demo_sample(url_val or None, sync_widget=False)
            except FileNotFoundError as exc:
                _set_tab_error(
                    "url_error",
                    exc,
                    action="loading offline URL demo",
                    recover_key="demo_recover_url_cache",
                    query_or_url=url_val,
                )
        elif not url_val:
            st.warning(
                "Please enter an article or blog URL to audit.",
                icon=":material/warning:",
            )
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
            _show_skeleton_slot(url_skel_slot, (68, 128, 165, 165))
            with st.status("Extracting article via Tavily Extract", expanded=True) as url_status:
                url_steps: list[str] = []
                st.write("Fetching clean article text via Tavily Extract...")
                article_info = extract_article_from_url(url_val)

                ext_credits = float(article_info.get("credits_used") or 0.0)
                if ext_credits > 0:
                    st.session_state["tavily_credits_used"] = round(
                        float(st.session_state.get("tavily_credits_used", 0.0)) + ext_credits,
                        2,
                    )
                    render_sidebar_credits(
                        credits_placeholder,
                        st.session_state["tavily_credits_used"],
                    )

                resolved_url = str(article_info.get("url") or url_val).strip()
                article_text = str(article_info.get("content") or "")
                st.session_state["url_audited"] = resolved_url
                st.session_state["url_article_title"] = str(article_info.get("title") or "").strip()
                st.session_state["url_article_text"] = article_text
                st.session_state["url_article_truncated"] = bool(article_info.get("truncated"))
                st.session_state["url_extract_credits"] = ext_credits
                st.session_state["url_extract_cache_hit"] = bool(article_info.get("cache_hit"))

                if article_info.get("cache_hit"):
                    ext_msg = f"Extracted article ({len(article_text):,} chars, 6h cache hit — 0 credits)"
                else:
                    ext_msg = (
                        f"Extracted article ({len(article_text):,} chars, "
                        f"{format_credits(ext_credits)} Tavily credit)"
                    )
                url_steps.append(ext_msg)
                st.write(f":material/check: {ext_msg}")

                url_status.update(label="Extracting claims", state="running")
                url_meta_out: dict = {}
                ext_model, ext_record, url_verified = run_stale_audit(
                    answer_text=article_text,
                    status_widget=url_status,
                    steps_log=url_steps,
                    credits_placeholder=credits_placeholder,
                    gateway_placeholder=gateway_placeholder,
                    max_claims=8,
                    audit_meta_out=url_meta_out,
                )

                st.session_state["url_extract_model"] = ext_model
                st.session_state["url_extract_meta"] = ext_record
                st.session_state["url_results"] = url_verified
                st.session_state["url_total_claims_found"] = int(
                    url_meta_out.get("total_claims_extracted") or len(url_verified)
                )
                st.session_state["url_audit_at"] = datetime.now().astimezone()
                st.session_state["url_status_steps"] = url_steps
            url_finished = True

        except Exception as exc:
            if url_status is not None:
                url_status.update(label="URL audit failed", state="error", expanded=False)
            _set_tab_error(
                "url_error",
                exc,
                action="extracting the URL",
                recover_key="demo_recover_url",
                query_or_url=url_val,
            )
            url_finished = True
        finally:
            if st.session_state.get("busy_action") == "url":
                st.session_state["busy_action"] = None
            _clear_slot(url_skel_slot)
        if url_finished:
            st.rerun()

    elif st.session_state.get("url_status_steps"):
        with st.status(
            "Done",
            state="complete",
            expanded=st.session_state["keep_status_expanded"],
        ):
            for step_line in st.session_state["url_status_steps"]:
                st.write(step_line)

    if st.session_state.get("demo_url_notice"):
        _render_demo_banner(st.session_state["demo_url_notice"])

    _render_stored_error_with_demo_recovery(
        "url_error",
        recover_scope="url",
        default_recover_key="demo_recover_url",
    )

    saved_url = st.session_state.get("url_audited")
    saved_url_results = st.session_state.get("url_results")
    saved_article_text = st.session_state.get("url_article_text", "")
    if saved_url and saved_url_results is not None:
        url_title = st.session_state.get("url_article_title") or extract_domain(saved_url)
        safe_href = html.escape(saved_url, quote=True)
        cache_badge = (
            "6h cache hit (0 credits)"
            if st.session_state.get("url_extract_cache_hit")
            else f"{format_credits(float(st.session_state.get('url_extract_credits') or 0.0))} extract credit"
        )
        st.markdown(
            '<div class="sad-url-source-bar">'
            '<div>'
            f'<strong>{escape_html(url_title)}</strong> &bull; '
            f'<a href="{safe_href}" target="_blank" rel="noopener noreferrer">{escape_html(saved_url)} &#8599;</a>'
            '</div>'
            f'<div class="sad-annotated-hint">{len(saved_article_text):,} chars &bull; Tavily Extract: {escape_html(cache_badge)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        total_found = int(st.session_state.get("url_total_claims_found") or 0)
        if total_found > 8:
            st.info(
                f"Extracted {total_found} claims from the article — auditing the first 8 claims.",
                icon=":material/info:",
            )
        if st.session_state.get("url_article_truncated"):
            st.caption("Article text was truncated to the first 12,000 characters for focused auditing.")

    if saved_url_results is not None or not st.session_state.get("url_error"):
        render_audit_results(
            results=saved_url_results,
            extract_model=st.session_state.get("url_extract_model"),
            extract_meta=st.session_state.get("url_extract_meta"),
            key_prefix="url",
            just_clicked=bool(
                is_url_running or (url_clicked and saved_url_results is not None)
            ),
            annotated_text=saved_article_text,
            annotated_title="Annotated article",
            empty_title="Ready to audit a URL",
            empty_body=(
                "Paste an article or blog URL above and click <strong>Audit URL</strong> to extract "
                "clean text with Tavily Extract and audit its first 8 time-sensitive claims."
            ),
            empty_pills=("Tavily Extract", "First 8 claims", "Annotated article", "Export JSON / MD"),
        )

    render_corrected_and_export_section(
        mode="Audit a URL",
        key_prefix="url",
        original_answer=saved_article_text,
        source_url=saved_url,
        extra_tavily_credits=float(st.session_state.get("url_extract_credits") or 0.0),
        results=saved_url_results,
        corrected_answer=st.session_state.get("url_corrected_answer"),
        corrected_model=st.session_state.get("url_corrected_model"),
        corrected_meta=st.session_state.get("url_corrected_meta"),
        extract_model=st.session_state.get("url_extract_model"),
        extract_meta=st.session_state.get("url_extract_meta"),
        audit_at=st.session_state.get("url_audit_at"),
        gateway_placeholder=gateway_placeholder,
    )

with tab_benchmark:
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
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="sad-footer">'
    '<span>Built for <strong>Global AI Hackathon</strong></span> &bull; '
    '<a href="https://github.com/PurabTomar/stale-detector" target="_blank" rel="noopener noreferrer">'
    'GitHub Repository &#8599;</a>'
    '</div>',
    unsafe_allow_html=True,
)
