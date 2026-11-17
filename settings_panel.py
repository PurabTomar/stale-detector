"""
settings_panel.py - Resource estimation, strictness, category filters, and settings logic.

Calculates estimated Tavily credits and LLM calls before running audits,
and manages strictness thresholds, active category filters, and dual-judge options.
"""

from __future__ import annotations

from typing import Any

from settings import DEFAULT_MAX_CLAIMS_PER_AUDIT, MAX_CLAIMS_PER_AUDIT

SUPPORTED_CATEGORIES = [
    "price",
    "version",
    "regulation",
    "company",
    "statistic",
    "other",
]

SUPPORTED_LANGUAGES = [
    "Auto-Detect",
    "English",
    "Hindi (हिन्दी)",
    "Hinglish",
]


def estimate_audit_resources(
    text: str = "",
    max_claims: int = DEFAULT_MAX_CLAIMS_PER_AUDIT,
    dual_judge: bool = False,
    active_categories: list[str] | None = None,
    is_url: bool = False,
    strictness_threshold: int = 40,
) -> dict[str, Any]:
    """
    Estimate Tavily credits and LLM calls before triggering an audit run.

    Returns:
        Dict with keys:
        - estimated_claims: projected number of claims to audit
        - estimated_llm_calls: projected LLM calls (extractor + judges)
        - estimated_tavily_credits: projected Tavily credits
        - is_dual_judge: boolean
        - active_categories: list of enabled categories
        - strictness_threshold: integer sensitivity cutoff
    """
    clean_text = str(text or "").strip()
    char_len = len(clean_text)

    # Heuristic claims projection: approx 1 claim per 200-300 characters, up to max_claims
    if char_len == 0:
        projected_claims = 0
    elif char_len < 150:
        projected_claims = 1
    elif char_len < 400:
        projected_claims = min(max_claims, 2)
    elif char_len < 800:
        projected_claims = min(max_claims, 3)
    else:
        projected_claims = min(max_claims, max(1, char_len // 250))

    # Factor in category filtering: if user disabled some categories, reduce projected claims proportionally
    cats = active_categories if active_categories is not None else SUPPORTED_CATEGORIES
    if len(cats) < len(SUPPORTED_CATEGORIES) and projected_claims > 1:
        ratio = len(cats) / len(SUPPORTED_CATEGORIES)
        projected_claims = max(1, int(round(projected_claims * ratio)))

    # Factor in strictness: higher threshold filters out lower-sensitivity claims
    if strictness_threshold >= 70 and projected_claims > 1:
        projected_claims = max(1, int(round(projected_claims * 0.7)))

    # LLM calls: 1 extraction + (claims * 2 if dual_judge else claims)
    judge_multiplier = 2 if dual_judge else 1
    if projected_claims == 0:
        est_llm_calls = 0
        est_tavily_credits = 0.0
    else:
        est_llm_calls = 1 + (projected_claims * judge_multiplier)
        # Tavily credits: 1 credit per claim for standard search; +1 if URL extraction
        url_extract_credits = 1.0 if is_url else 0.0
        est_tavily_credits = (projected_claims * 1.0) + url_extract_credits

    return {
        "estimated_claims": projected_claims,
        "estimated_llm_calls": est_llm_calls,
        "estimated_tavily_credits": round(est_tavily_credits, 1),
        "is_dual_judge": dual_judge,
        "active_categories": cats,
        "strictness_threshold": strictness_threshold,
    }
