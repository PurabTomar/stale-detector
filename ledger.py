"""
ledger.py - Contribution Ledger tracking Tavily and Bifrost impact per audit and session.
Implements PS06 Requirement 5:
- Tavily: searches, extracts, credits, unique domains, primary sources.
- Bifrost: calls, models, fallbacks, tokens, latency, cost (if pricing configured).
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from verifier import _extract_independent_domain

_PRICING_FILE = Path(__file__).resolve().parent / "lab" / "pricing.json"


def load_pricing_table(path: Path | str | None = None) -> dict[str, Any]:
    """Load model token pricing table from JSON file."""
    p = Path(path) if path else _PRICING_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def create_empty_ledger() -> dict[str, Any]:
    """Return a fresh, zeroed-out contribution ledger structure."""
    return {
        "tavily": {
            "searches": 0,
            "extracts": 0,
            "credits": 0.0,
            "unique_domains": set(),
            "primary_sources": 0,
        },
        "bifrost": {
            "calls": 0,
            "models": set(),
            "fallbacks": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "total_tokens": 0,
            "total_latency_ms": 0.0,
            "cost": 0.0,
            "cost_set": False,
        },
    }


def _clean_model_name(model_str: str | None) -> str:
    """Strip provider prefix (e.g. 'gemini/gemini-2.5-flash' -> 'gemini-2.5-flash')."""
    if not model_str:
        return ""
    m = str(model_str).strip()
    return m.split("/", 1)[-1] if "/" in m else m


def compute_audit_ledger(
    claims_results: list[dict[str, Any]],
    extract_meta: dict[str, Any] | None = None,
    rewrite_meta: dict[str, Any] | None = None,
    gateway_requests: list[dict[str, Any]] | None = None,
    pricing_table: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute the contribution ledger for a single completed audit.
    """
    ledger = create_empty_ledger()
    if pricing_table is None:
        pricing_table = load_pricing_table()

    # --- 1. Tavily Contributions ---
    tavily_credits = 0.0
    searches_count = 0
    extracts_count = 0
    unique_domains: set[str] = set()
    primary_sources = 0

    for r in claims_results:
        # Credits
        cr = float(r.get("tavily_credits_used") or 0.0)
        tavily_credits += cr
        if cr > 0.0 or r.get("search_strategy"):
            searches_count += 1
        if r.get("extracted_url") or r.get("extract_used"):
            extracts_count += 1

        # Sources analysis
        for src in r.get("sources", []):
            if isinstance(src, dict):
                url = src.get("url", "")
                dom = src.get("domain") or _extract_independent_domain(url)
                if dom:
                    unique_domains.add(dom)
                stype = str(src.get("source_type") or "").lower()
                if stype == "primary" or "primary" in str(src.get("title", "")).lower():
                    primary_sources += 1

    ledger["tavily"]["searches"] = searches_count
    ledger["tavily"]["extracts"] = extracts_count
    ledger["tavily"]["credits"] = round(tavily_credits, 2)
    ledger["tavily"]["unique_domains"] = unique_domains
    ledger["tavily"]["primary_sources"] = primary_sources

    # --- 2. Bifrost Contributions ---
    bifrost_calls = 0
    models_used: set[str] = set()
    fallbacks = 0
    tokens_in = 0
    tokens_out = 0
    latency_ms = 0.0
    cost_total = 0.0
    has_valid_pricing = True

    # Collect all gateway telemetry dicts from extraction, judges, rewrite, or sidebar list
    all_metas: list[dict[str, Any]] = []
    if extract_meta and isinstance(extract_meta, dict) and (extract_meta.get("operation") or extract_meta.get("model_used") or extract_meta.get("prompt_tokens")):
        all_metas.append(extract_meta)
    if rewrite_meta and isinstance(rewrite_meta, dict) and (rewrite_meta.get("operation") or rewrite_meta.get("model_used") or rewrite_meta.get("prompt_tokens")):
        all_metas.append(rewrite_meta)

    for r in claims_results:
        # Dual-judge or single-judge telemetry
        telemetry = r.get("judge_telemetry")
        if isinstance(telemetry, dict):
            all_metas.append(telemetry)
        elif isinstance(telemetry, list):
            all_metas.extend([t for t in telemetry if isinstance(t, dict)])

    # If explicit gateway requests exist, prefer them for full ledger
    if gateway_requests:
        candidate_metas = [g for g in gateway_requests if isinstance(g, dict)]
        if len(candidate_metas) >= len(all_metas):
            all_metas = candidate_metas

    for meta in all_metas:
        bifrost_calls += 1
        raw_m = str(meta.get("model_used") or meta.get("model") or "")
        clean_m = _clean_model_name(raw_m)
        if clean_m:
            models_used.add(clean_m)

        if bool(meta.get("fallback_used")):
            fallbacks += 1

        tin = int(meta.get("prompt_tokens") or meta.get("tokens_in") or 0)
        tout = int(meta.get("completion_tokens") or meta.get("tokens_out") or 0)
        tokens_in += tin
        tokens_out += tout
        latency_ms += float(meta.get("latency_ms") or 0.0)

        # Pricing lookup
        rates = pricing_table.get(clean_m) or pricing_table.get(raw_m)
        if isinstance(rates, dict) and rates.get("input_per_million") is not None:
            in_rate = float(rates["input_per_million"])
            out_rate = float(rates["output_per_million"])
            call_cost = ((tin * in_rate) + (tout * out_rate)) / 1_000_000.0
            cost_total += call_cost
        else:
            has_valid_pricing = False

    ledger["bifrost"]["calls"] = bifrost_calls
    ledger["bifrost"]["models"] = models_used
    ledger["bifrost"]["fallbacks"] = fallbacks
    ledger["bifrost"]["tokens_in"] = tokens_in
    ledger["bifrost"]["tokens_out"] = tokens_out
    ledger["bifrost"]["total_tokens"] = tokens_in + tokens_out
    ledger["bifrost"]["total_latency_ms"] = round(latency_ms, 1)
    if bifrost_calls > 0 and has_valid_pricing:
        ledger["bifrost"]["cost"] = round(cost_total, 6)
        ledger["bifrost"]["cost_set"] = True
    else:
        ledger["bifrost"]["cost"] = None
        ledger["bifrost"]["cost_set"] = False

    return ledger


def merge_ledgers(dest_ledger: dict[str, Any], new_ledger: dict[str, Any]) -> dict[str, Any]:
    """Accumulate an audit ledger into a session ledger."""
    merged = deepcopy(dest_ledger)
    # Tavily
    merged["tavily"]["searches"] += new_ledger["tavily"]["searches"]
    merged["tavily"]["extracts"] += new_ledger["tavily"]["extracts"]
    merged["tavily"]["credits"] = round(merged["tavily"]["credits"] + new_ledger["tavily"]["credits"], 2)
    merged["tavily"]["unique_domains"] = set(merged["tavily"]["unique_domains"]) | set(new_ledger["tavily"]["unique_domains"])
    merged["tavily"]["primary_sources"] += new_ledger["tavily"]["primary_sources"]

    # Bifrost
    merged["bifrost"]["calls"] += new_ledger["bifrost"]["calls"]
    merged["bifrost"]["models"] = set(merged["bifrost"]["models"]) | set(new_ledger["bifrost"]["models"])
    merged["bifrost"]["fallbacks"] += new_ledger["bifrost"]["fallbacks"]
    merged["bifrost"]["tokens_in"] += new_ledger["bifrost"]["tokens_in"]
    merged["bifrost"]["tokens_out"] += new_ledger["bifrost"]["tokens_out"]
    merged["bifrost"]["total_tokens"] += new_ledger["bifrost"]["total_tokens"]
    merged["bifrost"]["total_latency_ms"] = round(merged["bifrost"]["total_latency_ms"] + new_ledger["bifrost"]["total_latency_ms"], 1)

    if merged["bifrost"]["cost_set"] and new_ledger["bifrost"]["cost_set"]:
        merged["bifrost"]["cost"] = round((merged["bifrost"]["cost"] or 0.0) + (new_ledger["bifrost"]["cost"] or 0.0), 6)
    else:
        merged["bifrost"]["cost"] = None
        merged["bifrost"]["cost_set"] = False

    return merged
