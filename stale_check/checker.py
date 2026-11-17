"""
stale_check/checker.py - Core engine for auditing Markdown and documentation files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extractor import extract_claims
from settings import MAX_CLAIMS_PER_AUDIT
from verifier import compute_risk_weighted_overall_score, verify_claim


def find_line_number(doc_text: str, target: str) -> int:
    """
    Locate the 1-based line number in `doc_text` where `target` appears.
    Falls back to matching the first 4-5 words if exact match fails.
    """
    if not target or not str(target).strip():
        return 1

    clean_target = str(target).strip()
    lines = doc_text.splitlines()

    # 1. Exact substring in line
    for idx, line in enumerate(lines, start=1):
        if clean_target in line:
            return idx

    # 2. Case-insensitive substring match
    lower_target = clean_target.lower()
    for idx, line in enumerate(lines, start=1):
        if lower_target in line.lower():
            return idx

    # 3. First 4 words prefix match
    words = clean_target.split()
    if len(words) >= 3:
        prefix = " ".join(words[:4]).lower()
        for idx, line in enumerate(lines, start=1):
            if prefix in line.lower():
                return idx

    return 1


def audit_markdown_document(
    doc_text: str,
    max_claims: int = MAX_CLAIMS_PER_AUDIT,
    filename: str = "document.md",
    use_cache: bool = True,
    tavily_client: Any | None = None,
) -> dict[str, Any]:
    """
    Extract factual claims from markdown/text, map to source line numbers,
    and verify freshness against live web intelligence.
    """
    clean_text = str(doc_text or "").strip()
    if not clean_text:
        return {
            "filename": filename,
            "overall_score": 100,
            "claims": [],
            "fresh_count": 0,
            "outdated_count": 0,
            "unverifiable_count": 0,
            "error": "Document is empty.",
        }

    claims, model_used = extract_claims(clean_text)
    if max_claims and len(claims) > max_claims:
        claims = claims[:max_claims]

    verified_claims: list[dict[str, Any]] = []
    for claim_item in claims:
        # Determine 1-based line number in original file
        search_target = claim_item.get("span") or claim_item.get("claim") or ""
        line_num = find_line_number(doc_text, search_target)

        # Run verification
        try:
            ver_res = verify_claim(
                claim_item,
                tavily_client=tavily_client,
                use_cache=use_cache,
            )
        except Exception as exc:
            ver_res = {
                "claim": claim_item.get("claim", ""),
                "status": "unverifiable",
                "score": 0,
                "reason": f"Verification error: {exc}",
                "error": str(exc),
            }

        ver_res["line_number"] = line_num
        ver_res["category"] = claim_item.get("category", "other")
        ver_res["span"] = claim_item.get("span", "")
        verified_claims.append(ver_res)

    overall_score = compute_risk_weighted_overall_score(verified_claims) if verified_claims else 100
    if overall_score is None:
        overall_score = 100
    fresh_cnt = sum(1 for c in verified_claims if c.get("status") == "fresh")
    outdated_cnt = sum(1 for c in verified_claims if c.get("status") in {"outdated", "incorrect"})
    unverifiable_cnt = sum(1 for c in verified_claims if c.get("status") in {"unverifiable", "uncertain", "partially_correct"})

    return {
        "filename": filename,
        "overall_score": overall_score,
        "claims": verified_claims,
        "fresh_count": fresh_cnt,
        "outdated_count": outdated_cnt,
        "unverifiable_count": unverifiable_cnt,
        "total_claims": len(verified_claims),
        "extract_model": model_used,
    }


def format_markdown_report(result: dict[str, Any]) -> str:
    """Format audit findings into a human-readable Markdown report."""
    fname = result.get("filename", "document.md")
    score = result.get("overall_score", 100)
    fresh = result.get("fresh_count", 0)
    outdated = result.get("outdated_count", 0)
    unverifiable = result.get("unverifiable_count", 0)

    lines = [
        f"# Stale-Answer Detector: Document Freshness Audit",
        f"**File:** `{fname}` | **Overall Freshness:** `{score}/100` | **Fresh:** `{fresh}` | **Outdated:** `{outdated}` | **Unverifiable:** `{unverifiable}`",
        "",
        "| Line | Status | Category | Claim | Current Reality / Verified Finding | Source |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    claims = result.get("claims") or []
    if not claims:
        lines.append("| - | - | - | *No factual claims detected in document* | - | - |")
    else:
        for c in claims:
            ln = f"L{c.get('line_number', 1)}"
            st = str(c.get("status", "unknown")).upper()
            status_icon = "🟢" if st == "FRESH" else ("🔴" if st in {"OUTDATED", "INCORRECT"} else "⚪")
            status_cell = f"{status_icon} {st}"
            cat = str(c.get("category", "other")).title()
            clm = str(c.get("claim", "")).replace("|", "\\|")
            current = str(c.get("current_info") or c.get("reason") or "Verified against web evidence").replace("|", "\\|")
            
            src_str = "None"
            sources = c.get("sources") or []
            if sources:
                top_src = sources[0]
                url = top_src.get("url", "#")
                title = top_src.get("title") or top_src.get("domain") or "Link"
                src_str = f"[{str(title)[:20]}]({url})"

            lines.append(f"| {ln} | {status_cell} | {cat} | {clm} | {current} | {src_str} |")

    lines.append("")
    lines.append(f"*Report generated by Stale-Answer Detector CLI (stale_check)*")
    return "\n".join(lines)


def format_json_report(result: dict[str, Any]) -> str:
    """Format audit findings into formatted JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False)
