"""
reporting.py - Word-level diff rendering, change-log extraction, and report exports.

Provides beginner-friendly, deterministic helpers for:
1. Building side-by-side HTML word-level diffs between the original answer
   (removed text struck through in red) and the corrected answer (new text
   highlighted in green).
2. Extracting a structured "What changed & why" list with source links from
   verified claim results.
3. Packaging complete audit reports into both Markdown and JSON formats
   containing claims, scores, sources, timestamps, models used, and tokens used.
"""

from datetime import datetime
import difflib
import html
import json
import re
from urllib.parse import urlparse

from verifier import (
    classify_source_type,
    compute_risk_weighted_overall_score,
    truncate_snippet_words,
)


def _normalize_source_entry(
    src: dict,
    entity_domains: list[str] | None = None,
) -> dict:
    """Normalize a verified source dict, keeping source_type, published_date, and <=25w snippet."""
    url = str(src.get("url", "")).strip()
    title = str(src.get("title", "")).strip() or extract_source_domain(url)
    raw_type = str(src.get("source_type") or "").strip()
    if raw_type in {"Primary", "News", "Blog", "Forum"}:
        src_type = raw_type
    else:
        src_type = classify_source_type(url, entity_domains=entity_domains)
    pub_date = src.get("published_date")
    pub_date_str = str(pub_date).strip() if pub_date else None
    snippet = truncate_snippet_words(src.get("snippet") or "")
    return {
        "title": title,
        "url": url,
        "source_type": src_type,
        "published_date": pub_date_str,
        "snippet": snippet,
    }


# Tokenize text into whitespace, atomic numbers/currencies/versions, words, and punctuation.
_DIFF_TOKEN_RE = re.compile(
    r"\s+|[$€£₹]?\d[\d,]*(?:\.\d+)*%?|[\w'-]+|[^\s\w]"
)


def escape_html_text(text: str) -> str:
    """Escape HTML characters and '$' so Streamlit renders literal currency symbols safely."""
    return html.escape(str(text or ""), quote=True).replace("$", "&#36;")


def extract_source_domain(url: str) -> str:
    """Return a clean domain name (without 'www.') from a URL."""
    parsed = urlparse((url or "").strip())
    domain = parsed.netloc or (url or "")
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def tokenize_diff_words(text: str) -> list[str]:
    """
    Split text into word/number/whitespace/punctuation tokens so word-level diffs
    highlight changed numbers (like '$1,199' or '3.14.7') cleanly without tagging
    surrounding punctuation unless it also changed.
    """
    if not text:
        return []
    return _DIFF_TOKEN_RE.findall(text)


def _format_plain_html(text: str) -> str:
    """Escape plain text for HTML card rendering while preserving newlines."""
    return escape_html_text(text).replace("\n", "<br>")


def _wrap_diff_segment(segment: str, tag_name: str, css_class: str) -> str:
    """
    Wrap changed non-whitespace text inside `<del>` or `<ins>` while keeping
    leading/trailing whitespace outside the colored pill and splitting across lines.
    """
    if not segment:
        return ""
    match = re.match(r"^(\s*)([\s\S]*?)(\s*)$", segment)
    if not match:
        return _format_plain_html(segment)

    leading_ws, core, trailing_ws = match.groups()
    if not core:
        return _format_plain_html(segment)

    line_parts: list[str] = []
    for line in core.split("\n"):
        if line:
            line_parts.append(f'<{tag_name} class="{css_class}">{escape_html_text(line)}</{tag_name}>')
        else:
            line_parts.append("")

    return (
        _format_plain_html(leading_ws)
        + "<br>".join(line_parts)
        + _format_plain_html(trailing_ws)
    )


def _count_word_tokens(tokens: list[str]) -> int:
    """Count word or numeric tokens (ignoring pure whitespace and punctuation)."""
    return sum(1 for tok in tokens if any(ch.isalnum() for ch in tok))


def build_side_by_side_word_diff(original_text: str, corrected_text: str) -> dict:
    """
    Compute a word-level diff formatted for side-by-side display:
    - `original_html`: original answer with removed/replaced words struck through in red.
    - `corrected_html`: corrected answer with inserted/replacement words highlighted in green.
    - `stats`: word counts (`removed_words`, `added_words`, `unchanged_words`, `has_changes`).
    """
    orig_tokens = tokenize_diff_words(original_text or "")
    corr_tokens = tokenize_diff_words(corrected_text or "")

    matcher = difflib.SequenceMatcher(None, orig_tokens, corr_tokens, autojunk=False)
    orig_parts: list[str] = []
    corr_parts: list[str] = []

    removed_words = 0
    added_words = 0
    unchanged_words = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_slice = orig_tokens[i1:i2]
        corr_slice = corr_tokens[j1:j2]

        if tag == "equal":
            shared_html = _format_plain_html("".join(orig_slice))
            orig_parts.append(shared_html)
            corr_parts.append(shared_html)
            unchanged_words += _count_word_tokens(orig_slice)
        elif tag == "delete":
            orig_parts.append(
                _wrap_diff_segment("".join(orig_slice), "del", "sad-diff-del")
            )
            removed_words += _count_word_tokens(orig_slice)
        elif tag == "insert":
            corr_parts.append(
                _wrap_diff_segment("".join(corr_slice), "ins", "sad-diff-ins")
            )
            added_words += _count_word_tokens(corr_slice)
        elif tag == "replace":
            orig_parts.append(
                _wrap_diff_segment("".join(orig_slice), "del", "sad-diff-del")
            )
            corr_parts.append(
                _wrap_diff_segment("".join(corr_slice), "ins", "sad-diff-ins")
            )
            removed_words += _count_word_tokens(orig_slice)
            added_words += _count_word_tokens(corr_slice)

    return {
        "original_html": "".join(orig_parts),
        "corrected_html": "".join(corr_parts),
        "stats": {
            "removed_words": removed_words,
            "added_words": added_words,
            "unchanged_words": unchanged_words,
            "has_changes": bool(removed_words or added_words or (original_text or "").strip() != (corrected_text or "").strip()),
        },
    }


def build_change_log(verification_results: list[dict] | None) -> list[dict]:
    """
    Build a structured list of what changed during answer correction and why,
    including supporting sources from verified claims (outdated, uncertain, unverifiable).
    """
    changes: list[dict] = []
    priority_order = {"outdated": 0, "uncertain": 1, "unverifiable": 2}
    action_labels = {
        "outdated": "Replaced",
        "uncertain": "Qualified",
        "unverifiable": "Flagged (unverifiable)",
    }

    for idx, item in enumerate(verification_results or [], start=1):
        status = str(item.get("status", "uncertain")).strip().lower()
        if status not in priority_order:
            continue

        raw_score = item.get("score")
        score_val = None if (raw_score is None or status == "unverifiable") else max(0, min(100, int(raw_score)))

        sources: list[dict] = []
        seen_urls: set[str] = set()
        entity_doms = item.get("entity_domains") if isinstance(item.get("entity_domains"), list) else None
        for src in item.get("sources") or []:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append(_normalize_source_entry(src, entity_domains=entity_doms))

        changes.append(
            {
                "claim_index": idx,
                "status": status,
                "action": action_labels[status],
                "category": str(item.get("category", "other")).lower(),
                "risk_level": str(item.get("risk_level", "moderate")).lower(),
                "score": score_val,
                "original_claim": str(item.get("claim", "")).strip(),
                "updated_to": str(item.get("current_info", "")).strip(),
                "current_value": item.get("current_value"),
                "effective_date": item.get("effective_date"),
                "change_summary": item.get("change_summary"),
                "stale_since": item.get("stale_since"),
                "stale_for_months": item.get("stale_for_months"),
                "stale_duration_label": item.get("stale_duration_label"),
                "agreement_summary": item.get("agreement_summary"),
                "conflict_detected": bool(item.get("conflict_detected", False)),
                "conflict_note": item.get("conflict_note"),
                "why": str(item.get("reason", "")).strip(),
                "sources": sources,
            }
        )

    changes.sort(key=lambda c: (priority_order.get(c["status"], 9), c["claim_index"]))
    return changes


def build_draft_corrected_answer(
    original_answer: str,
    verification_results: list[dict] | None,
    matched_spans: list[dict] | None = None,
) -> str:
    """
    Build a deterministic draft preview of the corrected answer without making an
    extra LLM call, so a side-by-side comparison and word diff can be shown
    immediately after any audit finishes.
    """
    text = str(original_answer or "")
    results = list(verification_results or [])
    if not text.strip() or not results:
        return text

    outdated_entries = [
        (idx, item)
        for idx, item in enumerate(results)
        if isinstance(item, dict) and str(item.get("status", "")).lower() == "outdated"
    ]
    if not outdated_entries:
        return text

    def _ensure_sentence(val: str) -> str:
        clean = str(val or "").strip()
        if not clean:
            return ""
        if clean[-1] not in ".!?":
            clean += "."
        return clean

    # Group outdated updates by matched character span when span positions are available.
    span_updates: dict[tuple[int, int], list[str]] = {}
    unmatched_updates: list[str] = []

    if matched_spans:
        span_by_claim_idx = {
            int(sp["claim_index"]): (int(sp["start"]), int(sp["end"]))
            for sp in matched_spans
            if isinstance(sp, dict) and "claim_index" in sp and "start" in sp and "end" in sp
        }
    else:
        span_by_claim_idx = {}

    sentence_ranges = [
        (m.start(), m.end())
        for m in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", text)
        if m.group(0).strip()
    ]
    stop = {"the", "and", "for", "with", "that", "this", "from", "are", "was", "is", "in", "on", "of", "to"}

    def _fallback_sentence_span(claim_text: str) -> tuple[int, int] | None:
        c_tokens = {
            tok for tok in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", claim_text.lower())
            if (len(tok) >= 3 or any(ch.isdigit() for ch in tok)) and tok not in stop
        }
        if not c_tokens or not sentence_ranges:
            return None
        best_span = None
        best_ratio = 0.0
        for s_start, s_end in sentence_ranges:
            s_tokens = set(re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text[s_start:s_end].lower()))
            overlap = len(c_tokens & s_tokens)
            ratio = overlap / len(c_tokens)
            if overlap >= 2 and ratio > best_ratio:
                best_ratio = ratio
                best_span = (s_start, s_end)
        return best_span if best_ratio >= 0.35 else None

    for claim_idx, item in outdated_entries:
        replacement_text = _ensure_sentence(
            item.get("current_info") or item.get("current_value") or ""
        )
        if not replacement_text:
            continue

        span = span_by_claim_idx.get(claim_idx) or _fallback_sentence_span(str(item.get("claim", "")))
        if span and 0 <= span[0] < span[1] <= len(text):
            span_updates.setdefault(span, []).append(replacement_text)
        else:
            unmatched_updates.append(replacement_text)

    if not span_updates and not unmatched_updates:
        return text

    # Apply span replacements from left to right (skipping any overlapping spans).
    parts: list[str] = []
    cursor = 0
    for (start, end), updates in sorted(span_updates.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if start < cursor:
            unmatched_updates.extend(u for u in updates if u not in unmatched_updates)
            continue
        parts.append(text[cursor:start])
        # Preserve unique update sentences for that span.
        dedup_updates = list(dict.fromkeys(updates))
        parts.append(" ".join(dedup_updates))
        cursor = end
    parts.append(text[cursor:])

    updated_text = "".join(parts).strip()
    if unmatched_updates:
        extra = " ".join(dict.fromkeys(unmatched_updates))
        updated_text = f"{updated_text}\n\n{extra}".strip() if updated_text else extra
    return updated_text


def _score_tier_label(score: int) -> str:
    """Return tier label matching UI thresholds (Fresh 80-100, Mixed 40-79, Stale 0-39)."""
    if score >= 80:
        return "Fresh"
    if score >= 40:
        return "Mixed"
    return "Stale"


def _normalize_gateway_record(
    operation: str,
    raw_metrics: dict | None,
    fallback_model: str | None = None,
    fallback_used: bool = False,
    primary_model: str | None = None,
) -> dict | None:
    """Normalize a single Bifrost/LLM telemetry dict into a consistent report record."""
    if isinstance(raw_metrics, dict):
        p_tok = int(raw_metrics.get("prompt_tokens") or 0)
        c_tok = int(raw_metrics.get("completion_tokens") or 0)
        t_tok = int(raw_metrics.get("total_tokens") or (p_tok + c_tok))
        m_used = str(raw_metrics.get("model_used") or fallback_model or "Unknown")
        return {
            "operation": str(raw_metrics.get("operation") or operation),
            "task": str(raw_metrics.get("task") or "default"),
            "timestamp": str(raw_metrics.get("timestamp") or ""),
            "primary_model": str(raw_metrics.get("primary_model") or primary_model or m_used),
            "model_used": m_used,
            "fallback_used": bool(raw_metrics.get("fallback_used") or fallback_used),
            "latency_ms": round(float(raw_metrics.get("latency_ms") or 0.0), 1),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
        }

    if not fallback_model:
        return None

    m_used = str(fallback_model)
    return {
        "operation": operation,
        "task": "default",
        "timestamp": "",
        "primary_model": str(primary_model or m_used),
        "model_used": m_used,
        "fallback_used": bool(fallback_used),
        "latency_ms": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def collect_report_telemetry(
    results: list[dict] | None,
    ask_meta: dict | None = None,
    extract_model: str | None = None,
    extract_meta: dict | None = None,
    corrected_model: str | None = None,
    corrected_meta: dict | None = None,
    audit_duration_ms: float | None = None,
) -> dict:
    """
    Aggregate models used, fallback routes, timestamps, latency, and token usage
    across every LLM call in the audit (Q&A, extraction, verification, and rewrite).
    """
    requests: list[dict] = []

    ask_rec = _normalize_gateway_record("Ask & Audit Q&A", ask_meta)
    if ask_rec:
        requests.append(ask_rec)

    extract_rec = _normalize_gateway_record(
        "Extract claims",
        extract_meta,
        fallback_model=extract_model,
    )
    if extract_rec:
        requests.append(extract_rec)

    for idx, item in enumerate(results or [], start=1):
        judge_tel = [
            entry
            for entry in (item.get("judge_telemetry") or [])
            if isinstance(entry, dict)
        ]
        if judge_tel:
            for j_idx, j_meta in enumerate(judge_tel, start=1):
                default_op = (
                    f"Verify claim #{idx}"
                    if len(judge_tel) == 1
                    else f"Verify claim #{idx} (Judge {j_idx})"
                )
                j_rec = _normalize_gateway_record(
                    default_op,
                    j_meta,
                    fallback_model=j_meta.get("model_used") or item.get("model"),
                    fallback_used=bool(j_meta.get("fallback_used", False)),
                    primary_model=j_meta.get("primary_model") or item.get("primary_model"),
                )
                if j_rec:
                    requests.append(j_rec)
        else:
            gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else None
            claim_rec = _normalize_gateway_record(
                f"Verify claim #{idx}",
                gw,
                fallback_model=item.get("model"),
                fallback_used=bool(item.get("fallback_used", False)),
                primary_model=item.get("primary_model"),
            )
            if claim_rec:
                requests.append(claim_rec)

    rewrite_rec = _normalize_gateway_record(
        "Rewrite answer",
        corrected_meta,
        fallback_model=corrected_model,
    )
    if rewrite_rec:
        requests.append(rewrite_rec)

    models_used: list[str] = []
    fallback_routes: list[str] = []
    request_timestamps: list[dict[str, str]] = []

    for rec in requests:
        m = rec["model_used"]
        if m and m not in models_used:
            models_used.append(m)
        if rec["fallback_used"]:
            route = f"{rec['primary_model']} -> {m}"
            if route not in fallback_routes:
                fallback_routes.append(route)
        if rec["timestamp"]:
            request_timestamps.append(
                {
                    "operation": rec["operation"],
                    "timestamp": rec["timestamp"],
                }
            )

    total_prompt = sum(r["prompt_tokens"] for r in requests)
    total_completion = sum(r["completion_tokens"] for r in requests)
    total_tokens = sum(r["total_tokens"] for r in requests)
    total_latency_ms = round(sum(r["latency_ms"] for r in requests), 1)
    avg_latency_ms = round(total_latency_ms / len(requests), 1) if requests else 0.0
    fallback_count = sum(1 for r in requests if r["fallback_used"])
    resolved_audit_ms = (
        round(float(audit_duration_ms), 1)
        if audit_duration_ms is not None
        else total_latency_ms
    )

    return {
        "models_used": models_used,
        "fallback_routes": fallback_routes,
        "tokens_used": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "requests_count": len(requests),
            "fallback_count": fallback_count,
            "total_latency_ms": total_latency_ms,
            "avg_latency_ms": avg_latency_ms,
            "audit_duration_ms": resolved_audit_ms,
        },
        "request_timestamps": request_timestamps,
        "gateway_requests": requests,
    }


def build_report_payload(
    mode: str,
    original_answer: str,
    verification_results: list[dict] | None,
    question: str | None = None,
    source_url: str | None = None,
    corrected_answer: str | None = None,
    ask_meta: dict | None = None,
    extract_model: str | None = None,
    extract_meta: dict | None = None,
    corrected_model: str | None = None,
    corrected_meta: dict | None = None,
    tavily_credits_used: float | None = None,
    generated_at: datetime | None = None,
    audit_duration_ms: float | None = None,
) -> dict:
    """
    Build a complete JSON-serializable audit report dictionary containing claims,
    scores, sources, timestamps, models used, tokens used, and answer diff data.
    """
    results = list(verification_results or [])
    ts_dt = generated_at or datetime.now().astimezone()
    ts_iso = ts_dt.isoformat(timespec="seconds")

    if results:
        overall_score = compute_risk_weighted_overall_score(results)
        overall_tier = (
            _score_tier_label(overall_score)
            if overall_score is not None
            else "Unverifiable"
        )
    else:
        overall_score = None
        overall_tier = "No claims checked"

    fresh_count = sum(1 for r in results if r.get("status") == "fresh")
    uncertain_count = sum(1 for r in results if r.get("status") == "uncertain")
    outdated_count = sum(1 for r in results if r.get("status") == "outdated")
    unverifiable_count = sum(1 for r in results if r.get("status") == "unverifiable")

    if tavily_credits_used is None:
        credits_total = round(
            sum(float(r.get("tavily_credits_used") or 0.0) for r in results),
            2,
        )
    else:
        credits_total = round(float(tavily_credits_used), 2)

    telemetry = collect_report_telemetry(
        results=results,
        ask_meta=ask_meta,
        extract_model=extract_model,
        extract_meta=extract_meta,
        corrected_model=corrected_model,
        corrected_meta=corrected_meta,
        audit_duration_ms=audit_duration_ms,
    )

    claims_payload: list[dict] = []
    for idx, item in enumerate(results, start=1):
        gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else {}
        judge_tel = [
            entry
            for entry in (item.get("judge_telemetry") or [])
            if isinstance(entry, dict)
        ]
        if judge_tel:
            p_tok = sum(int(j.get("prompt_tokens") or 0) for j in judge_tel)
            c_tok = sum(int(j.get("completion_tokens") or 0) for j in judge_tel)
            t_tok = sum(
                int(j.get("total_tokens") or (int(j.get("prompt_tokens") or 0) + int(j.get("completion_tokens") or 0)))
                for j in judge_tel
            )
            claim_latency_ms = round(sum(float(j.get("latency_ms") or 0.0) for j in judge_tel), 1)
        else:
            p_tok = int(gw.get("prompt_tokens") or 0)
            c_tok = int(gw.get("completion_tokens") or 0)
            t_tok = int(gw.get("total_tokens") or (p_tok + c_tok))
            claim_latency_ms = round(float(gw.get("latency_ms") or 0.0), 1)

        status_str = str(item.get("status", "uncertain")).strip().lower()
        raw_score = item.get("score")
        score_val = (
            None
            if (raw_score is None or status_str == "unverifiable")
            else max(0, min(100, int(raw_score)))
        )
        entity_doms = item.get("entity_domains") if isinstance(item.get("entity_domains"), list) else None
        judge_verdicts = [
            {
                "model": str(v.get("model") or ""),
                "status": str(v.get("status") or ""),
                "score": v.get("score"),
                "current_info": str(v.get("current_info") or ""),
                "reason": str(v.get("reason") or ""),
            }
            for v in (item.get("judge_verdicts") or [])
            if isinstance(v, dict)
        ]
        claims_payload.append(
            {
                "index": idx,
                "claim": str(item.get("claim", "")),
                "span": item.get("span"),
                "category": str(item.get("category", "other")),
                "search_query": str(item.get("search_query", "")),
                "status": status_str,
                "score": score_val,
                "confidence": str(item.get("confidence", "low")),
                "risk_level": str(item.get("risk_level", "moderate")),
                "current_info": str(item.get("current_info", "")),
                "current_value": item.get("current_value"),
                "effective_date": item.get("effective_date"),
                "change_summary": item.get("change_summary"),
                "stale_since": item.get("stale_since"),
                "stale_for_months": item.get("stale_for_months"),
                "stale_duration_label": item.get("stale_duration_label"),
                "entity_domains": list(entity_doms or []),
                "sources_agreeing": item.get("sources_agreeing"),
                "sources_total": item.get("sources_total"),
                "agreement_summary": item.get("agreement_summary"),
                "conflict_detected": bool(item.get("conflict_detected", False)),
                "conflict_note": item.get("conflict_note"),
                "dual_judge_used": bool(item.get("dual_judge_used", False)),
                "judge_models": [str(m) for m in (item.get("judge_models") or []) if m],
                "judge_verdicts": judge_verdicts,
                "model_disagreement": bool(item.get("model_disagreement", False)),
                "model_agreement_summary": item.get("model_agreement_summary"),
                "reason": str(item.get("reason", "")),
                "why_this_score": str(item.get("why_this_score", "")),
                "score_breakdown": item.get("score_breakdown"),
                "sources": [
                    _normalize_source_entry(s, entity_domains=entity_doms)
                    for s in (item.get("sources") or [])
                    if isinstance(s, dict) and s.get("url")
                ],
                "extracted_url": item.get("extracted_url"),
                "search_strategy": item.get("search_strategy"),
                "cache_hit": bool(item.get("cache_hit", False)),
                "tavily_credits_used": round(float(item.get("tavily_credits_used") or 0.0), 2),
                "model": item.get("model") or gw.get("model_used"),
                "primary_model": item.get("primary_model") or gw.get("primary_model"),
                "fallback_used": bool(item.get("fallback_used") or gw.get("fallback_used")),
                "tokens": {
                    "prompt_tokens": p_tok,
                    "completion_tokens": c_tok,
                    "total_tokens": t_tok,
                },
                "latency_ms": claim_latency_ms,
                "timestamp": gw.get("timestamp") or None,
                "error": item.get("error"),
            }
        )

    diff_stats = None
    if corrected_answer is not None:
        diff_stats = build_side_by_side_word_diff(original_answer or "", corrected_answer)["stats"]

    changes_made = build_change_log(results)

    return {
        "report_version": "1.0",
        "app": "Stale-Answer Detector",
        "mode": mode,
        "generated_at": ts_iso,
        "timestamps": {
            "generated_at": ts_iso,
            "date": ts_dt.date().isoformat(),
            "request_timestamps": telemetry["request_timestamps"],
        },
        "question": (question or "").strip() or None,
        "source_url": (source_url or "").strip() or None,
        "original_answer": original_answer or "",
        "corrected_answer": corrected_answer,
        "diff_summary": diff_stats,
        "summary": {
            "overall_score": overall_score,
            "overall_tier": overall_tier,
            "claims_checked": len(results),
            "fresh": fresh_count,
            "uncertain": uncertain_count,
            "outdated": outdated_count,
            "unverifiable": unverifiable_count,
            "tavily_credits_used": credits_total,
        },
        "models_used": telemetry["models_used"],
        "fallback_routes": telemetry["fallback_routes"],
        "tokens_used": telemetry["tokens_used"],
        "gateway_requests": telemetry["gateway_requests"],
        "changes_made": changes_made,
        "claims": claims_payload,
    }


def format_report_json(payload: dict) -> str:
    """Serialize an audit report payload dictionary into formatted JSON."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_report_markdown(payload: dict) -> str:
    """Render an audit report payload dictionary as a clean Markdown document."""
    summary = payload.get("summary") or {}
    tokens = payload.get("tokens_used") or {}
    models = payload.get("models_used") or []
    fallbacks = payload.get("fallback_routes") or []
    timestamps = payload.get("timestamps") or {}

    overall_score = summary.get("overall_score")
    if overall_score is not None:
        score_display = f"{overall_score}/100 ({summary.get('overall_tier', '')}) — risk-weighted"
    else:
        score_display = summary.get("overall_tier") or "No claims checked"

    lines: list[str] = [
        "# Stale-Answer Detector Report",
        "",
        f"- **Generated at:** {timestamps.get('generated_at') or payload.get('generated_at', '')}",
        f"- **Mode:** {payload.get('mode', '')}",
        f"- **Overall freshness score:** {score_display}",
        f"- **Claims summary:** {summary.get('claims_checked', 0)} checked "
        f"({summary.get('fresh', 0)} fresh, {summary.get('uncertain', 0)} uncertain, "
        f"{summary.get('outdated', 0)} outdated, {summary.get('unverifiable', 0)} unverifiable)",
        f"- **Models used:** {', '.join(models) if models else 'None'}",
        f"- **Fallback routes:** {', '.join(fallbacks) if fallbacks else 'None (primary only)'}",
        f"- **Tokens used:** {int(tokens.get('total_tokens', 0)):,} total "
        f"({int(tokens.get('prompt_tokens', 0)):,} prompt / {int(tokens.get('completion_tokens', 0)):,} completion) "
        f"across {int(tokens.get('requests_count', 0))} request(s) "
        f"(avg latency {float(tokens.get('avg_latency_ms', 0.0)):.0f} ms, "
        f"audit time {float(tokens.get('audit_duration_ms', 0.0)):.0f} ms, "
        f"{int(tokens.get('fallback_count', 0))} fallback(s))",
        f"- **Tavily credits used:** {summary.get('tavily_credits_used', 0)}",
        "",
    ]

    if payload.get("question"):
        lines.extend(
            [
                "## Question",
                "",
                payload["question"],
                "",
            ]
        )

    if payload.get("source_url"):
        lines.extend(
            [
                "## Source URL",
                "",
                payload["source_url"],
                "",
            ]
        )

    lines.extend(
        [
            "## Original Answer",
            "",
            payload.get("original_answer") or "_Empty_",
            "",
        ]
    )

    if payload.get("corrected_answer"):
        diff_stats = payload.get("diff_summary") or {}
        lines.extend(
            [
                "## Corrected Answer",
                "",
                payload["corrected_answer"],
                "",
                f"_Word diff: {int(diff_stats.get('removed_words', 0))} removed, "
                f"{int(diff_stats.get('added_words', 0))} added, "
                f"{int(diff_stats.get('unchanged_words', 0))} unchanged._",
                "",
            ]
        )

    changes = payload.get("changes_made") or []
    lines.extend(["## What Changed & Why", ""])
    if changes:
        for ch in changes:
            ch_score_str = "score N/A" if ch.get("score") is None else f"score {ch['score']}/100"
            lines.append(
                f"### Claim #{ch['claim_index']} — {ch['action']} "
                f"({ch['category']}, {ch_score_str})"
            )
            lines.append(f"- **Original claim:** {ch['original_claim']}")
            lines.append(f"- **Updated info:** {ch['updated_to']}")
            if ch.get("current_value"):
                lines.append(f"- **Current value:** {ch['current_value']}")
            if ch.get("change_summary"):
                lines.append(f"- **Change summary:** {ch['change_summary']}")
            if ch.get("effective_date"):
                lines.append(f"- **Effective date:** {ch['effective_date']}")
            if ch.get("stale_since") or ch.get("stale_duration_label"):
                stale_raw = str(ch.get("stale_since") or "").strip()
                dur_raw = str(ch.get("stale_duration_label") or "").strip()
                if stale_raw and dur_raw and dur_raw.lower() not in stale_raw.lower():
                    stale_display = f"{stale_raw} ({dur_raw})"
                else:
                    stale_display = stale_raw or dur_raw
                lines.append(f"- **Stale since:** {stale_display}")
            if ch.get("agreement_summary"):
                conflict_txt = f" — {ch['conflict_note']}" if ch.get("conflict_note") else ""
                lines.append(f"- **Source agreement:** {ch['agreement_summary']}{conflict_txt}")
            lines.append(f"- **Why:** {ch['why']}")
            srcs = ch.get("sources") or []
            if srcs:
                src_parts = []
                for s in srcs:
                    tag = f"[{s.get('source_type')}] " if s.get("source_type") else ""
                    date_str = f" ({s['published_date']})" if s.get("published_date") else ""
                    src_parts.append(f"{tag}[{s['title']}]({s['url']}){date_str}")
                lines.append(f"- **Sources:** {', '.join(src_parts)}")
            lines.append("")
    else:
        lines.extend(["No outdated, uncertain, or unverifiable claims needed modification.", ""])

    claims = payload.get("claims") or []
    lines.extend(["## Verified Claims", ""])
    if claims:
        for c in claims:
            c_tok = c.get("tokens") or {}
            c_score_str = "N/A" if c.get("score") is None else f"{c['score']}/100"
            lines.append(
                f"### {c['index']}. {c['claim']} "
                f"[{str(c['status']).upper()} — {c_score_str}]"
            )
            lines.append(
                f"- **Category / Risk / Confidence:** {c['category']} / {c['risk_level']} / {c['confidence']}"
            )
            if c.get("agreement_summary"):
                conflict_txt = f" — {c['conflict_note']}" if c.get("conflict_note") else ""
                lines.append(f"- **Source agreement:** {c['agreement_summary']}{conflict_txt}")
            if c.get("model_agreement_summary"):
                verdict_bits = [
                    f"`{v['model']}` → {str(v['status']).upper()}"
                    for v in (c.get("judge_verdicts") or [])
                    if isinstance(v, dict) and v.get("model")
                ]
                verdict_detail = f" ({'; '.join(verdict_bits)})" if verdict_bits else ""
                lines.append(f"- **Dual-judge consensus:** {c['model_agreement_summary']}{verdict_detail}")
            lines.append(f"- **Current info:** {c['current_info']}")
            if c.get("current_value"):
                lines.append(f"- **Current value:** {c['current_value']}")
            if c.get("change_summary"):
                lines.append(f"- **Change summary:** {c['change_summary']}")
            if c.get("effective_date"):
                lines.append(f"- **Effective date:** {c['effective_date']}")
            if c.get("stale_since") or c.get("stale_duration_label"):
                stale_raw = str(c.get("stale_since") or "").strip()
                dur_raw = str(c.get("stale_duration_label") or "").strip()
                if stale_raw and dur_raw and dur_raw.lower() not in stale_raw.lower():
                    stale_display = f"{stale_raw} ({dur_raw})"
                else:
                    stale_display = stale_raw or dur_raw
                lines.append(f"- **Stale since:** {stale_display}")
            lines.append(f"- **Reason:** {c['reason']}")
            if c.get("why_this_score"):
                lines.append(f"- **Why this score:** {c['why_this_score']}")
            if c.get("model"):
                fb_note = f" (fallback from `{c['primary_model']}`)" if c.get("fallback_used") and c.get("primary_model") else ""
                lines.append(
                    f"- **Model & tokens:** `{c['model']}`{fb_note} — "
                    f"{int(c_tok.get('total_tokens', 0)):,} tokens "
                    f"({int(c_tok.get('prompt_tokens', 0)):,} in / {int(c_tok.get('completion_tokens', 0)):,} out)"
                )
            if c.get("timestamp"):
                lines.append(f"- **Timestamp:** {c['timestamp']}")
            srcs = c.get("sources") or []
            if srcs:
                src_parts = []
                for s in srcs:
                    tag = f"[{s.get('source_type')}] " if s.get("source_type") else ""
                    date_str = f" ({s['published_date']})" if s.get("published_date") else ""
                    src_parts.append(f"{tag}[{s['title']}]({s['url']}){date_str}")
                lines.append("- **Sources:** " + ", ".join(src_parts))
                for s in srcs:
                    if s.get("snippet"):
                        lines.append(f"  - _{s.get('source_type') or 'Source'}_ ({extract_source_domain(s['url'])}): \"{s['snippet']}\"")
            if c.get("error"):
                lines.append(f"- **Error:** {c['error']}")
            lines.append("")
    else:
        lines.extend(["No time-sensitive claims were extracted.", ""])

    gw_reqs = payload.get("gateway_requests") or []
    lines.extend(["## Gateway Requests, Models & Token Usage", ""])
    if gw_reqs:
        lines.extend(
            [
                "| # | Operation | Timestamp | Model Used | Fallback | Latency (ms) | Prompt Tokens | Completion Tokens | Total Tokens |",
                "|---|-----------|-----------|------------|----------|--------------|---------------|-------------------|--------------|",
            ]
        )
        for req_idx, req in enumerate(gw_reqs, start=1):
            fb_text = f"Yes ({req['primary_model']} -> {req['model_used']})" if req.get("fallback_used") else "No"
            ts_text = req.get("timestamp") or "-"
            lines.append(
                f"| {req_idx} | {req['operation']} | {ts_text} | `{req['model_used']}` | "
                f"{fb_text} | {req['latency_ms']} | {req['prompt_tokens']} | "
                f"{req['completion_tokens']} | {req['total_tokens']} |"
            )
        lines.append("")
    else:
        lines.extend(["No LLM telemetry records captured.", ""])

    return "\n".join(lines).rstrip() + "\n"
