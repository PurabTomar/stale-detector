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
    including supporting sources from verified claims (outdated first, then uncertain).
    """
    changes: list[dict] = []
    priority_order = {"outdated": 0, "uncertain": 1}

    for idx, item in enumerate(verification_results or [], start=1):
        status = str(item.get("status", "uncertain")).strip().lower()
        if status not in priority_order:
            continue

        action = "Replaced" if status == "outdated" else "Qualified"
        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for src in item.get("sources") or []:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(src.get("title", "")).strip() or extract_source_domain(url)
            sources.append({"title": title, "url": url})

        changes.append(
            {
                "claim_index": idx,
                "status": status,
                "action": action,
                "category": str(item.get("category", "other")).lower(),
                "risk_level": str(item.get("risk_level", "moderate")).lower(),
                "score": max(0, min(100, int(item.get("score", 50)))),
                "original_claim": str(item.get("claim", "")).strip(),
                "updated_to": str(item.get("current_info", "")).strip(),
                "why": str(item.get("reason", "")).strip(),
                "sources": sources,
            }
        )

    changes.sort(key=lambda c: (priority_order.get(c["status"], 9), c["claim_index"]))
    return changes


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

    return {
        "models_used": models_used,
        "fallback_routes": fallback_routes,
        "tokens_used": {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "requests_count": len(requests),
            "total_latency_ms": total_latency_ms,
        },
        "request_timestamps": request_timestamps,
        "gateway_requests": requests,
    }


def build_report_payload(
    mode: str,
    original_answer: str,
    verification_results: list[dict] | None,
    question: str | None = None,
    corrected_answer: str | None = None,
    ask_meta: dict | None = None,
    extract_model: str | None = None,
    extract_meta: dict | None = None,
    corrected_model: str | None = None,
    corrected_meta: dict | None = None,
    tavily_credits_used: float | None = None,
    generated_at: datetime | None = None,
) -> dict:
    """
    Build a complete JSON-serializable audit report dictionary containing claims,
    scores, sources, timestamps, models used, tokens used, and answer diff data.
    """
    results = list(verification_results or [])
    ts_dt = generated_at or datetime.now().astimezone()
    ts_iso = ts_dt.isoformat(timespec="seconds")

    if results:
        overall_score = round(sum(int(r.get("score", 50)) for r in results) / len(results))
        overall_tier = _score_tier_label(overall_score)
    else:
        overall_score = None
        overall_tier = "No claims checked"

    fresh_count = sum(1 for r in results if r.get("status") == "fresh")
    uncertain_count = sum(1 for r in results if r.get("status") == "uncertain")
    outdated_count = sum(1 for r in results if r.get("status") == "outdated")

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
    )

    claims_payload: list[dict] = []
    for idx, item in enumerate(results, start=1):
        gw = item.get("gateway_metrics") if isinstance(item.get("gateway_metrics"), dict) else {}
        p_tok = int(gw.get("prompt_tokens") or 0)
        c_tok = int(gw.get("completion_tokens") or 0)
        t_tok = int(gw.get("total_tokens") or (p_tok + c_tok))
        claims_payload.append(
            {
                "index": idx,
                "claim": str(item.get("claim", "")),
                "category": str(item.get("category", "other")),
                "search_query": str(item.get("search_query", "")),
                "status": str(item.get("status", "uncertain")),
                "score": max(0, min(100, int(item.get("score", 50)))),
                "confidence": str(item.get("confidence", "medium")),
                "risk_level": str(item.get("risk_level", "moderate")),
                "current_info": str(item.get("current_info", "")),
                "reason": str(item.get("reason", "")),
                "why_this_score": str(item.get("why_this_score", "")),
                "score_breakdown": item.get("score_breakdown"),
                "sources": [
                    {
                        "title": str(s.get("title", "")),
                        "url": str(s.get("url", "")),
                    }
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
                "latency_ms": round(float(gw.get("latency_ms") or 0.0), 1),
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
    score_display = (
        f"{overall_score}/100 ({summary.get('overall_tier', '')})"
        if overall_score is not None
        else "No claims checked"
    )

    lines: list[str] = [
        "# Stale-Answer Detector Report",
        "",
        f"- **Generated at:** {timestamps.get('generated_at') or payload.get('generated_at', '')}",
        f"- **Mode:** {payload.get('mode', '')}",
        f"- **Overall freshness score:** {score_display}",
        f"- **Claims summary:** {summary.get('claims_checked', 0)} checked "
        f"({summary.get('fresh', 0)} fresh, {summary.get('uncertain', 0)} uncertain, {summary.get('outdated', 0)} outdated)",
        f"- **Models used:** {', '.join(models) if models else 'None'}",
        f"- **Fallback routes:** {', '.join(fallbacks) if fallbacks else 'None (primary only)'}",
        f"- **Tokens used:** {int(tokens.get('total_tokens', 0)):,} total "
        f"({int(tokens.get('prompt_tokens', 0)):,} prompt / {int(tokens.get('completion_tokens', 0)):,} completion) "
        f"across {int(tokens.get('requests_count', 0))} request(s)",
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
            lines.append(
                f"### Claim #{ch['claim_index']} — {ch['action']} "
                f"({ch['category']}, score {ch['score']}/100)"
            )
            lines.append(f"- **Original claim:** {ch['original_claim']}")
            lines.append(f"- **Updated info:** {ch['updated_to']}")
            lines.append(f"- **Why:** {ch['why']}")
            srcs = ch.get("sources") or []
            if srcs:
                src_md = ", ".join(f"[{s['title']}]({s['url']})" for s in srcs)
                lines.append(f"- **Sources:** {src_md}")
            lines.append("")
    else:
        lines.extend(["No outdated or uncertain claims needed modification.", ""])

    claims = payload.get("claims") or []
    lines.extend(["## Verified Claims", ""])
    if claims:
        for c in claims:
            c_tok = c.get("tokens") or {}
            lines.append(
                f"### {c['index']}. {c['claim']} "
                f"[{str(c['status']).upper()} — {c['score']}/100]"
            )
            lines.append(
                f"- **Category / Risk / Confidence:** {c['category']} / {c['risk_level']} / {c['confidence']}"
            )
            lines.append(f"- **Current info:** {c['current_info']}")
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
                lines.append(
                    "- **Sources:** " + ", ".join(f"[{s['title']}]({s['url']})" for s in srcs)
                )
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
