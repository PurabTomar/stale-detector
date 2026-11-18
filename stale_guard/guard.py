"""stale_guard/guard.py - Real-data Freshness Guard for RAG and chatbot replies.

Verifies an LLM answer against live Tavily web sources, computes an honest
verdict and risk-weighted score, returns cited corrections when stale or
incorrect claims are found, and reports only measured latency and returned
token/credit usage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any
from urllib.parse import urlparse

from extractor import extract_claims, get_last_gateway_metrics as get_extract_metrics
from llm import is_use_bifrost
from reporting import build_draft_corrected_answer
from settings import MAX_CLAIMS_PER_AUDIT, MAX_TEXT_CHARS
from verifier import (
    build_citation_map,
    compute_risk_weighted_overall_score,
    generate_corrected_answer,
    get_last_gateway_metrics as get_rewrite_metrics,
    verify_claims,
)

DEFAULT_MAX_CLAIMS: int = 6
MAX_GUARD_QUESTION_CHARS: int = 1000
MAX_GUARD_ANSWER_CHARS: int = MAX_TEXT_CHARS
MAX_GUARD_CONTEXT_CHARS: int = 8000


def compute_guard_verdict(
    claims: list[dict[str, Any]],
    overall_score: int | None,
) -> str:
    """Map verified claim statuses to a single deterministic guard verdict.

    Returns one of:
    - "no_claims": no verifiable time-sensitive claims detected
    - "verification_failed": all claims failed verification (LLM/Tavily error)
    - "incorrect": one or more claims were contradicted by authoritative sources
    - "stale": one or more claims were outdated (or overall score < 40)
    - "needs_review": partial verification failure, uncertain, unverifiable, or partially_correct
    - "fresh": all checked claims verified current
    """
    if not claims:
        return "no_claims"

    statuses = [str(c.get("status") or "").strip().lower() for c in claims if isinstance(c, dict)]
    if not statuses:
        return "no_claims"

    failed = sum(1 for s in statuses if s == "verification_failed")
    if failed == len(statuses):
        return "verification_failed"
    if any(s == "incorrect" for s in statuses):
        return "incorrect"
    if any(s == "outdated" for s in statuses) or (overall_score is not None and overall_score < 40):
        return "stale"
    if failed > 0 or any(s in {"partially_correct", "uncertain", "unverifiable"} for s in statuses):
        return "needs_review"
    return "fresh"


def _collect_guard_sources(
    claims: list[dict[str, Any]],
    citation_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect deduplicated http(s) sources, placing cited primary sources first with stable 1-based indices."""
    ordered: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for cit in citation_map.get("citations") or []:
        if not isinstance(cit, dict):
            continue
        url = str(cit.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        ordered.append(
            {
                "index": int(cit.get("number") or (len(ordered) + 1)),
                "title": str(cit.get("title") or cit.get("domain") or url),
                "url": url,
                "domain": str(cit.get("domain") or urlparse(url).netloc),
                "source_type": str(cit.get("source_type") or "Primary"),
                "published_date": cit.get("published_date"),
                "cited": True,
            }
        )

    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for src in claim.get("sources") or []:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url") or "").strip()
            if not url or not url.lower().startswith(("http://", "https://")) or url in seen_urls:
                continue
            seen_urls.add(url)
            dom = str(src.get("domain") or "").strip() or urlparse(url).netloc
            if dom.startswith("www."):
                dom = dom[4:]
            ordered.append(
                {
                    "index": len(ordered) + 1,
                    "title": str(src.get("title") or dom or url),
                    "url": url,
                    "domain": dom,
                    "source_type": str(src.get("source_type") or "Source"),
                    "published_date": src.get("published_date"),
                    "cited": False,
                }
            )
    return ordered


def _collect_real_usage(
    claims: list[dict[str, Any]],
    extract_meta: dict[str, Any] | None,
    rewrite_meta: dict[str, Any] | None,
    elapsed_ms: float,
) -> dict[str, Any]:
    """Aggregate ONLY real returned usage and measured latency (zero invented numbers)."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    llm_calls = 0
    models_used: list[str] = []

    def _add_call(meta: dict[str, Any] | None) -> None:
        nonlocal prompt_tokens, completion_tokens, total_tokens, llm_calls
        if not isinstance(meta, dict):
            return
        if meta.get("error"):
            return
        p_tok = int(meta.get("prompt_tokens") or 0)
        c_tok = int(meta.get("completion_tokens") or 0)
        t_tok = int(meta.get("total_tokens") or (p_tok + c_tok))
        if meta.get("usage_returned") is False and t_tok == 0:
            return
        if t_tok == 0 and p_tok == 0 and c_tok == 0 and meta.get("usage_returned") is not True:
            return
        llm_calls += 1
        prompt_tokens += p_tok
        completion_tokens += c_tok
        total_tokens += t_tok
        m = str(meta.get("model_used") or meta.get("model") or "").strip()
        if m and m not in models_used:
            models_used.append(m)

    _add_call(extract_meta)
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        jt = claim.get("judge_telemetry")
        if isinstance(jt, list) and jt:
            for item_meta in jt:
                _add_call(item_meta)
        else:
            _add_call(claim.get("gateway_metrics"))
    _add_call(rewrite_meta)

    tavily_credits = round(
        sum(float(c.get("tavily_credits_used") or 0.0) for c in claims if isinstance(c, dict)),
        2,
    )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_calls": llm_calls,
        "tavily_credits_used": tavily_credits,
        "latency_ms": round(float(elapsed_ms), 1),
        "routing_mode": "bifrost" if is_use_bifrost() else "direct_gemini",
        "models_used": models_used,
    }


def guard(
    question: str,
    answer: str,
    context: str | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    *,
    use_cache: bool = True,
    dual_judge: bool = False,
    auto_correct: bool = True,
) -> dict[str, Any]:
    """Verify an AI answer for freshness and optionally return a cited correction.

    Args:
        question: The user's original prompt or question.
        answer: The LLM or RAG answer to audit.
        context: Optional retrieved RAG context (validated for size, never executed as instructions).
        max_claims: Maximum number of extracted claims to verify (clamped to 1..MAX_CLAIMS_PER_AUDIT).
        use_cache: Reuse the 6-hour SQLite Tavily cache when True.
        dual_judge: Run two judge models per claim when True.
        auto_correct: Generate a cited corrected answer when outdated/incorrect/partially_correct claims exist.

    Returns:
        Dict with keys: verdict, overall_score, claims, corrected_answer, sources, usage,
        question, answer, context_provided, generated_at.
    """
    q_clean = str(question or "").strip()
    a_clean = str(answer or "").strip()
    ctx_clean = str(context).strip() if context is not None else None

    if not q_clean:
        raise ValueError("The 'question' argument must be a non-empty string.")
    if len(q_clean) > MAX_GUARD_QUESTION_CHARS:
        raise ValueError(f"The 'question' argument exceeds the {MAX_GUARD_QUESTION_CHARS}-character limit.")
    if not a_clean:
        raise ValueError("The 'answer' argument must be a non-empty string.")
    if len(a_clean) > MAX_GUARD_ANSWER_CHARS:
        raise ValueError(f"The 'answer' argument exceeds the {MAX_GUARD_ANSWER_CHARS}-character limit.")
    if ctx_clean and len(ctx_clean) > MAX_GUARD_CONTEXT_CHARS:
        raise ValueError(f"The 'context' argument exceeds the {MAX_GUARD_CONTEXT_CHARS}-character limit.")

    try:
        clamped_max = int(max_claims)
    except (TypeError, ValueError) as exc:
        raise ValueError("The 'max_claims' argument must be an integer.") from exc
    if clamped_max < 1 or clamped_max > MAX_CLAIMS_PER_AUDIT:
        raise ValueError(f"The 'max_claims' argument must be between 1 and {MAX_CLAIMS_PER_AUDIT}.")

    t0 = time.perf_counter()

    # Step 1: Extract factual claims from the answer.
    extracted, _extract_model = extract_claims(a_clean)
    extract_meta = get_extract_metrics()
    claims_to_verify = list(extracted[:clamped_max])

    # Step 2: Verify claims against live web sources.
    if not claims_to_verify:
        verified_claims: list[dict[str, Any]] = []
    elif dual_judge:
        verified_claims = verify_claims(claims_to_verify, use_cache=use_cache, dual_judge=True)
    else:
        verified_claims = verify_claims(claims_to_verify, use_cache=use_cache)

    overall_score = compute_risk_weighted_overall_score(verified_claims)
    verdict = compute_guard_verdict(verified_claims, overall_score)
    citation_map = build_citation_map(verified_claims)

    # Step 3: Build cited corrected answer when replaceable claims exist.
    needs_correction = any(
        isinstance(c, dict) and str(c.get("status") or "").lower() in {"outdated", "incorrect", "partially_correct"}
        for c in verified_claims
    )
    has_unverified = bool(citation_map.get("unverified_claims"))
    rewrite_meta: dict[str, Any] | None = None

    if auto_correct and needs_correction:
        try:
            corrected_answer, _rw_model = generate_corrected_answer(a_clean, verified_claims)
            rewrite_meta = get_rewrite_metrics()
        except Exception:
            corrected_answer = build_draft_corrected_answer(a_clean, verified_claims)
    elif auto_correct and has_unverified:
        corrected_answer = build_draft_corrected_answer(a_clean, verified_claims)
    else:
        corrected_answer = a_clean

    sources = _collect_guard_sources(verified_claims, citation_map)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    usage = _collect_real_usage(verified_claims, extract_meta, rewrite_meta, elapsed_ms)

    return {
        "verdict": verdict,
        "overall_score": overall_score,
        "question": q_clean,
        "answer": a_clean,
        "context_provided": bool(ctx_clean),
        "claims": verified_claims,
        "corrected_answer": corrected_answer,
        "sources": sources,
        "usage": usage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
