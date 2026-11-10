"""
verifier.py - Searches live web sources via Tavily and judges claim freshness.

For each extracted claim, this module:
1. Fetches 5 live web results using the Tavily Search API.
2. Passes those results and today's date to the LLM to judge freshness.
3. Attaches the top 3 source URLs and handles Tavily/JSON errors gracefully.
4. Also provides a helper to rewrite the original answer with verified info.
"""

from datetime import date
import os
from typing import Callable

from dotenv import load_dotenv
from tavily import TavilyClient

from llm import call_llm, parse_json_response

# Load environment variables from .env if present.
load_dotenv()

VALID_STATUSES = ("fresh", "outdated", "uncertain")

VERIFICATION_SYSTEM_PROMPT = """You are an objective fact-checking judge.
Given today's date, a specific factual claim from an older LLM answer, and live
web search results, determine whether the claim is still accurate today.

Return ONLY a valid JSON object (no markdown fences or extra commentary) with
these exact keys:
- "status": Must be one of "fresh", "outdated", or "uncertain".
  * "fresh": The web results clearly confirm the claim is still accurate today.
  * "outdated": The web results show the claim is no longer true (e.g., new price,
    newer software version, leadership change, new law, updated statistic).
  * "uncertain": The web results are conflicting, incomplete, or inconclusive.
- "score": An integer from 0 to 100 representing freshness:
  * 80 to 100 for "fresh"
  * 40 to 79 for "uncertain" or partially outdated
  * 0 to 39 for "outdated"
- "current_info": A concise statement of the current, up-to-date facts found in
  the search results (include specific numbers, dates, names, or versions).
- "reason": A brief 1-2 sentence explanation comparing the claim to the live sources.
"""

REWRITE_SYSTEM_PROMPT = """You are a helpful editor updating an outdated AI answer.
Rewrite the original answer so that all outdated facts are replaced with the
verified current information provided below.

Guidelines:
- Preserve the helpful tone, clarity, and structure of the original answer.
- Keep claims marked "fresh" intact.
- Replace claims marked "outdated" using the verified "current_info".
- For claims marked "uncertain", qualify them carefully (or note that details
  could not be fully confirmed) without inventing unverified facts.
- Return ONLY the rewritten answer text (no preamble like "Here is the updated answer:").
"""


def get_tavily_client() -> TavilyClient:
    """
    Create a TavilyClient using TAVILY_API_KEY from .env.

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    load_dotenv(override=False)
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Please add your Tavily API key to the .env file."
        )
    return TavilyClient(api_key=api_key)


def search_tavily(
    query: str,
    max_results: int = 5,
    client: TavilyClient | None = None,
) -> list[dict]:
    """
    Query the Tavily Search API and return a list of result dicts.

    Args:
        query: Web search query string.
        max_results: Number of search results to request (default: 5).
        client: Optional pre-initialized TavilyClient.

    Returns:
        List of search result dictionaries from Tavily.

    Raises:
        RuntimeError: If the Tavily API call fails.
    """
    tavily_client = client or get_tavily_client()
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
        results = response.get("results", [])
        return results if isinstance(results, list) else []
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed for query {query!r}: {exc}") from exc


def _extract_top_sources(search_results: list[dict], limit: int = 3) -> list[dict[str, str]]:
    """
    Extract up to `limit` unique source links (title + url) from Tavily results.
    """
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for result in search_results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = str(result.get("title", "")).strip() or url
        sources.append({"title": title, "url": url})
        if len(sources) >= limit:
            break

    return sources


def _format_search_results(search_results: list[dict]) -> str:
    """
    Format Tavily search results into readable text for the LLM prompt.
    """
    if not search_results:
        return "No web search results were returned."

    formatted_blocks: list[str] = []
    for idx, result in enumerate(search_results, start=1):
        title = str(result.get("title", "Untitled")).strip()
        url = str(result.get("url", "")).strip()
        published_date = str(result.get("published_date", "") or "").strip()
        content = str(result.get("content", "")).strip()

        # Keep each snippet concise so prompts remain fast and readable.
        if len(content) > 800:
            content = content[:800] + "..."

        date_line = f"Published: {published_date}\n" if published_date else ""
        formatted_blocks.append(
            f"[{idx}] {title}\nURL: {url}\n{date_line}Snippet: {content}"
        )

    return "\n\n".join(formatted_blocks)


def _normalize_judgement(parsed: object) -> dict:
    """
    Validate and normalize the LLM's freshness judgement JSON.
    """
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object for freshness judgement, got {type(parsed).__name__}."
        )

    raw_status = str(parsed.get("status", "uncertain")).strip().lower()
    status = raw_status if raw_status in VALID_STATUSES else "uncertain"

    # Safely parse and clamp the score into 0..100.
    default_scores = {"fresh": 90, "uncertain": 50, "outdated": 15}
    raw_score = parsed.get("score", default_scores[status])
    try:
        score = int(round(float(raw_score)))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = default_scores[status]

    current_info = str(parsed.get("current_info", "")).strip()
    if not current_info:
        current_info = "No current details could be extracted from the sources."

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No detailed reason was provided."

    return {
        "status": status,
        "score": score,
        "current_info": current_info,
        "reason": reason,
    }


def verify_claim(
    claim_item: dict[str, str],
    tavily_client: TavilyClient | None = None,
) -> dict:
    """
    Verify a single extracted claim against live Tavily search results.

    Errors during Tavily search or LLM JSON parsing are caught and returned as
    an 'uncertain' result with an 'error' field so the rest of the check continues.

    Args:
        claim_item: Dict containing 'claim', 'category', and 'search_query'.
        tavily_client: Optional shared TavilyClient instance.

    Returns:
        Dict with claim, category, search_query, status, score, current_info,
        reason, sources (top 3), and optional error message.
    """
    claim_text = claim_item.get("claim", "").strip()
    category = claim_item.get("category", "other").strip()
    search_query = claim_item.get("search_query", "").strip() or claim_text

    # Step 1: Search live web sources via Tavily (5 results).
    try:
        search_results = search_tavily(
            query=search_query,
            max_results=5,
            client=tavily_client,
        )
    except Exception as exc:
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": 50,
            "current_info": "Unable to retrieve live web sources.",
            "reason": f"Tavily search failed: {exc}",
            "sources": [],
            "error": str(exc),
        }

    top_sources = _extract_top_sources(search_results, limit=3)

    if not search_results:
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": 50,
            "current_info": "No relevant live search results were found.",
            "reason": "Tavily returned 0 results for this search query.",
            "sources": [],
        }

    # Step 2: Ask the LLM to judge freshness using today's date + search snippets.
    today_str = date.today().isoformat()
    formatted_sources = _format_search_results(search_results)

    user_prompt = (
        f"Today's date: {today_str}\n"
        f"Claim category: {category}\n"
        f"Claim to verify: {claim_text}\n\n"
        f"Live web search results:\n{formatted_sources}"
    )

    model_used: str | None = None
    try:
        raw_response, model_used = call_llm(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        parsed = parse_json_response(raw_response)
        judgement = _normalize_judgement(parsed)
    except Exception as exc:
        # Even if the LLM step fails, still attach the top 3 Tavily URLs.
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": 50,
            "current_info": "Could not judge freshness automatically due to an LLM error.",
            "reason": f"Verification error: {exc}",
            "sources": top_sources,
            "model": model_used,
            "error": str(exc),
        }

    return {
        "claim": claim_text,
        "category": category,
        "search_query": search_query,
        "status": judgement["status"],
        "score": judgement["score"],
        "current_info": judgement["current_info"],
        "reason": judgement["reason"],
        "sources": top_sources,
        "model": model_used,
    }


def verify_claims(
    claims: list[dict[str, str]],
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    """
    Verify a list of claims against live web sources.

    Args:
        claims: List of claim dicts from extractor.extract_claims().
        progress_callback: Optional callback invoked as callback(index, total, result).

    Returns:
        List of verification result dicts in the same order.

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    if not claims:
        return []

    # Validate Tavily configuration upfront before running individual searches.
    tavily_client = get_tavily_client()

    verified_results: list[dict] = []
    total = len(claims)

    for idx, claim_item in enumerate(claims, start=1):
        result = verify_claim(claim_item, tavily_client=tavily_client)
        verified_results.append(result)
        if progress_callback is not None:
            progress_callback(idx, total, result)

    return verified_results


def generate_corrected_answer(
    original_answer: str,
    verification_results: list[dict],
) -> tuple[str, str]:
    """
    Rewrite the original LLM answer using the verified current information.

    Args:
        original_answer: The text originally pasted by the user.
        verification_results: The list of dicts returned by verify_claims().

    Returns:
        A tuple of (updated_answer_text, model_used).
    """
    today_str = date.today().isoformat()

    findings_lines: list[str] = []
    for idx, item in enumerate(verification_results, start=1):
        findings_lines.append(
            f"{idx}. Claim: {item.get('claim', '')}\n"
            f"   Status: {item.get('status', 'uncertain')} (Score: {item.get('score', 50)}/100)\n"
            f"   Current Info: {item.get('current_info', '')}\n"
            f"   Reason: {item.get('reason', '')}"
        )

    findings_text = "\n\n".join(findings_lines)

    user_prompt = (
        f"Today's date: {today_str}\n\n"
        f"Original answer:\n{original_answer.strip()}\n\n"
        f"Verified claim findings:\n{findings_text}"
    )

    return call_llm(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )
