"""
extractor.py - Extracts potentially outdated factual claims from an LLM answer.

Uses the configured LLM to scan a pasted text and return a structured list of
claims, each tagged with a category and a web search query for live verification.
"""

import threading
import time

from llm import call_llm, parse_json_response, pop_last_gateway_metrics

_extractor_state = threading.local()


def get_last_gateway_metrics() -> dict | None:
    """Return telemetry recorded during the most recent `extract_claims` call."""
    return getattr(_extractor_state, "last_metrics", None)

# Allowed categories for extracted claims.
VALID_CATEGORIES = (
    "price",
    "version",
    "regulation",
    "company",
    "statistic",
    "other",
)

EXTRACTION_SYSTEM_PROMPT = """You are a careful fact-checking assistant.
Your job is to inspect an LLM-generated answer and extract specific factual claims
that could become outdated over time (for example: prices, API/software versions,
laws or regulations, company info like CEO/funding/acquisitions, and statistics).

Return ONLY a valid JSON array (no markdown explanation, no extra text).
Each element in the JSON array must be an object with exactly these keys:
- "claim": A clear, self-contained statement of the factual claim (replace pronouns
  like "it" or "their" with the actual entity name so the claim makes sense on its own).
- "category": Must be one of: "price", "version", "regulation", "company", "statistic", "other".
- "search_query": A concise, targeted web search query to check the current truth of this claim.

If the text contains no time-sensitive or verifiable factual claims, return [].
"""


def _normalize_claims(parsed_data: object) -> list[dict[str, str]]:
    """
    Validate and normalize the JSON returned by the LLM into a clean list of claim dicts.
    """
    # Some models wrap a JSON list inside an object like {"claims": [...]}.
    if isinstance(parsed_data, dict):
        for value in parsed_data.values():
            if isinstance(value, list):
                parsed_data = value
                break

    if not isinstance(parsed_data, list):
        raise ValueError(
            "Expected the LLM to return a JSON list of claims, "
            f"but got {type(parsed_data).__name__}."
        )

    cleaned_claims: list[dict[str, str]] = []
    for item in parsed_data:
        if not isinstance(item, dict):
            continue

        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue

        raw_category = str(item.get("category", "other")).strip().lower()
        category = raw_category if raw_category in VALID_CATEGORIES else "other"

        search_query = str(item.get("search_query", "")).strip() or claim_text

        cleaned_claims.append(
            {
                "claim": claim_text,
                "category": category,
                "search_query": search_query,
            }
        )

    return cleaned_claims


def extract_claims(answer_text: str) -> tuple[list[dict[str, str]], str]:
    """
    Extract potentially stale claims from a pasted LLM answer.

    Args:
        answer_text: The full LLM-generated text pasted by the user.

    Returns:
        A tuple of (claims_list, model_used), where claims_list contains dicts with:
        - claim (str)
        - category (str): one of price, version, regulation, company, statistic, other
        - search_query (str)

    Raises:
        ValueError: If the LLM returns invalid/malformed JSON.
        RuntimeError: If the LLM API call fails even after retries/fallbacks.
    """
    if not answer_text or not answer_text.strip():
        _extractor_state.last_metrics = None
        return [], ""

    user_prompt = (
        "Extract all potentially outdated or time-sensitive factual claims from "
        "the following LLM answer:\n\n"
        f"{answer_text.strip()}"
    )

    pop_last_gateway_metrics()
    t0 = time.perf_counter()
    raw_response, model_used = call_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    metrics = pop_last_gateway_metrics()
    if metrics is None:
        metrics = {
            "primary_model": model_used,
            "requested_model": model_used,
            "model_used": model_used,
            "provider_used": model_used.split("/", 1)[0] if "/" in str(model_used) else None,
            "fallback_models": [],
            "fallback_used": False,
            "latency_ms": elapsed_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "simulated_outage": False,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    metrics["operation"] = "Extract claims"
    _extractor_state.last_metrics = metrics

    # Parse and validate the JSON list returned by the LLM.
    parsed = parse_json_response(raw_response)
    return _normalize_claims(parsed), model_used
