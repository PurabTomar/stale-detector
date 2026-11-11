"""
verifier.py - Searches live web sources via Tavily and computes explainable freshness scores.

For each extracted claim, this module:
1. Chooses a Tavily search strategy by claim category and risk level:
   - price / version / company: topic="news" (recent time_range) + normal search
   - regulation: prefers official and government domains via include_domains
   - high-risk claims (regulation, price): search_depth="advanced"; others use "basic"
2. Uses Tavily Extract on the single best source URL to confirm exact figures.
3. Caches Tavily search & extract responses in an on-disk SQLite cache (6h TTL).
4. Passes live sources + extracted content + today's date to the LLM for a verdict.
5. Computes an explainable 0-100 freshness score from 4 weighted signals:
   - LLM verdict (50%)
   - Source recency from Tavily published_date (20%)
   - Source agreement across top search results (20%)
   - Source authority by domain type (10%)
6. Returns score_breakdown, why_this_score, confidence, risk_level, and Tavily credits used.
7. Provides a helper to rewrite the original answer with verified current info.
"""

from contextlib import contextmanager
from datetime import date, datetime
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Callable
from urllib.parse import urlparse

from dotenv import load_dotenv
import requests
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    TimeoutError as TavilyTimeoutError,
    UsageLimitExceededError,
)

from llm import _get_config_value, call_llm, format_llm_error, parse_json_response, pop_last_gateway_metrics

_verifier_state = threading.local()


def get_last_gateway_metrics() -> dict | None:
    """Return telemetry recorded during the most recent `generate_corrected_answer` call."""
    return getattr(_verifier_state, "last_metrics", None)


def _capture_call_metrics(operation: str, model_used: str | None, elapsed_ms: float) -> dict | None:
    """Capture gateway telemetry recorded by `call_llm` (or build a minimal fallback)."""
    metrics = pop_last_gateway_metrics()
    if metrics is None:
        if not model_used:
            return None
        metrics = {
            "primary_model": model_used,
            "requested_model": model_used,
            "model_used": model_used,
            "provider_used": model_used.split("/", 1)[0] if "/" in str(model_used) else None,
            "fallback_models": [],
            "fallback_used": False,
            "latency_ms": round(elapsed_ms, 1),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "simulated_outage": False,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    metrics["operation"] = operation
    return metrics

# Load environment variables from .env if present.
load_dotenv()

VALID_STATUSES = ("fresh", "outdated", "uncertain")

# Weights for the final 0-100 freshness score (sum = 1.0).
SCORE_WEIGHTS = {
    "llm_verdict": 0.50,
    "source_recency": 0.20,
    "source_agreement": 0.20,
    "source_authority": 0.10,
}

# Category-based risk levels (wrong legal/pricing info has higher real-world impact).
HIGH_RISK_CATEGORIES = {"regulation", "price"}
MODERATE_RISK_CATEGORIES = {"version", "company", "statistic"}

# Tavily category strategy and on-disk cache configuration (TTL = 6 hours).
DEFAULT_CACHE_PATH = ".tavily_cache.db"
CACHE_TTL_SECONDS = 6 * 60 * 60
NEWS_DUAL_SEARCH_CATEGORIES = {"price", "version", "company"}
RECENT_NEWS_TIME_RANGE = "month"
TAVILY_RETRY_DELAY_SECONDS = 1.0
MAX_EXTRACTED_CHARS = 2500

# Official and government domain patterns preferred when verifying regulation claims.
REGULATION_INCLUDE_DOMAINS = [
    "*.gov",
    "*.gov.in",
    "*.gov.uk",
    "*.gov.au",
    "*.gc.ca",
    "*.edu",
    "*.int",
    "europa.eu",
    "eur-lex.europa.eu",
    "who.int",
    "gst.gov.in",
    "cbic.gov.in",
    "incometax.gov.in",
    "rbi.org.in",
    "sebi.gov.in",
    "sec.gov",
    "irs.gov",
    "ftc.gov",
    "legislation.gov.uk",
    "canada.ca",
]

# Domain patterns used by the simple source authority heuristic.
OFFICIAL_AUTHORITY_DOMAINS = {
    "python.org",
    "pypi.org",
    "w3.org",
    "ietf.org",
    "who.int",
    "europa.eu",
    "sec.gov",
    "irs.gov",
    "apple.com",
    "openai.com",
    "microsoft.com",
    "google.com",
    "github.com",
    "anthropic.com",
    "wikipedia.org",
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
}

FORUM_OR_SOCIAL_DOMAINS = {
    "reddit.com",
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "linkedin.com",
    "youtube.com",
    "tiktok.com",
    "news.ycombinator.com",
}

BLOG_DOMAINS = {
    "medium.com",
    "substack.com",
    "dev.to",
    "hashnode.dev",
    "wordpress.com",
    "blogspot.com",
    "tumblr.com",
}

# Common English stopwords ignored when checking keyword overlap across snippets.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "have", "has", "had", "not", "but", "into", "about", "current", "currently",
    "today", "latest", "now", "more", "than", "per", "its", "their", "been",
}

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
- "score": An integer from 0 to 100 representing your verdict on freshness:
  * 80 to 100 for "fresh"
  * 40 to 79 for "uncertain" or partially outdated
  * 0 to 39 for "outdated"
- "agreeing_sources": An integer count (0 to N) of how many of the provided
  search results support the same current fact stated in "current_info".
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
    Create a TavilyClient using TAVILY_API_KEY from .env or Streamlit Secrets.

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    load_dotenv(override=False)
    api_key = _get_config_value("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is not set. Add your Tavily API key to .env or Streamlit Secrets."
        )
    return TavilyClient(api_key=api_key)


def get_cache_path(cache_path: str | None = None) -> str:
    """Return the path to the on-disk SQLite cache file for Tavily responses."""
    if cache_path:
        return cache_path
    return os.getenv("TAVILY_CACHE_PATH", "").strip() or DEFAULT_CACHE_PATH


@contextmanager
def _open_cache_db(path: str):
    """Open (and initialize if needed) the SQLite cache table and always close it."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5.0)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tavily_cache (
                cache_key TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _build_cache_key(endpoint: str, params: dict) -> str:
    """Build a deterministic SHA-256 key for a Tavily request payload."""
    serialized = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_get(
    cache_key: str,
    cache_path: str | None = None,
    ttl_seconds: int = CACHE_TTL_SECONDS,
    now: float | None = None,
) -> dict | None:
    """Fetch a fresh cached Tavily JSON response if present and within TTL."""
    path = get_cache_path(cache_path)
    current_time = time.time() if now is None else now
    try:
        with _open_cache_db(path) as conn:
            row = conn.execute(
                "SELECT response_json, created_at FROM tavily_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None

            raw_json, created_at = row
            if (current_time - float(created_at)) >= ttl_seconds:
                # Expired entry: remove it so the next API call refreshes cleanly.
                conn.execute("DELETE FROM tavily_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                return None

            parsed = json.loads(raw_json)
            return parsed if isinstance(parsed, dict) else None
    except (sqlite3.Error, ValueError, TypeError):
        return None


def _cache_set(
    cache_key: str,
    endpoint: str,
    response: dict,
    cache_path: str | None = None,
    now: float | None = None,
) -> None:
    """Persist a valid Tavily response dictionary into the on-disk SQLite cache."""
    if not isinstance(response, dict):
        return
    path = get_cache_path(cache_path)
    current_time = time.time() if now is None else now
    try:
        payload_json = json.dumps(response)
        with _open_cache_db(path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tavily_cache (cache_key, endpoint, response_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, endpoint, payload_json, current_time),
            )
            conn.commit()
    except (sqlite3.Error, TypeError, ValueError):
        # Cache write failure should never interrupt live verification.
        pass


def clear_tavily_cache(cache_path: str | None = None) -> int:
    """Delete all cached Tavily entries and return the number of rows removed."""
    path = get_cache_path(cache_path)
    if not os.path.exists(path):
        return 0
    try:
        with _open_cache_db(path) as conn:
            cur = conn.execute("DELETE FROM tavily_cache")
            conn.commit()
            return cur.rowcount or 0
    except sqlite3.Error:
        return 0


def _is_retryable_tavily_error(exc: Exception) -> bool:
    """Return True if a Tavily exception is transient (rate limit, timeout, or 5xx)."""
    if isinstance(
        exc,
        (
            UsageLimitExceededError,
            TavilyTimeoutError,
            TimeoutError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True

    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return exc.response.status_code in {429, 500, 502, 503, 504}

    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "timed out" in msg or "timeout" in msg


def _sanitize_single_line(text: str, max_len: int = 180) -> str:
    """Extract a clean single-line message and strip any accidental stack trace lines."""
    lines = [
        ln.strip()
        for ln in str(text or "").splitlines()
        if ln.strip()
        and not ln.strip().startswith("Traceback ")
        and not ln.strip().startswith('File "')
    ]
    summary = lines[0] if lines else "Unexpected error"
    if len(summary) > max_len:
        summary = summary[: max_len - 3] + "..."
    return summary


def _format_tavily_error(exc: Exception) -> str:
    """Format Tavily SDK and HTTP exceptions into concise, helpful error text."""
    if isinstance(exc, InvalidAPIKeyError):
        return "Invalid Tavily API key (401). Please check TAVILY_API_KEY in .env or Streamlit Secrets."
    if isinstance(exc, UsageLimitExceededError):
        detail = _sanitize_single_line(str(exc), max_len=120)
        return (
            f"Tavily rate or credit limit reached (429): {detail}"
            if detail and detail != "Unexpected error"
            else "Tavily rate or credit limit reached (429). Wait a moment or check your Tavily plan usage."
        )
    if isinstance(exc, ForbiddenError):
        lower = str(exc).lower()
        if any(tok in lower for tok in ("limit", "quota", "credit", "plan", "432", "433")):
            return "Tavily credit or plan limit reached (432/433). Check your Tavily quota at app.tavily.com."
        return "Invalid or unauthorized Tavily API key (403). Please verify TAVILY_API_KEY."
    if isinstance(exc, BadRequestError):
        return f"Invalid Tavily search request (400): {_sanitize_single_line(str(exc), max_len=120)}"
    if isinstance(exc, (TavilyTimeoutError, requests.exceptions.Timeout, TimeoutError)):
        return "Tavily web search timed out. Please try again in a moment."
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        status_code = exc.response.status_code
        if status_code == 401:
            return "Invalid Tavily API key (401). Please check TAVILY_API_KEY in .env or Streamlit Secrets."
        if status_code == 429:
            return "Tavily rate limit reached (429). Please wait a moment and try again."
        if status_code in {432, 433, 434}:
            return f"Tavily credit or plan limit exceeded ({status_code}). Check your Tavily account quota."
        return f"Tavily HTTP error ({status_code}). Please try again."
    raw = str(exc)
    lower = raw.lower()
    if raw.startswith("Invalid Tavily API key") or raw.startswith("Tavily rate or credit limit") or raw.startswith("Tavily credit or plan limit"):
        return _sanitize_single_line(raw)
    if "401" in lower or "invalid api key" in lower or "unauthorized" in lower:
        return "Invalid Tavily API key (401). Please check TAVILY_API_KEY in .env or Streamlit Secrets."
    if any(tok in lower for tok in ("429", "432", "433", "434", "usage limit", "rate limit", "credit limit")):
        return "Tavily rate or credit limit reached (429/quota). Check your Tavily quota at app.tavily.com."
    return _sanitize_single_line(raw)


def _extract_usage_credits(response: dict, default_credits: float) -> float:
    """
    Read credit cost from Tavily's `usage.credits` object when present, or fall
    back to Tavily's documented per-call estimate (1 basic / 2 advanced).
    """
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict) and usage.get("credits") is not None:
            try:
                return max(0.0, float(usage["credits"]))
            except (TypeError, ValueError):
                pass
    return max(0.0, float(default_credits))


def _call_tavily_search_with_retry(
    tavily_client: TavilyClient,
    kwargs: dict,
) -> dict:
    """
    Call `tavily_client.search(**kwargs)` with one transient retry and graceful
    fallback if a custom/older client does not recognize newer optional flags.
    """
    call_kwargs = dict(kwargs)
    last_exc: Exception | None = None

    for attempt in range(2):
        try:
            response = tavily_client.search(**call_kwargs)
            return response if isinstance(response, dict) else {"results": []}
        except (TypeError, BadRequestError) as exc:
            # Gracefully strip newer optional flags if an older proxy/mock rejects them.
            err_text = str(exc)
            stripped = False
            for opt_key in ("include_domains_mode", "include_usage", "topic", "time_range"):
                if opt_key in call_kwargs and opt_key in err_text:
                    call_kwargs.pop(opt_key, None)
                    stripped = True
            if stripped:
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and _is_retryable_tavily_error(exc):
                time.sleep(TAVILY_RETRY_DELAY_SECONDS)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return {"results": []}


def _run_cached_tavily_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    include_domains_mode: str | None = None,
    client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> tuple[list[dict], float, bool]:
    """
    Run a single Tavily search (checking the 6h on-disk cache first).

    Returns:
        Tuple of (results_list, credits_used, cache_hit).
    """
    normalized_query = " ".join(query.split())
    cache_params = {
        "query": normalized_query.lower(),
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "time_range": time_range,
        "include_domains": sorted(include_domains) if include_domains else None,
        "include_domains_mode": include_domains_mode if include_domains else None,
    }
    cache_key = _build_cache_key("search", cache_params)

    if use_cache:
        cached = _cache_get(cache_key, cache_path=cache_path)
        if cached is not None:
            results = cached.get("results", [])
            return (results if isinstance(results, list) else []), 0.0, True

    tavily_client = client or get_tavily_client()
    api_kwargs: dict = {
        "query": normalized_query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": topic,
        "include_usage": True,
    }
    if time_range:
        api_kwargs["time_range"] = time_range
    if include_domains:
        api_kwargs["include_domains"] = list(include_domains)
        if include_domains_mode:
            api_kwargs["include_domains_mode"] = include_domains_mode

    try:
        response = _call_tavily_search_with_retry(tavily_client, api_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Tavily search failed for query {query!r}: {_format_tavily_error(exc)}"
        ) from exc

    results = response.get("results", [])
    if not isinstance(results, list):
        results = []

    default_credits = 2.0 if search_depth == "advanced" else 1.0
    credits_used = _extract_usage_credits(response, default_credits=default_credits)

    if use_cache:
        _cache_set(
            cache_key,
            endpoint="search",
            response={"results": results, "usage": {"credits": credits_used}},
            cache_path=cache_path,
        )

    return results, credits_used, False


def _merge_search_results(
    primary_results: list[dict],
    secondary_results: list[dict],
    limit: int = 5,
) -> list[dict]:
    """
    Interleave and deduplicate results from two Tavily searches (e.g., news + general)
    so both fresh updates and official reference pages stay in the top results.
    """
    merged: list[dict] = []
    seen_urls: set[str] = set()
    max_len = max(len(primary_results), len(secondary_results))

    for idx in range(max_len):
        for group in (primary_results, secondary_results):
            if idx >= len(group):
                continue
            item = group[idx]
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            dedupe_key = url.rstrip("/").lower() if url else f"no-url-{len(merged)}"
            if dedupe_key in seen_urls:
                continue
            seen_urls.add(dedupe_key)
            merged.append(item)
            if len(merged) >= limit:
                return merged

    return merged


def search_tavily_for_claim(
    query: str,
    category: str = "other",
    risk_level: str | None = None,
    max_results: int = 5,
    client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> dict:
    """
    Execute the category-aware Tavily search strategy for a claim:
    - high-risk claims (regulation, price): search_depth="advanced"; others use "basic"
    - price / version / company: topic="news" (recent time_range) + normal search
    - regulation: prefer official/gov domains via include_domains + include_domains_mode="prefer"
    """
    normalized_cat = str(category or "other").strip().lower()
    effective_risk = risk_level or get_risk_level(normalized_cat)
    search_depth = "advanced" if effective_risk == "critical" else "basic"

    # Strategy 1: Time-sensitive price/version/company -> recent news + normal search.
    if normalized_cat in NEWS_DUAL_SEARCH_CATEGORIES:
        news_results: list[dict] = []
        general_results: list[dict] = []
        credits = 0.0
        errors: list[str] = []
        cache_hits: list[bool] = []

        try:
            news_results, news_credits, news_hit = _run_cached_tavily_search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                topic="news",
                time_range=RECENT_NEWS_TIME_RANGE,
                client=client,
                use_cache=use_cache,
                cache_path=cache_path,
            )
            credits += news_credits
            cache_hits.append(news_hit)
        except Exception as exc:
            errors.append(str(exc))

        try:
            general_results, gen_credits, gen_hit = _run_cached_tavily_search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                topic="general",
                client=client,
                use_cache=use_cache,
                cache_path=cache_path,
            )
            credits += gen_credits
            cache_hits.append(gen_hit)
        except Exception as exc:
            errors.append(str(exc))

        if len(errors) == 2:
            raise RuntimeError(errors[0])

        merged = _merge_search_results(news_results, general_results, limit=max_results)
        return {
            "results": merged,
            "credits_used": credits,
            "cache_hit": bool(cache_hits) and all(cache_hits) and not errors,
            "strategy": f"news ({RECENT_NEWS_TIME_RANGE}) + general [{search_depth}]",
            "partial_error": errors[0] if errors else None,
        }

    # Strategy 2: Regulations -> prefer official & government domains with advanced depth.
    if normalized_cat == "regulation":
        results, credits, hit = _run_cached_tavily_search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            topic="general",
            include_domains=REGULATION_INCLUDE_DOMAINS,
            include_domains_mode="prefer",
            client=client,
            use_cache=use_cache,
            cache_path=cache_path,
        )
        return {
            "results": results,
            "credits_used": credits,
            "cache_hit": hit,
            "strategy": f"official-domains prefer [{search_depth}]",
            "partial_error": None,
        }

    # Strategy 3: General / statistic / other -> standard basic search.
    results, credits, hit = _run_cached_tavily_search(
        query=query,
        max_results=max_results,
        search_depth=search_depth,
        topic="general",
        client=client,
        use_cache=use_cache,
        cache_path=cache_path,
    )
    return {
        "results": results,
        "credits_used": credits,
        "cache_hit": hit,
        "strategy": f"general [{search_depth}]",
        "partial_error": None,
    }


def search_tavily(
    query: str,
    max_results: int = 5,
    client: TavilyClient | None = None,
    category: str = "other",
    use_cache: bool = True,
    cache_path: str | None = None,
) -> list[dict]:
    """
    Query the Tavily Search API and return a list of result dicts.

    Args:
        query: Web search query string.
        max_results: Number of search results to return (default: 5).
        client: Optional pre-initialized TavilyClient.
        category: Optional claim category for strategy selection.
        use_cache: Whether to use the 6-hour on-disk cache.
        cache_path: Optional override path for the SQLite cache file.

    Returns:
        List of search result dictionaries from Tavily.

    Raises:
        RuntimeError: If the Tavily API call fails.
    """
    search_outcome = search_tavily_for_claim(
        query=query,
        category=category,
        max_results=max_results,
        client=client,
        use_cache=use_cache,
        cache_path=cache_path,
    )
    return search_outcome["results"]


def select_best_source_url(search_results: list[dict]) -> str | None:
    """
    Pick the single best HTTP(S) URL from Tavily search results for deep extraction,
    prioritizing highest domain authority and breaking ties by Tavily rank.
    """
    candidates: list[tuple[int, int, str]] = []
    for idx, result in enumerate(search_results):
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            continue
        authority_score, _ = _score_single_domain_authority(url)
        candidates.append((authority_score, -idx, url))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def extract_best_source(
    url: str,
    query: str,
    client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> dict:
    """
    Run Tavily Extract on a single URL (with 6h cache and graceful error handling)
    to confirm the exact current figure on the best authoritative page.
    """
    clean_url = str(url or "").strip()
    if not clean_url:
        return {"url": "", "content": "", "credits_used": 0.0, "cache_hit": False, "error": None}

    normalized_query = " ".join((query or "").split())
    cache_params = {
        "url": clean_url,
        "extract_depth": "basic",
        "format": "markdown",
        "query": normalized_query.lower(),
        "chunks_per_source": 3,
    }
    cache_key = _build_cache_key("extract", cache_params)

    if use_cache:
        cached = _cache_get(cache_key, cache_path=cache_path)
        if cached is not None:
            return {
                "url": clean_url,
                "content": str(cached.get("content", "")),
                "credits_used": 0.0,
                "cache_hit": True,
                "error": None,
            }

    tavily_client = client or get_tavily_client()
    extract_kwargs: dict = {
        "urls": [clean_url],
        "extract_depth": "basic",
        "format": "markdown",
        "include_usage": True,
    }
    if normalized_query:
        extract_kwargs["query"] = normalized_query
        extract_kwargs["chunks_per_source"] = 3

    response = None
    last_exc: Exception | None = None

    for attempt in range(2):
        try:
            response = tavily_client.extract(**extract_kwargs)
            last_exc = None
            break
        except TypeError as exc:
            err_text = str(exc)
            stripped = False
            for opt_key in ("chunks_per_source", "query", "include_usage", "format", "extract_depth"):
                if opt_key in extract_kwargs and opt_key in err_text:
                    extract_kwargs.pop(opt_key, None)
                    stripped = True
            if stripped:
                continue
            last_exc = exc
            break
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and _is_retryable_tavily_error(exc):
                time.sleep(TAVILY_RETRY_DELAY_SECONDS)
                continue
            break

    if last_exc is not None:
        return {
            "url": clean_url,
            "content": "",
            "credits_used": 0.0,
            "cache_hit": False,
            "error": _format_tavily_error(last_exc),
        }

    # Unconfigured mocks in tests may return a non-dict object; skip gracefully.
    if not isinstance(response, dict):
        return {"url": clean_url, "content": "", "credits_used": 0.0, "cache_hit": False, "error": None}

    extracted_results = response.get("results", [])
    extracted_text = ""
    if isinstance(extracted_results, list) and extracted_results:
        first = extracted_results[0]
        if isinstance(first, dict):
            raw = first.get("raw_content") or first.get("content") or ""
            extracted_text = str(raw).strip()
            if len(extracted_text) > MAX_EXTRACTED_CHARS:
                extracted_text = extracted_text[:MAX_EXTRACTED_CHARS] + "..."

    # Tavily never charges credits for failed URL extractions.
    default_credits = 0.2 if extracted_text else 0.0
    credits_used = _extract_usage_credits(response, default_credits=default_credits) if extracted_results else 0.0

    if use_cache and extracted_text:
        _cache_set(
            cache_key,
            endpoint="extract",
            response={"url": clean_url, "content": extracted_text, "usage": {"credits": credits_used}},
            cache_path=cache_path,
        )

    return {
        "url": clean_url,
        "content": extracted_text,
        "credits_used": credits_used,
        "cache_hit": False,
        "error": None,
    }


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

    # Safely parse and clamp the LLM's verdict score into 0..100.
    default_scores = {"fresh": 90, "uncertain": 50, "outdated": 15}
    raw_score = parsed.get("score", default_scores[status])
    try:
        score = int(round(float(raw_score)))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = default_scores[status]

    # Parse optional agreeing_sources count; fall back to None if omitted/invalid.
    agreeing_sources: int | None = None
    raw_agreeing = parsed.get("agreeing_sources")
    if raw_agreeing is not None and raw_agreeing != "":
        try:
            agreeing_sources = max(0, int(raw_agreeing))
        except (TypeError, ValueError):
            agreeing_sources = None

    current_info = str(parsed.get("current_info", "")).strip()
    if not current_info:
        current_info = "No current details could be extracted from the sources."

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No detailed reason was provided."

    return {
        "status": status,
        "score": score,
        "agreeing_sources": agreeing_sources,
        "current_info": current_info,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Explainable scoring helpers (recency, agreement, authority, risk, confidence)
# ---------------------------------------------------------------------------
def _parse_published_date(raw_value: object) -> date | None:
    """
    Parse a Tavily published_date string (ISO, datetime, or RFC 2822) into a date.
    """
    if not raw_value:
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    # Handle ISO dates like "2026-10-15" or "2026-10-15T12:00:00Z".
    iso_prefix = text[:10]
    try:
        return date.fromisoformat(iso_prefix)
    except ValueError:
        pass

    # Handle RFC 2822 / HTTP dates like "Tue, 10 Nov 2026 08:00:00 GMT".
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        pass

    for fmt in ("%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def _score_source_recency(
    search_results: list[dict],
    today: date | None = None,
) -> tuple[int, str]:
    """
    Score how recent the Tavily sources are using `published_date` (0-100).

    When no `published_date` is provided by Tavily, returns a neutral 50/100 so
    undated official documentation is not unfairly penalized.
    """
    reference_date = today or date.today()
    dated_scores: list[int] = []
    ages_days: list[int] = []

    for result in search_results:
        if not isinstance(result, dict):
            continue
        pub_date = _parse_published_date(result.get("published_date"))
        if pub_date is None:
            continue

        age_days = max(0, (reference_date - pub_date).days)
        ages_days.append(age_days)

        if age_days <= 30:
            dated_scores.append(100)
        elif age_days <= 90:
            dated_scores.append(85)
        elif age_days <= 180:
            dated_scores.append(70)
        elif age_days <= 365:
            dated_scores.append(55)
        elif age_days <= 730:
            dated_scores.append(35)
        else:
            dated_scores.append(15)

    if not dated_scores:
        return 50, "No published_date metadata on sources (neutral 50/100)"

    avg_score = round(sum(dated_scores) / len(dated_scores))
    avg_age = round(sum(ages_days) / len(ages_days))
    detail = (
        f"{len(dated_scores)}/{len(search_results)} sources dated "
        f"(avg age {avg_age} day{'s' if avg_age != 1 else ''})"
    )
    return avg_score, detail


def _count_agreeing_sources(current_info: str, search_results: list[dict]) -> int:
    """
    Fallback heuristic: count how many search results share key numbers/terms
    with `current_info` when the LLM does not report `agreeing_sources`.
    """
    if not search_results or not current_info:
        return 0

    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", current_info.lower())
    key_tokens = {
        tok for tok in raw_tokens
        if (len(tok) >= 3 or any(ch.isdigit() for ch in tok)) and tok not in STOPWORDS
    }
    if not key_tokens:
        return 0

    matches = 0
    for result in search_results:
        if not isinstance(result, dict):
            continue
        blob = f"{result.get('title', '')} {result.get('content', '')}".lower()
        overlap = sum(1 for tok in key_tokens if tok in blob)
        # Support threshold: at least 2 matching key tokens (or 1 if info is very short).
        needed = 1 if len(key_tokens) <= 2 else min(2, len(key_tokens))
        if overlap >= needed:
            matches += 1

    return matches


def _score_source_agreement(
    search_results: list[dict],
    current_info: str,
    llm_agreeing_count: int | None = None,
) -> tuple[int, str, int]:
    """
    Score how strongly the top search results agree on the same current fact (0-100).
    """
    total = len(search_results)
    if total == 0:
        return 0, "0/0 sources available", 0

    if llm_agreeing_count is not None:
        agreeing = max(0, min(total, llm_agreeing_count))
    else:
        agreeing = _count_agreeing_sources(current_info, search_results)

    score = round((agreeing / total) * 100)
    detail = f"{agreeing}/{total} top results support the same current fact"
    return score, detail, agreeing


def _score_single_domain_authority(url: str) -> tuple[int, str]:
    """
    Classify a single URL's domain authority into high/medium/low tiers.

    Returns:
        Tuple of (score 0-100, tier_label).
    """
    parsed = urlparse(url.strip())
    domain = (parsed.netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path = (parsed.path or "").lower()

    if not domain:
        return 50, "unknown"

    # Low tier: forums, social networks, Q&A, and community blogs.
    if any(domain == d or domain.endswith("." + d) for d in FORUM_OR_SOCIAL_DOMAINS):
        return 25, "forum"
    if domain.startswith(("forum.", "community.", "discuss.")):
        return 25, "forum"

    if any(domain == d or domain.endswith("." + d) for d in BLOG_DOMAINS):
        return 40, "blog"
    if domain.startswith("blog.") or domain.endswith(".github.io") or "/blog/" in path:
        return 45, "blog"

    # High tier: government, academic, standards, official docs/support/IR subdomains.
    if domain.endswith((".gov", ".edu", ".ac.uk")) or ".gov." in domain:
        return 95, "official"
    if domain.startswith(("docs.", "developer.", "developers.", "api.", "support.", "help.", "learn.", "ir.")):
        return 95, "official"
    if any(domain == d or domain.endswith("." + d) for d in OFFICIAL_AUTHORITY_DOMAINS):
        return 90, "official"
    if any(seg in path for seg in ("/docs", "/documentation", "/api/", "/pricing", "/legal")):
        return 85, "official"

    # Standard primary company/news/reference websites.
    return 70, "general"


def _score_source_authority(search_results: list[dict]) -> tuple[int, str]:
    """
    Average authority score (0-100) across the retrieved search results.
    """
    scores: list[int] = []
    tiers: list[str] = []

    for result in search_results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        if not url:
            continue
        dom_score, tier = _score_single_domain_authority(url)
        scores.append(dom_score)
        tiers.append(tier)

    if not scores:
        return 50, "No source URLs available (neutral 50/100)"

    avg_score = round(sum(scores) / len(scores))
    official_cnt = tiers.count("official")
    low_cnt = tiers.count("blog") + tiers.count("forum")

    if official_cnt == len(tiers):
        summary = f"All {official_cnt} sources are official/docs/gov domains"
    elif official_cnt > 0 and low_cnt == 0:
        summary = f"{official_cnt}/{len(tiers)} official/docs domains, rest standard web"
    elif low_cnt > 0 and official_cnt == 0:
        summary = f"Includes {low_cnt}/{len(tiers)} blog/forum sources"
    else:
        summary = f"{official_cnt} official/docs, {low_cnt} blog/forum out of {len(tiers)} sources"

    return avg_score, summary


def get_risk_level(category: str) -> str:
    """
    Determine staleness risk level ('critical', 'moderate', 'low') by claim category.

    Regulation and price claims carry higher real-world risk if stale.
    """
    normalized = str(category or "other").strip().lower()
    if normalized in HIGH_RISK_CATEGORIES:
        return "critical"
    if normalized in MODERATE_RISK_CATEGORIES:
        return "moderate"
    return "low"


def get_confidence_level(
    status: str,
    total_sources: int,
    agreement_score: int,
    authority_score: int,
    has_error: bool = False,
) -> str:
    """
    Determine confidence ('high', 'medium', 'low') from source support and verdict clarity.
    """
    if has_error or total_sources == 0:
        return "low"
    if status != "uncertain" and total_sources >= 3 and agreement_score >= 60 and authority_score >= 60:
        return "high"
    if total_sources >= 2 and (agreement_score >= 40 or status != "uncertain"):
        return "medium"
    return "low"


def compute_freshness_score(
    status: str,
    llm_score: int,
    current_info: str,
    search_results: list[dict],
    llm_agreeing_count: int | None = None,
    today: date | None = None,
) -> dict:
    """
    Compute the weighted 0-100 freshness score, breakdown dict, explanation, and confidence.

    Weights:
    - LLM verdict: 50%
    - Source recency (Tavily published_date): 20%
    - Source agreement: 20%
    - Source authority (official/docs/gov vs blogs/forums): 10%
    """
    llm_score = max(0, min(100, int(llm_score)))
    recency_score, recency_detail = _score_source_recency(search_results, today=today)
    agreement_score, agreement_detail, agreeing_cnt = _score_source_agreement(
        search_results, current_info, llm_agreeing_count=llm_agreeing_count
    )
    authority_score, authority_detail = _score_source_authority(search_results)

    weighted_llm = round(llm_score * SCORE_WEIGHTS["llm_verdict"], 1)
    weighted_recency = round(recency_score * SCORE_WEIGHTS["source_recency"], 1)
    weighted_agreement = round(agreement_score * SCORE_WEIGHTS["source_agreement"], 1)
    weighted_authority = round(authority_score * SCORE_WEIGHTS["source_authority"], 1)

    final_score = round(weighted_llm + weighted_recency + weighted_agreement + weighted_authority)
    final_score = max(0, min(100, final_score))

    score_breakdown = {
        "llm_verdict": {
            "score": llm_score,
            "weight": SCORE_WEIGHTS["llm_verdict"],
            "weighted_points": weighted_llm,
            "detail": f"LLM judged '{status}' ({llm_score}/100)",
        },
        "source_recency": {
            "score": recency_score,
            "weight": SCORE_WEIGHTS["source_recency"],
            "weighted_points": weighted_recency,
            "detail": recency_detail,
        },
        "source_agreement": {
            "score": agreement_score,
            "weight": SCORE_WEIGHTS["source_agreement"],
            "weighted_points": weighted_agreement,
            "detail": agreement_detail,
        },
        "source_authority": {
            "score": authority_score,
            "weight": SCORE_WEIGHTS["source_authority"],
            "weighted_points": weighted_authority,
            "detail": authority_detail,
        },
        "final_score": final_score,
    }

    total = len(search_results)
    why_this_score = (
        f"Score {final_score}/100: LLM judged '{status}' ({llm_score}/100), "
        f"{agreeing_cnt}/{total} sources agree, recency scored {recency_score}/100, "
        f"and domain authority scored {authority_score}/100."
    )

    confidence = get_confidence_level(
        status=status,
        total_sources=total,
        agreement_score=agreement_score,
        authority_score=authority_score,
    )

    return {
        "score": final_score,
        "score_breakdown": score_breakdown,
        "why_this_score": why_this_score,
        "confidence": confidence,
    }


def _build_fallback_breakdown(
    llm_score: int,
    recency_score: int,
    recency_detail: str,
    agreement_score: int,
    authority_score: int,
    authority_detail: str,
    error_note: str,
) -> tuple[int, dict]:
    """
    Build a valid score_breakdown when Tavily or LLM evaluation fails.
    """
    weighted_llm = round(llm_score * SCORE_WEIGHTS["llm_verdict"], 1)
    weighted_recency = round(recency_score * SCORE_WEIGHTS["source_recency"], 1)
    weighted_agreement = round(agreement_score * SCORE_WEIGHTS["source_agreement"], 1)
    weighted_authority = round(authority_score * SCORE_WEIGHTS["source_authority"], 1)

    final_score = max(0, min(100, round(weighted_llm + weighted_recency + weighted_agreement + weighted_authority)))
    breakdown = {
        "llm_verdict": {
            "score": llm_score,
            "weight": SCORE_WEIGHTS["llm_verdict"],
            "weighted_points": weighted_llm,
            "detail": error_note,
        },
        "source_recency": {
            "score": recency_score,
            "weight": SCORE_WEIGHTS["source_recency"],
            "weighted_points": weighted_recency,
            "detail": recency_detail,
        },
        "source_agreement": {
            "score": agreement_score,
            "weight": SCORE_WEIGHTS["source_agreement"],
            "weighted_points": weighted_agreement,
            "detail": error_note,
        },
        "source_authority": {
            "score": authority_score,
            "weight": SCORE_WEIGHTS["source_authority"],
            "weighted_points": weighted_authority,
            "detail": authority_detail,
        },
        "final_score": final_score,
    }
    return final_score, breakdown


def verify_claim(
    claim_item: dict[str, str],
    tavily_client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> dict:
    """
    Verify a single extracted claim against live Tavily search + extract results.

    Errors during Tavily search or LLM JSON parsing are caught and returned as
    an 'uncertain' result with an 'error' field so the rest of the check continues.

    Args:
        claim_item: Dict containing 'claim', 'category', and 'search_query'.
        tavily_client: Optional shared TavilyClient instance.
        use_cache: Whether to read/write the 6-hour on-disk Tavily cache.
        cache_path: Optional override path for the SQLite cache file.

    Returns:
        Dict with claim, category, search_query, status, score, score_breakdown,
        why_this_score, confidence, risk_level, current_info, reason, sources (top 3),
        tavily_credits_used, cache_hit, search_strategy, extracted_url, model,
        and optional error message.
    """
    claim_text = claim_item.get("claim", "").strip()
    category = claim_item.get("category", "other").strip()
    search_query = claim_item.get("search_query", "").strip() or claim_text
    risk_level = get_risk_level(category)

    # Step 1: Run category-appropriate Tavily search (cached for 6 hours on disk).
    try:
        search_meta = search_tavily_for_claim(
            query=search_query,
            category=category,
            risk_level=risk_level,
            max_results=5,
            client=tavily_client,
            use_cache=use_cache,
            cache_path=cache_path,
        )
    except Exception as exc:
        friendly_tavily_err = _format_tavily_error(exc)
        score, breakdown = _build_fallback_breakdown(
            llm_score=50,
            recency_score=50,
            recency_detail="Unavailable due to Tavily search failure",
            agreement_score=50,
            authority_score=50,
            authority_detail="Unavailable due to Tavily search failure",
            error_note="Tavily search failed (neutral fallback)",
        )
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": f"Score {score}/100: Live web search failed, so scoring components defaulted cleanly.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "Unable to retrieve live web sources.",
            "reason": friendly_tavily_err,
            "sources": [],
            "tavily_credits_used": 0.0,
            "cache_hit": False,
            "search_strategy": None,
            "extracted_url": None,
            "error": friendly_tavily_err,
        }

    search_results = search_meta["results"]
    total_credits = float(search_meta["credits_used"])
    search_cache_hit = bool(search_meta["cache_hit"])
    search_strategy = search_meta["strategy"]
    top_sources = _extract_top_sources(search_results, limit=3)

    if not search_results:
        score, breakdown = _build_fallback_breakdown(
            llm_score=50,
            recency_score=50,
            recency_detail="No search results returned",
            agreement_score=0,
            authority_score=50,
            authority_detail="No search results returned",
            error_note="0 results returned",
        )
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": f"Score {score}/100: Tavily returned 0 results to confirm or refute this claim.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "No relevant live search results were found.",
            "reason": "Tavily returned 0 results for this search query.",
            "sources": [],
            "tavily_credits_used": round(total_credits, 2),
            "cache_hit": search_cache_hit,
            "search_strategy": search_strategy,
            "extracted_url": None,
        }

    # Step 2: Run Tavily Extract on the single best source URL to confirm exact figures.
    best_url = select_best_source_url(search_results)
    extract_info = (
        extract_best_source(
            url=best_url,
            query=search_query,
            client=tavily_client,
            use_cache=use_cache,
            cache_path=cache_path,
        )
        if best_url
        else {"url": "", "content": "", "credits_used": 0.0, "cache_hit": False, "error": None}
    )
    total_credits += float(extract_info.get("credits_used", 0.0))
    overall_cache_hit = search_cache_hit and (
        extract_info.get("cache_hit", False) or not best_url
    )

    # Step 3: Ask the LLM to judge freshness using today's date + snippets + extracted content.
    today_date = date.today()
    today_str = today_date.isoformat()
    formatted_sources = _format_search_results(search_results)
    extracted_section = ""
    if extract_info.get("content"):
        extracted_section = (
            f"\n\nDeep-extracted content from best source ({extract_info['url']}) "
            f"to confirm exact figures:\n{extract_info['content']}"
        )

    user_prompt = (
        f"Today's date: {today_str}\n"
        f"Claim category: {category}\n"
        f"Claim to verify: {claim_text}\n\n"
        f"Live web search results:\n{formatted_sources}"
        f"{extracted_section}"
    )

    model_used: str | None = None
    gateway_metrics: dict | None = None
    pop_last_gateway_metrics()
    llm_t0 = time.perf_counter()
    try:
        raw_response, model_used = call_llm(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )
        gateway_metrics = _capture_call_metrics(
            "Verify claim",
            model_used,
            (time.perf_counter() - llm_t0) * 1000.0,
        )
        parsed = parse_json_response(raw_response)
        judgement = _normalize_judgement(parsed)
    except Exception as exc:
        if gateway_metrics is None and model_used:
            gateway_metrics = _capture_call_metrics(
                "Verify claim",
                model_used,
                (time.perf_counter() - llm_t0) * 1000.0,
            )
        # Even if the LLM step fails, still score recency/authority & attach top 3 URLs.
        err_lower = str(exc).lower()
        if isinstance(exc, ValueError) and "json" in err_lower:
            friendly_err = "The LLM returned invalid non-JSON formatting while judging this claim."
        else:
            friendly_err = format_llm_error(exc)
        rec_score, rec_detail = _score_source_recency(search_results, today=today_date)
        auth_score, auth_detail = _score_source_authority(search_results)
        score, breakdown = _build_fallback_breakdown(
            llm_score=50,
            recency_score=rec_score,
            recency_detail=rec_detail,
            agreement_score=50,
            authority_score=auth_score,
            authority_detail=auth_detail,
            error_note="LLM evaluation failed (neutral 50/100)",
        )
        return {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
            "status": "uncertain",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": f"Score {score}/100: LLM verdict failed, so verdict and agreement defaulted to 50/100 alongside source signals.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "Could not judge freshness automatically due to an LLM error.",
            "reason": friendly_err,
            "sources": top_sources,
            "tavily_credits_used": round(total_credits, 2),
            "cache_hit": overall_cache_hit,
            "search_strategy": search_strategy,
            "extracted_url": extract_info["url"] if extract_info.get("content") else best_url,
            "gateway_metrics": gateway_metrics,
            "fallback_used": bool(gateway_metrics.get("fallback_used")) if gateway_metrics else False,
            "primary_model": gateway_metrics.get("primary_model") if gateway_metrics else None,
            "model": model_used,
            "error": friendly_err,
        }

    # Step 4: Compute explainable 0-100 composite score & metadata.
    scoring = compute_freshness_score(
        status=judgement["status"],
        llm_score=judgement["score"],
        current_info=judgement["current_info"],
        search_results=search_results,
        llm_agreeing_count=judgement["agreeing_sources"],
        today=today_date,
    )

    return {
        "claim": claim_text,
        "category": category,
        "search_query": search_query,
        "status": judgement["status"],
        "score": scoring["score"],
        "score_breakdown": scoring["score_breakdown"],
        "why_this_score": scoring["why_this_score"],
        "confidence": scoring["confidence"],
        "risk_level": risk_level,
        "current_info": judgement["current_info"],
        "reason": judgement["reason"],
        "sources": top_sources,
        "tavily_credits_used": round(total_credits, 2),
        "cache_hit": overall_cache_hit,
        "search_strategy": search_strategy,
        "extracted_url": extract_info["url"] if extract_info.get("content") else best_url,
        "search_warning": search_meta.get("partial_error"),
        "gateway_metrics": gateway_metrics,
        "fallback_used": bool(gateway_metrics.get("fallback_used")) if gateway_metrics else False,
        "primary_model": gateway_metrics.get("primary_model") if gateway_metrics else None,
        "model": model_used,
    }


def verify_claims(
    claims: list[dict[str, str]],
    progress_callback: Callable[[int, int, dict], None] | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
) -> list[dict]:
    """
    Verify a list of claims against live web sources.

    Args:
        claims: List of claim dicts from extractor.extract_claims().
        progress_callback: Optional callback invoked as callback(index, total, result).
        use_cache: Whether to use the 6-hour on-disk Tavily cache.
        cache_path: Optional override path for the SQLite cache file.

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
        result = verify_claim(
            claim_item,
            tavily_client=tavily_client,
            use_cache=use_cache,
            cache_path=cache_path,
        )
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

    pop_last_gateway_metrics()
    llm_t0 = time.perf_counter()
    rewritten, model_used = call_llm(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
    )
    _verifier_state.last_metrics = _capture_call_metrics(
        "Rewrite answer",
        model_used,
        (time.perf_counter() - llm_t0) * 1000.0,
    )
    return rewritten, model_used
