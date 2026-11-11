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

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import date, datetime
from email.utils import parsedate_to_datetime
import hashlib
import inspect
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

import llm as _llm_mod
from llm import (
    _get_config_value,
    call_llm,
    format_llm_error,
    get_dual_judge_models,
    get_task_routing_summary,
    parse_json_response,
    pop_last_gateway_metrics,
    set_simulate_outage,
)

_verifier_state = threading.local()
_CACHE_LOCK = threading.RLock()


def get_last_gateway_metrics() -> dict | None:
    """Return telemetry recorded during the most recent `generate_corrected_answer` call."""
    return getattr(_verifier_state, "last_metrics", None)


def _capture_call_metrics(
    operation: str,
    model_used: str | None,
    elapsed_ms: float,
    task: str | None = None,
) -> dict | None:
    """Capture gateway telemetry recorded by `call_llm` (or build a minimal fallback)."""
    metrics = pop_last_gateway_metrics()
    if metrics is None:
        if not model_used:
            return None
        metrics = {
            "task": task or "default",
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
    if task and (not metrics.get("task") or metrics.get("task") == "default"):
        metrics["task"] = task
    metrics["operation"] = operation
    return metrics


def _invoke_call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    simulate_outage: bool | None = None,
    task: str | None = None,
    model_override: str | None = None,
) -> tuple[str, str]:
    """
    Invoke `llm.call_llm` with task-routing/override args when supported, while
    remaining 100% compatible with 3-arg or 4-arg callable mocks in unit tests.
    """
    side_effect = getattr(call_llm, "side_effect", None)
    sig_target = side_effect if callable(side_effect) else call_llm
    kwargs: dict = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
    }
    try:
        params = inspect.signature(sig_target).parameters
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params = {}
        accepts_varkw = True

    if simulate_outage is not None and (accepts_varkw or "simulate_outage" in params):
        kwargs["simulate_outage"] = simulate_outage
    if task is not None and (accepts_varkw or "task" in params):
        kwargs["task"] = task
    if model_override is not None and (accepts_varkw or "model_override" in params):
        kwargs["model_override"] = model_override

    try:
        return call_llm(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )


def _same_model_name(model_a: str | None, model_b: str | None) -> bool:
    """Case-insensitive equality/suffix comparison between two model labels."""
    a = str(model_a or "").strip().lower()
    b = str(model_b or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if "/" in a and "/" in b and a.split("/", 1)[0] != b.split("/", 1)[0]:
        return False
    return a.split("/", 1)[-1] == b.split("/", 1)[-1]

# Load environment variables from .env if present.
load_dotenv()

VALID_STATUSES = ("fresh", "outdated", "uncertain", "unverifiable")

# Enforced score bands per status (evidence strength never pushes a score outside its status band).
STATUS_SCORE_BANDS = {
    "fresh": (75, 100),
    "uncertain": (40, 74),
    "outdated": (0, 39),
}

# Weights for the 0-100 evidence-strength breakdown (sum = 1.0).
SCORE_WEIGHTS = {
    "llm_verdict": 0.50,
    "source_recency": 0.20,
    "source_agreement": 0.20,
    "source_authority": 0.10,
}

# Category-based risk levels (wrong legal/pricing info has higher real-world impact).
HIGH_RISK_CATEGORIES = {"regulation", "price"}
MODERATE_RISK_CATEGORIES = {"version", "company", "statistic"}

# Risk weights used to compute the overall freshness score (regulation/price count 2x version).
RISK_SCORE_WEIGHTS = {
    "critical": 3.0,
    "moderate": 1.5,
    "low": 1.0,
}

# Multi-label public suffixes preserved when deduplicating independent domains (e.g., gst.gov.in vs cbic.gov.in).
MULTI_PART_PUBLIC_SUFFIXES = {
    "gov.in", "nic.in", "org.in", "co.in", "ac.in", "res.in",
    "gov.uk", "co.uk", "org.uk", "ac.uk", "ltd.uk",
    "gov.au", "com.au", "org.au", "edu.au",
    "gc.ca", "gov.sg", "com.sg", "edu.sg",
    "co.jp", "ne.jp", "or.jp", "go.jp",
    "com.br", "gov.br",
}

# Tavily category strategy and on-disk cache configuration (TTL = 6 hours).
DEFAULT_CACHE_PATH = ".tavily_cache.db"
CACHE_TTL_SECONDS = 6 * 60 * 60
NEWS_DUAL_SEARCH_CATEGORIES = {"price", "version", "company"}
RECENT_NEWS_TIME_RANGE = "month"
TAVILY_RETRY_DELAY_SECONDS = 1.0
MAX_EXTRACTED_CHARS = 2500
MAX_ARTICLE_EXTRACT_CHARS = 12000
MAX_SNIPPET_WORDS = 25
PARALLEL_VERIFICATION_WORKERS = 3

# Configurable map of entities/products/jurisdictions -> official primary domains.
# Can be extended at runtime via ENTITY_DOMAIN_MAP (JSON object) in .env / environment.
ENTITY_OFFICIAL_DOMAINS: dict[str, list[str]] = {
    "openai": ["openai.com", "platform.openai.com", "help.openai.com"],
    "gpt-4": ["openai.com", "platform.openai.com", "help.openai.com"],
    "gpt-5": ["openai.com", "platform.openai.com", "help.openai.com"],
    "chatgpt": ["openai.com", "help.openai.com"],
    "apple": ["apple.com", "support.apple.com", "developer.apple.com"],
    "iphone": ["apple.com", "support.apple.com"],
    "ipad": ["apple.com", "support.apple.com"],
    "macbook": ["apple.com", "support.apple.com"],
    "ios": ["apple.com", "support.apple.com", "developer.apple.com"],
    "macos": ["apple.com", "support.apple.com", "developer.apple.com"],
    "python": ["python.org", "docs.python.org", "peps.python.org", "pypi.org"],
    "pypi": ["pypi.org", "python.org"],
    "anthropic": ["anthropic.com", "docs.anthropic.com", "support.anthropic.com"],
    "claude": ["anthropic.com", "docs.anthropic.com"],
    "google": ["google.com", "ai.google.dev", "blog.google", "support.google.com"],
    "gemini": ["ai.google.dev", "gemini.google.com", "blog.google"],
    "android": ["android.com", "developer.android.com"],
    "microsoft": ["microsoft.com", "learn.microsoft.com"],
    "azure": ["azure.microsoft.com", "learn.microsoft.com"],
    "github": ["github.com", "docs.github.com"],
    "streamlit": ["streamlit.io", "docs.streamlit.io"],
    "tavily": ["tavily.com", "docs.tavily.com"],
    "fastapi": ["fastapi.tiangolo.com"],
    "nvidia": ["nvidia.com", "developer.nvidia.com", "nvidianews.nvidia.com"],
    "tesla": ["tesla.com", "ir.tesla.com"],
    "meta": ["meta.com", "about.meta.com", "ai.meta.com"],
    "amazon": ["aboutamazon.com", "aws.amazon.com", "amazon.com"],
    "aws": ["aws.amazon.com"],
    "stripe": ["stripe.com", "docs.stripe.com"],
    "twitter": ["x.com", "help.x.com", "about.x.com"],
    "formerly twitter": ["x.com", "help.x.com", "about.x.com"],
    "yaccarino": ["x.com", "help.x.com", "about.x.com"],
    "react": ["react.dev"],
    "next.js": ["nextjs.org", "vercel.com"],
    "nextjs": ["nextjs.org", "vercel.com"],
    "node.js": ["nodejs.org"],
    "nodejs": ["nodejs.org"],
    "typescript": ["typescriptlang.org"],
    "rust": ["rust-lang.org"],
    "golang": ["go.dev"],
    "gst": ["gstcouncil.gov.in", "gst.gov.in", "cbic.gov.in", "pib.gov.in"],
    "goods and services tax": ["gstcouncil.gov.in", "gst.gov.in", "cbic.gov.in", "pib.gov.in"],
    "gst council": ["gstcouncil.gov.in", "gst.gov.in", "cbic.gov.in"],
    "cbic": ["cbic.gov.in", "gstcouncil.gov.in", "gst.gov.in"],
    "india": ["*.gov.in", "pib.gov.in", "rbi.org.in", "sebi.gov.in", "incometax.gov.in"],
    "rbi": ["rbi.org.in", "*.gov.in"],
    "sebi": ["sebi.gov.in", "*.gov.in"],
    "income tax": ["incometax.gov.in", "*.gov.in", "irs.gov"],
    "sec": ["sec.gov"],
    "irs": ["irs.gov"],
    "ftc": ["ftc.gov"],
    "fda": ["fda.gov"],
    "cdc": ["cdc.gov"],
    "european union": ["europa.eu", "eur-lex.europa.eu"],
    "gdpr": ["europa.eu", "eur-lex.europa.eu"],
    "ai act": ["europa.eu", "eur-lex.europa.eu", "digital-strategy.ec.europa.eu"],
    "uk": ["gov.uk", "legislation.gov.uk"],
    "hmrc": ["gov.uk"],
    "who": ["who.int"],
}

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
    "gstcouncil.gov.in",
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

# Official primary company, government, and documentation domains.
OFFICIAL_AUTHORITY_DOMAINS = {
    "python.org",
    "pypi.org",
    "w3.org",
    "ietf.org",
    "who.int",
    "europa.eu",
    "sec.gov",
    "irs.gov",
    "ftc.gov",
    "fda.gov",
    "cdc.gov",
    "gstcouncil.gov.in",
    "gst.gov.in",
    "cbic.gov.in",
    "rbi.org.in",
    "sebi.gov.in",
    "incometax.gov.in",
    "pib.gov.in",
    "legislation.gov.uk",
    "gov.uk",
    "canada.ca",
    "apple.com",
    "openai.com",
    "microsoft.com",
    "google.com",
    "ai.google.dev",
    "github.com",
    "anthropic.com",
    "streamlit.io",
    "tavily.com",
    "fastapi.tiangolo.com",
    "nvidia.com",
    "tesla.com",
    "meta.com",
    "stripe.com",
    "nodejs.org",
    "react.dev",
    "nextjs.org",
    "vercel.com",
    "typescriptlang.org",
    "rust-lang.org",
    "go.dev",
}

NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "bbc.co.uk",
    "cnbc.com",
    "forbes.com",
    "techcrunch.com",
    "theverge.com",
    "arstechnica.com",
    "wired.com",
    "macrumors.com",
    "9to5mac.com",
    "engadget.com",
    "zdnet.com",
    "venturebeat.com",
    "theinformation.com",
    "axios.com",
    "politico.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "thehindu.com",
    "indianexpress.com",
    "ndtv.com",
    "moneycontrol.com",
    "timesofindia.indiatimes.com",
    "infoworld.com",
    "theregister.com",
    "cnn.com",
    "theguardian.com",
    "wikipedia.org",
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
    "discord.com",
    "threads.net",
}

BLOG_DOMAINS = {
    "medium.com",
    "substack.com",
    "dev.to",
    "hashnode.dev",
    "hashnode.com",
    "wordpress.com",
    "blogspot.com",
    "tumblr.com",
    "ghost.io",
    "beehiiv.com",
    "mirror.xyz",
}

# Common English stopwords ignored when checking keyword overlap across snippets.
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "have", "has", "had", "not", "but", "into", "about", "current", "currently",
    "today", "latest", "now", "more", "than", "per", "its", "their", "been",
}

VERIFICATION_SYSTEM_PROMPT = """You are an objective fact-checking judge.
Given today's date, a specific factual claim from an older LLM answer, and live
web search results (labeled [Primary], [News], [Blog], or [Forum]), determine
whether the claim is still accurate today.

Return ONLY a valid JSON object (no markdown fences or extra commentary) with
these exact keys:
- "status": Must be one of "fresh", "outdated", "uncertain", or "unverifiable".
  * "fresh": The web results clearly confirm the claim is still accurate today.
  * "outdated": The web results clearly show the claim is no longer accurate (e.g., new price,
    newer software version, leadership change, new law, updated statistic).
  * "uncertain": Credible sources conflict/disagree, or evidence is partial/ambiguous.
  * "unverifiable": No credible or relevant evidence in the search results can confirm or refute
    the claim. Never guess — choose "unverifiable" instead of guessing.
- "score": An integer matching the status band, or null for "unverifiable":
  * 75 to 100 for "fresh"
  * 40 to 74 for "uncertain"
  * 0 to 39 for "outdated"
  * null for "unverifiable"
- "agreeing_sources": An integer count (0 to N) of how many of the provided
  search results support the same current fact stated in "current_info".
- "conflicting_sources": Boolean (true/false) — true if retrieved credible sources
  actively contradict each other on the current fact (in which case set status="uncertain"
  and note "Sources disagree" in reason).
- "current_info": A concise statement of the current, up-to-date facts found in
  the search results (or state clearly that sources disagree / no reliable evidence was found).
- "current_value": Short, exact current value/figure/name/rate today (string, or null if unknown/unverifiable; e.g., "5% and 18% (+ 40% demerit rate)", "$999 / iPhone 17 Pro").
- "effective_date": When the current fact took effect (e.g., "2025-09-22" or "September 2025", or null if not stated in sources).
- "change_summary": Brief delta describing what changed vs. the claim (e.g., "12% and 28% slabs removed, 40% added", or null if fresh/unverifiable).
- "stale_since": Earliest date, month, or release when the claimed fact stopped being accurate (e.g., "2025-09-22" or "September 2025", or null if fresh/unknown).
- "reason": A brief 1-2 sentence explanation comparing the claim to the live sources.
"""

REWRITE_SYSTEM_PROMPT = """You are a helpful editor updating an outdated AI answer.
Rewrite the original answer so that all outdated facts are replaced with the
verified current information provided below.

Guidelines:
- Preserve the helpful tone, clarity, and structure of the original answer.
- Keep claims marked "fresh" intact.
- Replace claims marked "outdated" using the verified "current_info", "current_value", and "change_summary".
- For claims marked "uncertain" or "unverifiable", qualify them carefully (note what could not be confirmed) without inventing unverified facts.
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
    with _CACHE_LOCK:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0)
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
    """Fetch a fresh cached Tavily JSON response if present and within TTL (thread-safe)."""
    path = get_cache_path(cache_path)
    current_time = time.time() if now is None else now
    with _CACHE_LOCK:
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
    """Persist a valid Tavily response dictionary into the on-disk SQLite cache (thread-safe)."""
    if not isinstance(response, dict):
        return
    path = get_cache_path(cache_path)
    current_time = time.time() if now is None else now
    with _CACHE_LOCK:
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
    """Delete all cached Tavily entries and return the number of rows removed (thread-safe)."""
    path = get_cache_path(cache_path)
    with _CACHE_LOCK:
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


def _normalize_host(url_or_host: object) -> str:
    """Extract normalized lowercase hostname (without port or leading 'www.')."""
    raw = str(url_or_host or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw.lstrip("*.")
    host = (urlparse(raw).netloc or "").split(":")[0].lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _host_matches_pattern(host: str, pattern: object) -> bool:
    """Return True if normalized `host` matches a domain or '*.suffix' wildcard pattern."""
    clean_pat = str(pattern or "").strip().lower()
    if not host or not clean_pat:
        return False
    if clean_pat.startswith("*."):
        suffix = clean_pat[2:]
        return bool(suffix) and (host == suffix or host.endswith("." + suffix))
    if clean_pat.startswith("www."):
        clean_pat = clean_pat[4:]
    return host == clean_pat or host.endswith("." + clean_pat)


def get_entity_official_domains_map(
    custom_map: dict[str, list[str] | str] | None = None,
) -> dict[str, list[str]]:
    """
    Build the configurable entity-to-official-domain map, combining built-in
    entries, optional ENTITY_DOMAIN_MAP JSON in .env, and any caller override.
    """
    merged: dict[str, list[str]] = {
        k.lower(): list(v) for k, v in ENTITY_OFFICIAL_DOMAINS.items()
    }

    env_raw = _get_config_value("ENTITY_DOMAIN_MAP")
    if env_raw:
        try:
            env_parsed = json.loads(env_raw)
        except (TypeError, ValueError):
            env_parsed = None
        if isinstance(env_parsed, dict):
            for ent, doms in env_parsed.items():
                key = str(ent or "").strip().lower()
                if not key:
                    continue
                if isinstance(doms, str):
                    dom_list = [d.strip() for d in doms.split(",") if d.strip()]
                elif isinstance(doms, (list, tuple)):
                    dom_list = [str(d).strip() for d in doms if str(d).strip()]
                else:
                    continue
                if dom_list:
                    merged[key] = dom_list

    if isinstance(custom_map, dict):
        for ent, doms in custom_map.items():
            key = str(ent or "").strip().lower()
            if not key:
                continue
            if isinstance(doms, str):
                dom_list = [d.strip() for d in doms.split(",") if d.strip()]
            elif isinstance(doms, (list, tuple)):
                dom_list = [str(d).strip() for d in doms if str(d).strip()]
            else:
                continue
            if dom_list:
                merged[key] = dom_list

    return merged


def resolve_entity_official_domains(
    text: str,
    entity_domains: dict[str, list[str] | str] | None = None,
) -> list[str]:
    """
    Match entities mentioned in `text` against the official-domain map using
    longest-match boundary regexes and return deduplicated `include_domains`.
    """
    haystack = " ".join(str(text or "").split()).lower()
    if not haystack:
        return []

    domain_map = get_entity_official_domains_map(custom_map=entity_domains)
    resolved: list[str] = []
    seen: set[str] = set()

    for entity_key in sorted(domain_map.keys(), key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(entity_key)}(?![a-z0-9])"
        if re.search(pattern, haystack):
            for dom in domain_map[entity_key]:
                clean_dom = str(dom or "").strip().lower()
                if clean_dom and clean_dom not in seen:
                    seen.add(clean_dom)
                    resolved.append(clean_dom)

    return resolved


def truncate_snippet_words(text: object, max_words: int = MAX_SNIPPET_WORDS) -> str:
    """Return a clean short evidence snippet capped at `max_words` words."""
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return ""
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]) + "..."


def classify_source_type(
    url: object,
    entity_domains: list[str] | None = None,
    retrieved_from_primary: bool = False,
    result_topic: str | None = None,
) -> str:
    """
    Classify a source URL into one of: 'Primary', 'News', 'Blog', or 'Forum'.
    """
    raw_url = str(url or "").strip()
    parsed = urlparse(raw_url)
    host = (parsed.netloc or "").split(":")[0].lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    path = (parsed.path or "").lower()

    if not host:
        return "Primary" if retrieved_from_primary else "News"

    # Social / Q&A / Forum platforms first (allow official help/about subdomains as Primary).
    is_forum_domain = any(_host_matches_pattern(host, d) for d in FORUM_OR_SOCIAL_DOMAINS)
    is_forum_host = host.startswith(("forum.", "community.", "discuss."))
    is_official_social_sub = host.startswith(("help.", "about.", "investor.", "business.", "developers."))
    if (is_forum_domain and not is_official_social_sub) or is_forum_host or any(
        seg in path for seg in ("/forum/", "/forums/", "/discussions/", "/threads/")
    ):
        return "Forum"

    # Dedicated blogging / newsletter hosts and blog paths on non-entity domains.
    matches_entity = any(_host_matches_pattern(host, d) for d in (entity_domains or []))
    is_official_primary_host = (
        host.endswith((".gov", ".edu", ".ac.uk", ".nic.in", ".int"))
        or ".gov." in host
        or host.startswith(("docs.", "developer.", "developers.", "api.", "support.", "help.", "learn.", "ir.", "peps."))
        or any(_host_matches_pattern(host, d) for d in OFFICIAL_AUTHORITY_DOMAINS)
    )

    if any(_host_matches_pattern(host, d) for d in BLOG_DOMAINS):
        return "Blog"
    if not matches_entity and not is_official_primary_host:
        if host.startswith("blog.") or host.endswith(".github.io") or "/blog/" in path:
            return "Blog"

    # Reputable / explicit news outlets.
    if any(_host_matches_pattern(host, d) for d in NEWS_DOMAINS):
        return "News"

    # Official / entity-mapped primary sources.
    if matches_entity or is_official_primary_host or retrieved_from_primary:
        return "Primary"

    if host.startswith("news.") or "/news/" in path or result_topic == "news":
        return "News"

    if any(seg in path for seg in ("/docs", "/documentation", "/api/", "/pricing", "/legal")):
        return "Primary"

    if "/@" in path or "/notes/" in path:
        return "Blog"

    return "News"


def _merge_search_results(
    primary_results: list[dict],
    secondary_results: list[dict],
    limit: int = 5,
    prioritize_primary: bool = False,
) -> list[dict]:
    """
    Merge and deduplicate results from two Tavily searches.
    When `prioritize_primary=True`, keep primary-domain hits ahead of general results
    while still filling up to `limit` with non-primary sources for corroboration/conflict checks.
    """
    merged: list[dict] = []
    seen_urls: set[str] = set()

    if prioritize_primary:
        for group in (primary_results, secondary_results):
            for item in group:
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
    claim: str = "",
    entity_domains: dict[str, list[str] | str] | None = None,
) -> dict:
    """
    Execute the evidence-strengthened Tavily search strategy for a claim:
    - High-risk claims (regulation, price): search_depth="advanced"; others use "basic".
    - Primary-source first: when the claim's entity maps to official domains, run
      Search #1 restricted with include_domains=[...], then Search #2 general search.
    - Fallback when no specific entity maps:
      * price / version / company -> topic="news" (recent time_range) + general
      * regulation -> official/gov include_domains with include_domains_mode="prefer"
      * statistic / other -> topic="general"
    """
    normalized_cat = str(category or "other").strip().lower()
    effective_risk = risk_level or get_risk_level(normalized_cat)
    search_depth = "advanced" if effective_risk == "critical" else "basic"

    matched_entity_domains = resolve_entity_official_domains(
        f"{claim} {query}",
        entity_domains=entity_domains,
    )

    # Strategy 0: Primary-source first (entity include_domains search + general search).
    if matched_entity_domains:
        primary_results: list[dict] = []
        general_results: list[dict] = []
        credits = 0.0
        errors: list[str] = []
        cache_hits: list[bool] = []

        try:
            primary_results, prim_credits, prim_hit = _run_cached_tavily_search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                topic="general",
                include_domains=matched_entity_domains,
                client=client,
                use_cache=use_cache,
                cache_path=cache_path,
            )
            credits += prim_credits
            cache_hits.append(prim_hit)
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

        primary_urls = {
            str(r.get("url", "")).strip().rstrip("/").lower()
            for r in primary_results
            if isinstance(r, dict) and str(r.get("url", "")).strip()
        }
        # Keep top primary hits up front alongside independent general hits.
        merged = _merge_search_results(
            primary_results[:3] if len(general_results) > 0 else primary_results,
            general_results,
            limit=max_results,
            prioritize_primary=True,
        )
        return {
            "results": merged,
            "credits_used": credits,
            "cache_hit": bool(cache_hits) and all(cache_hits) and not errors,
            "strategy": f"primary-first ({len(matched_entity_domains)} domains) + general [{search_depth}]",
            "partial_error": errors[0] if errors else None,
            "entity_domains": matched_entity_domains,
            "primary_urls": primary_urls,
            "news_urls": set(),
        }

    # Strategy 1 (fallback without entity match): price/version/company -> news + general.
    if normalized_cat in NEWS_DUAL_SEARCH_CATEGORIES:
        news_results: list[dict] = []
        general_results = []
        credits = 0.0
        errors = []
        cache_hits = []

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

        news_urls = {
            str(r.get("url", "")).strip().rstrip("/").lower()
            for r in news_results
            if isinstance(r, dict) and str(r.get("url", "")).strip()
        }
        merged = _merge_search_results(news_results, general_results, limit=max_results)
        return {
            "results": merged,
            "credits_used": credits,
            "cache_hit": bool(cache_hits) and all(cache_hits) and not errors,
            "strategy": f"news ({RECENT_NEWS_TIME_RANGE}) + general [{search_depth}]",
            "partial_error": errors[0] if errors else None,
            "entity_domains": [],
            "primary_urls": set(),
            "news_urls": news_urls,
        }

    # Strategy 2 (fallback without entity match): Regulations -> prefer official/gov domains.
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
            "entity_domains": [],
            "primary_urls": set(),
            "news_urls": set(),
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
        "entity_domains": [],
        "primary_urls": set(),
        "news_urls": set(),
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


def select_best_source_url(
    search_results: list[dict],
    entity_domains: list[str] | None = None,
    primary_urls: set[str] | None = None,
) -> str | None:
    """
    Pick the single best HTTP(S) URL from Tavily search results for deep extraction,
    prioritizing Primary / official entity domains highest and breaking ties by Tavily rank.
    """
    candidates: list[tuple[int, int, str]] = []
    for idx, result in enumerate(search_results):
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            continue
        is_prim = url.rstrip("/").lower() in (primary_urls or set())
        src_type = classify_source_type(
            url,
            entity_domains=entity_domains,
            retrieved_from_primary=is_prim,
        )
        authority_score, _ = _score_single_domain_authority(url, entity_domains=entity_domains)
        type_bonus = 20 if src_type == "Primary" else 0
        candidates.append((authority_score + type_bonus, -idx, url))

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


def extract_article_from_url(
    url: str,
    client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
    max_chars: int = MAX_ARTICLE_EXTRACT_CHARS,
) -> dict:
    """
    Use Tavily Extract to fetch clean article or blog text from a user-supplied URL,
    reusing the 6-hour on-disk SQLite cache and reporting Tavily credit usage.

    Returns:
        Dict with keys: url, title, content, truncated, credits_used, cache_hit.

    Raises:
        ValueError: If `url` is missing or not a valid http(s) URL, or TAVILY_API_KEY is missing.
        RuntimeError: If Tavily fails to reach or extract readable text from the URL.
    """
    clean_url = str(url or "").strip()
    parsed = urlparse(clean_url)
    if not clean_url or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Please enter a valid http:// or https:// article or blog URL.")

    cache_params = {
        "url": clean_url,
        "extract_depth": "basic",
        "format": "markdown",
        "max_chars": int(max_chars),
    }
    cache_key = _build_cache_key("extract_article", cache_params)

    if use_cache:
        cached = _cache_get(cache_key, cache_path=cache_path)
        if cached is not None and str(cached.get("content", "")).strip():
            return {
                "url": clean_url,
                "title": cached.get("title"),
                "content": str(cached.get("content", "")),
                "truncated": bool(cached.get("truncated", False)),
                "credits_used": 0.0,
                "cache_hit": True,
            }

    tavily_client = client or get_tavily_client()
    extract_kwargs: dict = {
        "urls": [clean_url],
        "extract_depth": "basic",
        "format": "markdown",
        "include_usage": True,
    }

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
            for opt_key in ("include_usage", "format", "extract_depth"):
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
        raise RuntimeError(
            f"Tavily URL extract failed for {clean_url}: {_format_tavily_error(last_exc)}"
        ) from last_exc

    if not isinstance(response, dict):
        raise RuntimeError(f"Could not extract readable article text from {clean_url}.")

    extracted_results = response.get("results") or []
    raw_content = ""
    title_val: str | None = None
    if isinstance(extracted_results, list) and extracted_results:
        first = extracted_results[0]
        if isinstance(first, dict):
            raw_content = str(first.get("raw_content") or first.get("content") or "").strip()
            raw_title = str(first.get("title") or "").strip()
            if raw_title:
                title_val = raw_title

    if not raw_content:
        failed_results = response.get("failed_results") or []
        fail_reason = ""
        if isinstance(failed_results, list) and failed_results:
            f0 = failed_results[0]
            if isinstance(f0, dict):
                fail_reason = _sanitize_single_line(str(f0.get("error") or f0.get("reason") or ""))
            else:
                fail_reason = _sanitize_single_line(str(f0))
        detail = f" ({fail_reason})" if fail_reason else ""
        raise RuntimeError(
            f"Could not extract readable article text from this URL{detail}. "
            "The page may block crawlers, require login, or return empty text."
        )

    normalized = re.sub(r"\r\n?", "\n", raw_content)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    was_truncated = False
    if max_chars > 0 and len(normalized) > max_chars:
        was_truncated = True
        cut = normalized[:max_chars].rstrip()
        last_space = cut.rfind(" ")
        if last_space > int(max_chars * 0.7):
            cut = cut[:last_space].rstrip()
        normalized = cut + "..."

    credits_used = _extract_usage_credits(response, default_credits=0.2)

    if use_cache:
        _cache_set(
            cache_key,
            endpoint="extract_article",
            response={
                "url": clean_url,
                "title": title_val,
                "content": normalized,
                "truncated": was_truncated,
                "usage": {"credits": credits_used},
            },
            cache_path=cache_path,
        )

    return {
        "url": clean_url,
        "title": title_val,
        "content": normalized,
        "truncated": was_truncated,
        "credits_used": credits_used,
        "cache_hit": False,
    }


def _format_source_date_str(raw_value: object) -> str | None:
    """Normalize a source's published_date into 'YYYY-MM-DD' or clean string, or None."""
    cleaned = _normalize_optional_text(raw_value)
    if not cleaned:
        return None
    parsed = _parse_published_date(cleaned)
    if parsed is not None:
        return parsed.isoformat()
    return cleaned


def _extract_top_sources(
    search_results: list[dict],
    limit: int = 3,
    entity_domains: list[str] | None = None,
    primary_urls: set[str] | None = None,
    news_urls: set[str] | None = None,
    max_snippet_words: int = MAX_SNIPPET_WORDS,
) -> list[dict]:
    """
    Extract up to `limit` unique source dicts (title, url, source_type, published_date,
    and <=25-word evidence snippet) from Tavily results.
    """
    sources: list[dict] = []
    seen_urls: set[str] = set()

    for result in search_results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "")).strip()
        norm_key = url.rstrip("/").lower()
        if not url or norm_key in seen_urls:
            continue
        seen_urls.add(norm_key)
        title = str(result.get("title", "")).strip() or url
        is_prim = norm_key in (primary_urls or set())
        res_topic = "news" if norm_key in (news_urls or set()) else None
        src_type = classify_source_type(
            url,
            entity_domains=entity_domains,
            retrieved_from_primary=is_prim,
            result_topic=res_topic,
        )
        pub_date = _format_source_date_str(result.get("published_date"))
        raw_excerpt = result.get("content") or result.get("raw_content") or ""
        snippet = truncate_snippet_words(raw_excerpt, max_words=max_snippet_words)

        sources.append(
            {
                "title": title,
                "url": url,
                "source_type": src_type,
                "published_date": pub_date,
                "snippet": snippet,
            }
        )
        if len(sources) >= limit:
            break

    return sources


def _format_search_results(
    search_results: list[dict],
    entity_domains: list[str] | None = None,
    primary_urls: set[str] | None = None,
    news_urls: set[str] | None = None,
) -> str:
    """
    Format Tavily search results into readable text (with source type tags and dates)
    for the LLM freshness-judgement prompt.
    """
    if not search_results:
        return "No web search results were returned."

    formatted_blocks: list[str] = []
    for idx, result in enumerate(search_results, start=1):
        title = str(result.get("title", "Untitled")).strip()
        url = str(result.get("url", "")).strip()
        norm_key = url.rstrip("/").lower()
        src_type = classify_source_type(
            url,
            entity_domains=entity_domains,
            retrieved_from_primary=norm_key in (primary_urls or set()),
            result_topic="news" if norm_key in (news_urls or set()) else None,
        )
        published_date = str(result.get("published_date", "") or "").strip()
        content = str(result.get("content", "")).strip()

        # Keep each snippet concise so prompts remain fast and readable.
        if len(content) > 800:
            content = content[:800] + "..."

        date_line = f"Published: {published_date}\n" if published_date else ""
        formatted_blocks.append(
            f"[{idx}] [{src_type}] {title}\nURL: {url}\n{date_line}Snippet: {content}"
        )

    return "\n\n".join(formatted_blocks)


def _normalize_optional_text(value: object) -> str | None:
    """Normalize an optional structured verifier field, turning blanks/'null'/'none' into None."""
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "unknown"}:
        return None
    return text


def _normalize_judgement(parsed: object) -> dict:
    """
    Validate and normalize the LLM's freshness judgement JSON, enforcing status
    score bands and extracting structured change metadata.
    """
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object for freshness judgement, got {type(parsed).__name__}."
        )

    raw_status = str(parsed.get("status", "uncertain")).strip().lower()
    status = raw_status if raw_status in VALID_STATUSES else "uncertain"

    # Unverifiable claims carry no numeric score (rendered as N/A).
    if status == "unverifiable":
        score: int | None = None
    else:
        default_scores = {"fresh": 90, "uncertain": 55, "outdated": 15}
        raw_score = parsed.get("score", default_scores[status])
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = default_scores[status]
        band_min, band_max = STATUS_SCORE_BANDS[status]
        score = max(band_min, min(band_max, max(0, min(100, score))))

    # Parse optional agreeing_sources count; fall back to None if omitted/invalid.
    agreeing_sources: int | None = None
    raw_agreeing = parsed.get("agreeing_sources")
    if raw_agreeing is not None and raw_agreeing != "":
        try:
            agreeing_sources = max(0, int(raw_agreeing))
        except (TypeError, ValueError):
            agreeing_sources = None

    raw_conflicting = parsed.get("conflicting_sources")
    if isinstance(raw_conflicting, bool):
        conflicting_sources = raw_conflicting
    elif isinstance(raw_conflicting, str):
        conflicting_sources = raw_conflicting.strip().lower() in {"true", "yes", "1"}
    else:
        conflicting_sources = bool(raw_conflicting)

    current_info = str(parsed.get("current_info", "")).strip()
    if not current_info:
        current_info = (
            "No reliable evidence found in live sources to verify this claim."
            if status == "unverifiable"
            else "No current details could be extracted from the sources."
        )

    current_value = _normalize_optional_text(parsed.get("current_value"))
    effective_date = _normalize_optional_text(parsed.get("effective_date"))
    change_summary = _normalize_optional_text(parsed.get("change_summary"))
    stale_since = _normalize_optional_text(parsed.get("stale_since"))

    if status == "fresh":
        stale_since = None
        if change_summary and change_summary.lower() in {"no change", "no changes", "none", "unchanged"}:
            change_summary = None
    elif status == "unverifiable":
        stale_since = None

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No detailed reason was provided."

    return {
        "status": status,
        "score": score,
        "agreeing_sources": agreeing_sources,
        "conflicting_sources": conflicting_sources,
        "current_info": current_info,
        "current_value": current_value,
        "effective_date": effective_date,
        "change_summary": change_summary,
        "stale_since": stale_since,
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


def _extract_independent_domain(url: object) -> str:
    """
    Normalize a URL to its independent organization/registered domain so multiple
    pages or subdomains on the same site (e.g., docs.python.org and peps.python.org)
    do not falsely count as independent confirming sources.
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = (parsed.netloc or "").split(":")[0].lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""

    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_PUBLIC_SUFFIXES:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _parse_flexible_date(raw_value: object) -> date | None:
    """
    Parse a date string (ISO day, RFC 2822, 'YYYY-MM', or 'Month YYYY') into a date.
    Month-only strings resolve to the first day of that month for duration math.
    """
    parsed = _parse_published_date(raw_value)
    if parsed is not None:
        return parsed

    text = _normalize_optional_text(raw_value)
    if not text:
        return None

    for fmt in ("%Y-%m", "%Y/%m", "%B %Y", "%b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    match = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(20\d{2})\b", text, re.IGNORECASE)
    if match:
        snippet = f"{match.group(1)} {match.group(2)}"
        for fmt in ("%B %Y", "%b %Y"):
            try:
                return datetime.strptime(snippet, fmt).date()
            except ValueError:
                continue

    return None


def _extract_key_tokens(text: object) -> set[str]:
    """
    Extract contentful alphanumeric/version/numeric tokens from text, normalizing
    thousands separators (e.g., '$1,299' -> '1299') and ignoring stopwords.
    """
    normalized = re.sub(r"(?<=\d),(?=\d)", "", str(text or "").lower())
    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", normalized)
    return {
        tok
        for tok in raw_tokens
        if (len(tok) >= 3 or any(ch.isdigit() for ch in tok)) and tok not in STOPWORDS
    }


def _blob_contains_token(blob_normalized: str, tok: str) -> bool:
    """Return True when `tok` appears at alphanumeric boundaries (allows '3.14' in '3.14.7')."""
    if not tok:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", blob_normalized))


def estimate_stale_since_from_sources(
    status: str,
    claim_text: str,
    search_results: list[dict],
    current_info: str = "",
    current_value: str | None = None,
    change_summary: str | None = None,
    effective_date: str | None = None,
    llm_stale_since: str | None = None,
    agreeing_indices: list[int] | None = None,
    today: date | None = None,
) -> tuple[str | None, int | None, str | None]:
    """
    Estimate when an outdated fact changed using the earliest dated source about the change,
    and compute an approximate duration label such as 'stale for ~6 months'.

    Returns:
        Tuple (stale_since_label, stale_for_months, stale_duration_label).
    """
    if str(status or "").strip().lower() != "outdated":
        return None, None, None

    reference_date = today or date.today()
    dated_entries: list[tuple[int, date, dict]] = []
    for idx, result in enumerate(search_results or []):
        if not isinstance(result, dict):
            continue
        pub_date = _parse_published_date(result.get("published_date"))
        if pub_date is not None and pub_date <= reference_date:
            dated_entries.append((idx, pub_date, result))

    claim_toks = _extract_key_tokens(claim_text)
    change_blob = f"{current_info} {current_value or ''} {change_summary or ''}"
    change_toks = _extract_key_tokens(change_blob)
    generic_change_filler = {
        "new", "only", "still", "using", "used", "older", "longer",
        "no", "yes", "same", "remains", "became", "since",
    }
    new_change_toks = change_toks - claim_toks
    specific_new_toks = new_change_toks - generic_change_filler
    change_markers = {
        "new", "replaced", "removed", "updated", "launched", "released",
        "introduced", "effective", "revised", "changed", "successor",
        "reform", "abolished", "stepped", "resigned", "appointed",
    }

    specific_matches: list[tuple[int, date, dict]] = []
    new_tok_matches: list[tuple[int, date, dict]] = []
    fallback_matches: list[tuple[int, date, dict]] = []

    for idx, pub_date, result in dated_entries:
        raw_blob = f"{result.get('title', '')} {result.get('content', '')} {result.get('raw_content', '')}".lower()
        blob = re.sub(r"(?<=\d),(?=\d)", "", raw_blob)

        if specific_new_toks and any(_blob_contains_token(blob, tok) for tok in specific_new_toks):
            specific_matches.append((idx, pub_date, result))
            continue
        if new_change_toks and any(_blob_contains_token(blob, tok) for tok in new_change_toks):
            new_tok_matches.append((idx, pub_date, result))
            continue
        needed = 1 if len(change_toks) <= 2 else 2
        if (
            (change_toks and sum(1 for tok in change_toks if _blob_contains_token(blob, tok)) >= needed)
            or (agreeing_indices and idx in agreeing_indices)
            or any(_blob_contains_token(blob, m) for m in change_markers)
        ):
            fallback_matches.append((idx, pub_date, result))

    if specific_matches:
        earliest_source_date: date | None = min(d for _, d, _ in specific_matches)
    elif new_tok_matches:
        earliest_source_date = min(d for _, d, _ in new_tok_matches)
    elif fallback_matches:
        earliest_source_date = min(d for _, d, _ in fallback_matches)
    elif agreeing_indices and any(i in agreeing_indices for i, _, _ in dated_entries):
        earliest_source_date = min(d for i, d, _ in dated_entries if i in agreeing_indices)
    elif dated_entries:
        earliest_source_date = min(d for _, d, _ in dated_entries)
    else:
        earliest_source_date = None

    clean_llm_stale = _normalize_optional_text(llm_stale_since)
    clean_eff = _normalize_optional_text(effective_date)
    parsed_llm_stale = _parse_flexible_date(clean_llm_stale)
    parsed_eff = _parse_flexible_date(clean_eff)

    resolved_date: date | None = None
    resolved_label: str | None = None

    if earliest_source_date is not None and parsed_llm_stale is not None:
        same_month = (
            earliest_source_date.year == parsed_llm_stale.year
            and earliest_source_date.month == parsed_llm_stale.month
        )
        # Prefer the exact dated article when LLM gave a month-only string in that same month.
        if same_month and (reference_date - earliest_source_date).days > 7:
            resolved_date = earliest_source_date
            resolved_label = earliest_source_date.isoformat()
        elif earliest_source_date <= parsed_llm_stale:
            resolved_date = earliest_source_date
            resolved_label = earliest_source_date.isoformat()
        else:
            resolved_date = parsed_llm_stale
            resolved_label = clean_llm_stale
    elif earliest_source_date is not None and parsed_eff is not None:
        if earliest_source_date <= parsed_eff:
            resolved_date = earliest_source_date
            resolved_label = earliest_source_date.isoformat()
        else:
            resolved_date = parsed_eff
            resolved_label = clean_eff
    elif earliest_source_date is not None:
        resolved_date = earliest_source_date
        resolved_label = earliest_source_date.isoformat()
    elif parsed_llm_stale is not None:
        resolved_date = parsed_llm_stale
        resolved_label = clean_llm_stale
    elif parsed_eff is not None:
        resolved_date = parsed_eff
        resolved_label = clean_eff
    else:
        resolved_label = clean_llm_stale or clean_eff
        return resolved_label, None, None

    delta_days = max(0, (reference_date - resolved_date).days)
    approx_months = max(0, round(delta_days / 30.4375))
    if approx_months == 0:
        duration_label = "stale for <1 month"
    elif approx_months == 1:
        duration_label = "stale for ~1 month"
    else:
        duration_label = f"stale for ~{approx_months} months"

    return resolved_label, approx_months, duration_label


def _is_authoritative_source(
    url: object,
    entity_domains: list[str] | None = None,
) -> bool:
    """
    Return True when a URL belongs to an official/government/docs/reference source.
    """
    raw = str(url or "").strip()
    if not raw:
        return False
    score, tier = _score_single_domain_authority(raw, entity_domains=entity_domains)
    return tier == "official" or score >= 85


def _matching_result_indices(current_info: str, search_results: list[dict]) -> list[int]:
    """
    Return indices of search results whose title/content share key terms with `current_info`.
    """
    if not search_results or not current_info:
        return []

    key_tokens = _extract_key_tokens(current_info)
    if not key_tokens:
        return []

    matched_indices: list[int] = []
    needed = 1 if len(key_tokens) <= 2 else min(2, len(key_tokens))
    for idx, result in enumerate(search_results):
        if not isinstance(result, dict):
            continue
        raw_blob = f"{result.get('title', '')} {result.get('content', '')}".lower()
        blob = re.sub(r"(?<=\d),(?=\d)", "", raw_blob)
        overlap = sum(1 for tok in key_tokens if _blob_contains_token(blob, tok))
        if overlap >= needed:
            matched_indices.append(idx)

    return matched_indices


def _count_agreeing_sources(current_info: str, search_results: list[dict]) -> int:
    """
    Fallback heuristic: count how many search results share key numbers/terms
    with `current_info` when the LLM does not report `agreeing_sources`.
    """
    return len(_matching_result_indices(current_info, search_results))


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
        agreeing = max(0, min(total, int(llm_agreeing_count)))
    else:
        agreeing = _count_agreeing_sources(current_info, search_results)

    score = round((agreeing / total) * 100)
    detail = f"{agreeing}/{total} top results support the same current fact"
    return score, detail, agreeing


def _score_single_domain_authority(
    url: str,
    entity_domains: list[str] | None = None,
) -> tuple[int, str]:
    """
    Classify a single URL's domain authority into high/medium/low tiers.

    Returns:
        Tuple of (score 0-100, tier_label).
    """
    parsed = urlparse(url.strip())
    domain = (parsed.netloc or "").split(":")[0].lower().strip(".")
    if domain.startswith("www."):
        domain = domain[4:]
    path = (parsed.path or "").lower()

    if not domain:
        return 50, "unknown"

    # Low tier: forums, social networks, Q&A (allowing official help/about/investor subdomains).
    is_official_social_sub = domain.startswith(
        ("help.", "about.", "investor.", "business.", "developers.")
    )
    if (
        any(_host_matches_pattern(domain, d) for d in FORUM_OR_SOCIAL_DOMAINS)
        and not is_official_social_sub
    ):
        return 25, "forum"
    if domain.startswith(("forum.", "community.", "discuss.")) or any(
        seg in path for seg in ("/forum/", "/forums/", "/discussions/", "/threads/")
    ):
        return 25, "forum"

    if any(_host_matches_pattern(domain, d) for d in BLOG_DOMAINS):
        return 40, "blog"

    matches_entity = any(_host_matches_pattern(domain, d) for d in (entity_domains or []))

    # High tier: government, academic, entity official domains, docs/support/IR subdomains.
    if domain.endswith((".gov", ".edu", ".ac.uk", ".nic.in", ".int")) or ".gov." in domain:
        return 95, "official"
    if domain.startswith(
        ("docs.", "developer.", "developers.", "api.", "support.", "help.", "learn.", "ir.", "peps.")
    ):
        return 95, "official"
    if matches_entity:
        return 95, "official"
    if any(_host_matches_pattern(domain, d) for d in OFFICIAL_AUTHORITY_DOMAINS):
        return 90, "official"

    if domain.startswith("blog.") or domain.endswith(".github.io") or "/blog/" in path or "/@" in path:
        return 45, "blog"

    if any(seg in path for seg in ("/docs", "/documentation", "/api/", "/pricing", "/legal")):
        return 85, "official"

    if any(_host_matches_pattern(domain, d) for d in NEWS_DOMAINS):
        return 80, "news"

    # Standard primary company/news/reference websites.
    return 70, "general"


def _score_source_authority(
    search_results: list[dict],
    entity_domains: list[str] | None = None,
) -> tuple[int, str]:
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
        dom_score, tier = _score_single_domain_authority(url, entity_domains=entity_domains)
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


def get_claim_risk_weight(claim_item: dict) -> float:
    """
    Return the aggregation weight for a verified claim based on its risk level.
    Regulation/price claims ('critical') count more heavily than version/other claims.
    """
    risk_level = str(claim_item.get("risk_level", "")).strip().lower()
    if risk_level not in RISK_SCORE_WEIGHTS:
        risk_level = get_risk_level(str(claim_item.get("category", "other")))
    return RISK_SCORE_WEIGHTS.get(risk_level, 1.0)


def compute_risk_weighted_overall_score(verification_results: list[dict]) -> int | None:
    """
    Compute the 0-100 risk-weighted overall freshness score across scorable claims.

    Claims with status 'unverifiable' or score=None are excluded so missing evidence
    does not artificially skew the percentage. Returns None if no scorable claims remain.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for item in verification_results:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        score = item.get("score")
        if status == "unverifiable" or score is None:
            continue
        try:
            numeric_score = max(0.0, min(100.0, float(score)))
        except (TypeError, ValueError):
            continue
        weight = get_claim_risk_weight(item)
        weighted_sum += numeric_score * weight
        total_weight += weight

    if total_weight <= 0:
        return None
    return max(0, min(100, round(weighted_sum / total_weight)))


def get_confidence_level(
    status: str,
    total_sources: int,
    agreement_score: int,
    authority_score: int,
    has_error: bool = False,
    independent_sources: int | None = None,
    independent_agreeing_sources: int | None = None,
    has_authoritative_source: bool | None = None,
    has_conflict: bool = False,
) -> str:
    """
    Calibrate confidence ('high', 'medium', 'low') from independent agreeing evidence:
    - 'low': error, unverifiable, <= 1 independent source, or conflicting sources.
    - 'high': ONLY when status is decisive ('fresh' or 'outdated'), >= 2 independent
      sources agree without conflict, and at least one agreeing source is authoritative.
    - 'medium': partial or non-authoritative multi-source support (never defaults to 'high').
    """
    normalized_status = str(status or "uncertain").strip().lower()
    if has_error or normalized_status == "unverifiable" or total_sources <= 0:
        return "low"

    eff_total_ind = total_sources if independent_sources is None else independent_sources
    if eff_total_ind <= 1 or has_conflict:
        return "low"

    # When multiple sources exist and half or fewer agree, treat sources as conflicting.
    if total_sources >= 2 and agreement_score <= 50:
        return "low"

    eff_agreeing_ind = eff_total_ind if independent_agreeing_sources is None else independent_agreeing_sources
    if eff_agreeing_ind <= 1:
        return "low"

    eff_has_auth = (authority_score >= 85) if has_authoritative_source is None else bool(has_authoritative_source)

    if (
        normalized_status in ("fresh", "outdated")
        and eff_agreeing_ind >= 2
        and eff_has_auth
        and agreement_score >= 60
    ):
        return "high"

    return "medium"


def _detect_source_disagreement(
    normalized_status: str,
    current_info: str,
    reason: str,
    search_results: list[dict],
    agreeing_indices: list[int],
    agreeing_cnt: int,
    independent_sources_cnt: int,
    llm_agreeing_count: int | None,
    llm_conflicting: bool,
) -> bool:
    """
    Detect explicit source conflict either from LLM signals/text or when independent
    contemporaneous sources actively split (<= 50% agreeing on the current fact).
    """
    if normalized_status == "unverifiable":
        return False
    if llm_conflicting:
        return True

    text_blob = f"{current_info} {reason}".lower()
    conflict_phrases = (
        "sources disagree",
        "conflict",
        "contradict",
        "inconsistent",
        "mixed signals",
        "disagree",
    )
    if any(phrase in text_blob for phrase in conflict_phrases):
        return True

    total = len(search_results)
    if (
        llm_agreeing_count is not None
        and total >= 2
        and independent_sources_cnt >= 2
        and 1 <= agreeing_cnt
        and agreeing_cnt / total <= 0.5
    ):
        agreeing_domains = {
            _extract_independent_domain(search_results[i].get("url"))
            for i in agreeing_indices
            if 0 <= i < total and isinstance(search_results[i], dict)
        }
        agreeing_dates = [
            _parse_published_date(search_results[i].get("published_date"))
            for i in agreeing_indices
            if 0 <= i < total and isinstance(search_results[i], dict)
        ]
        valid_agreeing_dates = [d for d in agreeing_dates if d is not None]
        newest_agreeing = max(valid_agreeing_dates) if valid_agreeing_dates else None

        # Only treat non-agreeing domains as an active disagreement when at least one
        # dissenting independent domain is contemporaneous (not strictly >180d older history).
        for idx, res in enumerate(search_results):
            if idx in agreeing_indices or not isinstance(res, dict):
                continue
            dom = _extract_independent_domain(res.get("url"))
            if not dom or dom in agreeing_domains:
                continue
            res_date = _parse_published_date(res.get("published_date"))
            if res_date is None or newest_agreeing is None:
                return True
            if (newest_agreeing - res_date).days <= 180:
                return True

    return False


def compute_freshness_score(
    status: str,
    llm_score: int | None,
    current_info: str,
    search_results: list[dict],
    llm_agreeing_count: int | None = None,
    today: date | None = None,
    reason: str = "",
    llm_conflicting: bool = False,
    entity_domains: list[str] | None = None,
) -> dict:
    """
    Compute the status-banded freshness score (0-100 or None for unverifiable),
    the 'evidence_strength' breakdown dict, explanation, conflict state,
    agreement counts, and calibrated confidence.

    Evidence-strength signals inform the breakdown and fine position within the
    claim's status band, but NEVER push a score outside its status band:
    - fresh: 75-100
    - uncertain: 40-74
    - outdated: 0-39
    - unverifiable: None (N/A)
    """
    normalized_status = str(status or "uncertain").strip().lower()
    if normalized_status not in VALID_STATUSES:
        normalized_status = "uncertain"

    total = len(search_results)
    if search_results:
        recency_score, recency_detail = _score_source_recency(search_results, today=today)
        agreement_score, agreement_detail, agreeing_cnt = _score_source_agreement(
            search_results, current_info, llm_agreeing_count=llm_agreeing_count
        )
        authority_score, authority_detail = _score_source_authority(
            search_results, entity_domains=entity_domains
        )
        evidence_strength = round(
            0.40 * recency_score + 0.40 * agreement_score + 0.20 * authority_score
        )
    else:
        recency_score, recency_detail = 50, "No source URLs available (neutral 50/100)"
        agreement_score, agreement_detail, agreeing_cnt = 0, "0/0 sources available", 0
        authority_score, authority_detail = 50, "No source URLs available (neutral 50/100)"
        evidence_strength = 0

    # Deduplicate independent domains across all sources and across agreeing sources.
    all_independent_domains = {
        _extract_independent_domain(r.get("url"))
        for r in search_results
        if isinstance(r, dict) and _extract_independent_domain(r.get("url"))
    }
    independent_sources_cnt = len(all_independent_domains)

    if agreeing_cnt <= 0:
        agreeing_indices: list[int] = []
    elif llm_agreeing_count is None:
        agreeing_indices = _matching_result_indices(current_info, search_results)[:agreeing_cnt]
    else:
        matched_set = _matching_result_indices(current_info, search_results)
        ordered_indices = matched_set + [i for i in range(total) if i not in matched_set]
        agreeing_indices = ordered_indices[:agreeing_cnt]

    agreeing_results = [
        search_results[i]
        for i in agreeing_indices
        if 0 <= i < total and isinstance(search_results[i], dict)
    ]
    independent_agreeing_domains = {
        _extract_independent_domain(r.get("url"))
        for r in agreeing_results
        if _extract_independent_domain(r.get("url"))
    }
    independent_agreeing_cnt = len(independent_agreeing_domains)
    has_auth_agreeing = any(
        _is_authoritative_source(r.get("url"), entity_domains=entity_domains)
        for r in agreeing_results
    )

    agreement_summary = f"{agreeing_cnt} of {total} sources agree"
    conflict_detected = _detect_source_disagreement(
        normalized_status=normalized_status,
        current_info=current_info,
        reason=reason,
        search_results=search_results,
        agreeing_indices=agreeing_indices,
        agreeing_cnt=agreeing_cnt,
        independent_sources_cnt=independent_sources_cnt,
        llm_agreeing_count=llm_agreeing_count,
        llm_conflicting=llm_conflicting,
    )

    effective_status = normalized_status
    conflict_note: str | None = None
    if conflict_detected:
        if effective_status != "unverifiable":
            effective_status = "uncertain"
        conflict_note = (
            f"Sources disagree on the current fact ({agreement_summary}). "
            "Marked Uncertain pending clearer evidence."
        )

    has_confidence_conflict = conflict_detected or (total >= 2 and agreement_score <= 50)

    if effective_status == "unverifiable":
        clamped_llm = 0
        weighted_llm = 0.0
        weighted_recency = round(recency_score * SCORE_WEIGHTS["source_recency"], 1) if search_results else 0.0
        weighted_agreement = round(agreement_score * SCORE_WEIGHTS["source_agreement"], 1) if search_results else 0.0
        weighted_authority = round(authority_score * SCORE_WEIGHTS["source_authority"], 1) if search_results else 0.0
        raw_weighted_score: int | None = None
        final_score: int | None = None
        status_band = "N/A"
        llm_detail = "Insufficient or missing live evidence (unverifiable)"
    else:
        default_scores = {"fresh": 90, "uncertain": 55, "outdated": 15}
        raw_llm = default_scores[effective_status] if llm_score is None else int(round(float(llm_score)))
        clamped_llm = max(0, min(100, raw_llm))

        weighted_llm = round(clamped_llm * SCORE_WEIGHTS["llm_verdict"], 1)
        weighted_recency = round(recency_score * SCORE_WEIGHTS["source_recency"], 1)
        weighted_agreement = round(agreement_score * SCORE_WEIGHTS["source_agreement"], 1)
        weighted_authority = round(authority_score * SCORE_WEIGHTS["source_authority"], 1)

        raw_weighted_score = max(
            0,
            min(100, round(weighted_llm + weighted_recency + weighted_agreement + weighted_authority)),
        )
        band_min, band_max = STATUS_SCORE_BANDS[effective_status]
        status_band = f"{band_min}-{band_max}"

        if effective_status == "outdated":
            # Outdated claims must remain within 0-39 regardless of how strong source metrics are.
            llm_in_band = max(band_min, min(band_max, clamped_llm))
            evidence_gap_component = ((100 - evidence_strength) / 100.0) * band_max
            final_score = round(llm_in_band * 0.75 + evidence_gap_component * 0.25)
            final_score = max(band_min, min(band_max, final_score))
        else:
            final_score = max(band_min, min(band_max, raw_weighted_score))

        if conflict_detected and normalized_status != effective_status:
            llm_detail = (
                f"LLM judged '{normalized_status}' ({clamped_llm}/100) "
                "-> marked 'uncertain' (sources disagree)"
            )
        else:
            llm_detail = f"LLM judged '{effective_status}' ({clamped_llm}/100)"

    score_breakdown = {
        "llm_verdict": {
            "score": clamped_llm,
            "weight": SCORE_WEIGHTS["llm_verdict"],
            "weighted_points": weighted_llm,
            "detail": llm_detail,
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
        "evidence_strength": evidence_strength,
        "raw_weighted_score": raw_weighted_score,
        "status_band": status_band,
        "final_score": final_score,
    }

    if final_score is None:
        why_this_score = (
            f"Score N/A (unverifiable): {agreeing_cnt}/{total} sources checked "
            f"(evidence strength {evidence_strength}/100)."
        )
    elif conflict_detected:
        why_this_score = (
            f"Score {final_score}/100 ({effective_status} band {status_band}): "
            f"Sources disagree ({agreement_summary}, {independent_sources_cnt} independent domain(s)); "
            f"evidence strength {evidence_strength}/100 (recency {recency_score}/100, authority {authority_score}/100)."
        )
    else:
        why_this_score = (
            f"Score {final_score}/100 ({effective_status} band {status_band}): "
            f"LLM judged '{effective_status}' ({clamped_llm}/100), "
            f"{agreeing_cnt}/{total} sources ({independent_agreeing_cnt} independent domain(s)) agree, "
            f"evidence strength {evidence_strength}/100 (recency {recency_score}/100, authority {authority_score}/100)."
        )

    confidence = get_confidence_level(
        status=effective_status,
        total_sources=total,
        agreement_score=agreement_score,
        authority_score=authority_score,
        independent_sources=independent_sources_cnt,
        independent_agreeing_sources=independent_agreeing_cnt,
        has_authoritative_source=has_auth_agreeing,
        has_conflict=has_confidence_conflict,
    )

    return {
        "score": final_score,
        "resolved_status": effective_status,
        "score_breakdown": score_breakdown,
        "why_this_score": why_this_score,
        "confidence": confidence,
        "sources_agreeing": agreeing_cnt,
        "sources_total": total,
        "agreement_summary": agreement_summary,
        "agreeing_indices": agreeing_indices,
        "conflict_detected": conflict_detected,
        "conflict_note": conflict_note,
    }


def _build_fallback_breakdown(
    llm_score: int | None,
    recency_score: int,
    recency_detail: str,
    agreement_score: int,
    authority_score: int,
    authority_detail: str,
    error_note: str,
    status: str = "uncertain",
) -> tuple[int | None, dict]:
    """
    Build a status-consistent score_breakdown when Tavily returns no evidence or LLM evaluation fails.
    """
    normalized_status = str(status or "uncertain").strip().lower()
    evidence_strength = (
        round(0.40 * recency_score + 0.40 * agreement_score + 0.20 * authority_score)
        if normalized_status != "unverifiable"
        else 0
    )

    if normalized_status == "unverifiable" or llm_score is None:
        clamped_llm = 0
        weighted_llm = 0.0
        weighted_recency = 0.0
        weighted_agreement = 0.0
        weighted_authority = 0.0
        final_score: int | None = None
        raw_weighted: int | None = None
        status_band = "N/A"
    else:
        clamped_llm = max(0, min(100, int(llm_score)))
        weighted_llm = round(clamped_llm * SCORE_WEIGHTS["llm_verdict"], 1)
        weighted_recency = round(recency_score * SCORE_WEIGHTS["source_recency"], 1)
        weighted_agreement = round(agreement_score * SCORE_WEIGHTS["source_agreement"], 1)
        weighted_authority = round(authority_score * SCORE_WEIGHTS["source_authority"], 1)
        raw_weighted = max(0, min(100, round(weighted_llm + weighted_recency + weighted_agreement + weighted_authority)))
        band_min, band_max = STATUS_SCORE_BANDS.get(normalized_status, (40, 74))
        final_score = max(band_min, min(band_max, raw_weighted))
        status_band = f"{band_min}-{band_max}"

    breakdown = {
        "llm_verdict": {
            "score": clamped_llm,
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
        "evidence_strength": evidence_strength,
        "raw_weighted_score": raw_weighted,
        "status_band": status_band,
        "final_score": final_score,
    }
    return final_score, breakdown


def _run_judge_llm_pass(
    user_prompt: str,
    operation: str,
    model_override: str | None = None,
    simulate_outage: bool | None = None,
) -> tuple[dict | None, str | None, dict | None, str | None]:
    """
    Execute one LLM verification pass for a claim with task routing (`task="judge"`).

    Returns:
        `(judgement_dict_or_none, model_used, gateway_metrics, friendly_error_or_none)`
    """
    model_used: str | None = None
    gateway_metrics: dict | None = None
    pop_last_gateway_metrics()
    llm_t0 = time.perf_counter()
    try:
        raw_response, model_used = _invoke_call_llm(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            simulate_outage=simulate_outage,
            task="judge",
            model_override=model_override,
        )
        gateway_metrics = _capture_call_metrics(
            operation,
            model_used,
            (time.perf_counter() - llm_t0) * 1000.0,
            task="judge",
        )
        parsed = parse_json_response(raw_response)
        judgement = _normalize_judgement(parsed)
        return judgement, model_used, gateway_metrics, None
    except Exception as exc:
        if gateway_metrics is None and (model_used or model_override):
            gateway_metrics = _capture_call_metrics(
                operation,
                model_used or model_override,
                (time.perf_counter() - llm_t0) * 1000.0,
                task="judge",
            )
        err_lower = str(exc).lower()
        if isinstance(exc, ValueError) and "json" in err_lower:
            friendly_err = "The LLM returned invalid non-JSON formatting while judging this claim."
        else:
            friendly_err = format_llm_error(exc)
        return None, (model_used or model_override), gateway_metrics, friendly_err


def verify_claim(
    claim_item: dict[str, str],
    tavily_client: TavilyClient | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
    entity_domains: dict[str, list[str] | str] | None = None,
    dual_judge: bool = False,
    secondary_judge_model: str | None = None,
) -> dict:
    """
    Verify a single extracted claim against live Tavily search + extract results.

    Returns 'unverifiable' (score=None) when no reliable evidence is retrieved,
    or a status-banded score ('fresh' 75-100, 'uncertain' 40-74, 'outdated' 0-39)
    with primary-first evidence metadata, source labels, <=25-word snippets,
    agreement counts, explicit conflict detection, and stale duration estimates.

    When `dual_judge=True`, two different models judge the claim independently over
    the same Tavily evidence: if they disagree, the claim is marked 'uncertain'
    with 'Models disagree' (`40-74`, `low` confidence); if they agree, it carries
    '2/2 models agree'.

    Args:
        claim_item: Dict containing 'claim', 'category', and 'search_query'.
        tavily_client: Optional shared TavilyClient instance.
        use_cache: Whether to read/write the 6-hour on-disk Tavily cache.
        cache_path: Optional override path for the SQLite cache file.
        entity_domains: Optional override mapping entity keys to official domains.
        dual_judge: If True, run two distinct judge models on the same evidence.
        secondary_judge_model: Optional explicit 2nd judge model override.

    Returns:
        Dict with claim, span, category, search_query, status, score, score_breakdown,
        why_this_score, confidence, risk_level, current_info, current_value,
        effective_date, change_summary, stale_since, stale_for_months,
        stale_duration_label, reason, sources (top 3 with source_type,
        published_date, snippet), entity_domains, sources_agreeing,
        sources_total, agreement_summary, conflict_detected, conflict_note,
        dual_judge_used, judge_models, judge_verdicts, model_disagreement,
        model_agreement_summary, judge_telemetry, tavily_credits_used, cache_hit,
        search_strategy, extracted_url, model, and optional error message.
    """
    claim_text = claim_item.get("claim", "").strip()
    span_text = str(claim_item.get("span") or "").strip() or None
    category = claim_item.get("category", "other").strip()
    search_query = claim_item.get("search_query", "").strip() or claim_text
    risk_level = get_risk_level(category)
    matched_entity_domains = resolve_entity_official_domains(
        f"{claim_text} {search_query}",
        entity_domains=entity_domains,
    )

    # Step 1: Run primary-first / category-appropriate Tavily search (cached 6h on disk).
    try:
        search_meta = search_tavily_for_claim(
            query=search_query,
            category=category,
            risk_level=risk_level,
            max_results=5,
            client=tavily_client,
            use_cache=use_cache,
            cache_path=cache_path,
            claim=claim_text,
            entity_domains=entity_domains,
        )
    except Exception as exc:
        friendly_tavily_err = _format_tavily_error(exc)
        score, breakdown = _build_fallback_breakdown(
            llm_score=None,
            recency_score=50,
            recency_detail="Unavailable due to Tavily search failure",
            agreement_score=0,
            authority_score=50,
            authority_detail="Unavailable due to Tavily search failure",
            error_note="Tavily search failed (unverifiable)",
            status="unverifiable",
        )
        return {
            "claim": claim_text,
            "span": span_text,
            "category": category,
            "search_query": search_query,
            "status": "unverifiable",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": "Score N/A (unverifiable): Live web search failed, so no evidence could be retrieved.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "Unable to retrieve live web sources to verify this claim.",
            "current_value": None,
            "effective_date": None,
            "change_summary": None,
            "stale_since": None,
            "stale_for_months": None,
            "stale_duration_label": None,
            "reason": friendly_tavily_err,
            "sources": [],
            "entity_domains": matched_entity_domains,
            "sources_agreeing": 0,
            "sources_total": 0,
            "agreement_summary": "0 of 0 sources agree",
            "conflict_detected": False,
            "conflict_note": None,
            "dual_judge_used": bool(dual_judge),
            "judge_models": [],
            "judge_verdicts": [],
            "model_disagreement": False,
            "model_agreement_summary": None,
            "judge_telemetry": [],
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
    resolved_entity_domains = search_meta.get("entity_domains") or matched_entity_domains
    primary_urls = search_meta.get("primary_urls") or set()
    news_urls = search_meta.get("news_urls") or set()
    top_sources = _extract_top_sources(
        search_results,
        limit=3,
        entity_domains=resolved_entity_domains,
        primary_urls=primary_urls,
        news_urls=news_urls,
    )

    if not search_results:
        score, breakdown = _build_fallback_breakdown(
            llm_score=None,
            recency_score=50,
            recency_detail="No search results returned",
            agreement_score=0,
            authority_score=50,
            authority_detail="No search results returned",
            error_note="0 results returned (unverifiable)",
            status="unverifiable",
        )
        return {
            "claim": claim_text,
            "span": span_text,
            "category": category,
            "search_query": search_query,
            "status": "unverifiable",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": "Score N/A (unverifiable): Tavily returned 0 results to confirm or refute this claim.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "No relevant live search results were found.",
            "current_value": None,
            "effective_date": None,
            "change_summary": None,
            "stale_since": None,
            "stale_for_months": None,
            "stale_duration_label": None,
            "reason": "Tavily returned 0 results for this search query.",
            "sources": [],
            "entity_domains": resolved_entity_domains,
            "sources_agreeing": 0,
            "sources_total": 0,
            "agreement_summary": "0 of 0 sources agree",
            "conflict_detected": False,
            "conflict_note": None,
            "dual_judge_used": bool(dual_judge),
            "judge_models": [],
            "judge_verdicts": [],
            "model_disagreement": False,
            "model_agreement_summary": None,
            "judge_telemetry": [],
            "tavily_credits_used": round(total_credits, 2),
            "cache_hit": search_cache_hit,
            "search_strategy": search_strategy,
            "extracted_url": None,
        }

    # Step 2: Run Tavily Extract on the single best source URL to confirm exact figures.
    best_url = select_best_source_url(
        search_results,
        entity_domains=resolved_entity_domains,
        primary_urls=primary_urls,
    )
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

    has_usable_snippet = any(
        str(r.get("content", "")).strip() or str(r.get("raw_content", "")).strip()
        for r in search_results
        if isinstance(r, dict)
    ) or bool(str(extract_info.get("content", "")).strip())

    if not has_usable_snippet:
        rec_score, rec_detail = _score_source_recency(search_results, today=date.today())
        auth_score, auth_detail = _score_source_authority(
            search_results, entity_domains=resolved_entity_domains
        )
        score, breakdown = _build_fallback_breakdown(
            llm_score=None,
            recency_score=rec_score,
            recency_detail=rec_detail,
            agreement_score=0,
            authority_score=auth_score,
            authority_detail=auth_detail,
            error_note="Sources had no readable excerpt (unverifiable)",
            status="unverifiable",
        )
        total_cnt = len(search_results)
        return {
            "claim": claim_text,
            "span": span_text,
            "category": category,
            "search_query": search_query,
            "status": "unverifiable",
            "score": score,
            "score_breakdown": breakdown,
            "why_this_score": "Score N/A (unverifiable): Live sources returned empty snippets with no verifiable facts.",
            "confidence": "low",
            "risk_level": risk_level,
            "current_info": "Search results contained no readable evidence to verify this claim.",
            "current_value": None,
            "effective_date": None,
            "change_summary": None,
            "stale_since": None,
            "stale_for_months": None,
            "stale_duration_label": None,
            "reason": "Returned search results contained empty content snippets.",
            "sources": top_sources,
            "entity_domains": resolved_entity_domains,
            "sources_agreeing": 0,
            "sources_total": total_cnt,
            "agreement_summary": f"0 of {total_cnt} sources agree",
            "conflict_detected": False,
            "conflict_note": None,
            "dual_judge_used": bool(dual_judge),
            "judge_models": [],
            "judge_verdicts": [],
            "model_disagreement": False,
            "model_agreement_summary": None,
            "judge_telemetry": [],
            "tavily_credits_used": round(total_credits, 2),
            "cache_hit": overall_cache_hit,
            "search_strategy": search_strategy,
            "extracted_url": extract_info["url"] if extract_info.get("content") else best_url,
            "search_warning": search_meta.get("partial_error"),
        }

    # Step 3: Ask Judge model(s) to judge freshness using today's date + labeled sources + extract.
    today_date = date.today()
    today_str = today_date.isoformat()
    formatted_sources = _format_search_results(
        search_results,
        entity_domains=resolved_entity_domains,
        primary_urls=primary_urls,
        news_urls=news_urls,
    )
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

    judge_telemetry: list[dict] = []
    judge_verdicts: list[dict] = []
    judge_models: list[str] = []
    model_disagreement = False
    model_agreement_summary: str | None = None
    partial_judge_error: str | None = None

    if not dual_judge:
        judgement, model_used, gateway_metrics, friendly_err = _run_judge_llm_pass(
            user_prompt=user_prompt,
            operation="Verify claim",
        )
        if gateway_metrics is not None:
            judge_telemetry.append(gateway_metrics)

        if judgement is None:
            rec_score, rec_detail = _score_source_recency(search_results, today=today_date)
            auth_score, auth_detail = _score_source_authority(
                search_results, entity_domains=resolved_entity_domains
            )
            score, breakdown = _build_fallback_breakdown(
                llm_score=50,
                recency_score=rec_score,
                recency_detail=rec_detail,
                agreement_score=50,
                authority_score=auth_score,
                authority_detail=auth_detail,
                error_note="LLM evaluation failed (neutral 50/100)",
                status="uncertain",
            )
            total_cnt = len(search_results)
            return {
                "claim": claim_text,
                "span": span_text,
                "category": category,
                "search_query": search_query,
                "status": "uncertain",
                "score": score,
                "score_breakdown": breakdown,
                "why_this_score": f"Score {score}/100 (uncertain band 40-74): LLM verdict failed, so verdict and agreement defaulted to 50/100 alongside source signals.",
                "confidence": "low",
                "risk_level": risk_level,
                "current_info": "Could not judge freshness automatically due to an LLM error.",
                "current_value": None,
                "effective_date": None,
                "change_summary": None,
                "stale_since": None,
                "stale_for_months": None,
                "stale_duration_label": None,
                "reason": friendly_err or "LLM verification failed.",
                "sources": top_sources,
                "entity_domains": resolved_entity_domains,
                "sources_agreeing": 0,
                "sources_total": total_cnt,
                "agreement_summary": f"0 of {total_cnt} sources agree",
                "conflict_detected": False,
                "conflict_note": None,
                "dual_judge_used": False,
                "judge_models": [model_used] if model_used else [],
                "judge_verdicts": [],
                "model_disagreement": False,
                "model_agreement_summary": None,
                "judge_telemetry": judge_telemetry,
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

        if model_used:
            judge_models = [model_used]
            judge_verdicts = [
                {
                    "model": model_used,
                    "status": judgement["status"],
                    "score": judgement["score"],
                    "current_info": judgement["current_info"],
                    "reason": judgement["reason"],
                }
            ]
    else:
        # Dual-judge mode: two distinct judge models evaluate the same Tavily evidence.
        primary_judge_cfg, default_second_cfg = get_dual_judge_models()
        routing_summary = get_task_routing_summary()
        explicit_second = (secondary_judge_model or "").strip()
        resolved_second = explicit_second or default_second_cfg

        j1_judgement, j1_model, j1_metrics, j1_err = _run_judge_llm_pass(
            user_prompt=user_prompt,
            operation="Verify claim (Judge 1)",
        )
        # If simulated outage pushed Judge 1 onto default_second_cfg, pick another fallback if available.
        if not explicit_second and j1_model and _same_model_name(resolved_second, j1_model):
            for alt_fb in routing_summary.get("fallbacks", []):
                if (
                    alt_fb
                    and not _same_model_name(alt_fb, j1_model)
                    and not _same_model_name(alt_fb, primary_judge_cfg)
                ):
                    resolved_second = alt_fb
                    break

        j2_judgement, j2_model, j2_metrics, j2_err = _run_judge_llm_pass(
            user_prompt=user_prompt,
            operation="Verify claim (Judge 2)",
            model_override=resolved_second,
            simulate_outage=False,
        )

        judge_telemetry = [m for m in (j1_metrics, j2_metrics) if isinstance(m, dict)]
        gateway_metrics = j1_metrics or j2_metrics
        model_used = j1_model or j2_model

        j1_label = j1_model or primary_judge_cfg
        j2_label = j2_model or resolved_second
        for m_lbl in (j1_label, j2_label):
            if m_lbl and m_lbl not in judge_models:
                judge_models.append(m_lbl)

        if j1_judgement is None and j2_judgement is None:
            combined_err = " | ".join(
                e for e in (j1_err, j2_err) if e
            ) or "Both judge models failed to verify this claim."
            rec_score, rec_detail = _score_source_recency(search_results, today=today_date)
            auth_score, auth_detail = _score_source_authority(
                search_results, entity_domains=resolved_entity_domains
            )
            score, breakdown = _build_fallback_breakdown(
                llm_score=50,
                recency_score=rec_score,
                recency_detail=rec_detail,
                agreement_score=50,
                authority_score=auth_score,
                authority_detail=auth_detail,
                error_note="Dual-judge LLM evaluation failed (neutral 50/100)",
                status="uncertain",
            )
            total_cnt = len(search_results)
            return {
                "claim": claim_text,
                "span": span_text,
                "category": category,
                "search_query": search_query,
                "status": "uncertain",
                "score": score,
                "score_breakdown": breakdown,
                "why_this_score": f"Score {score}/100 (uncertain band 40-74): Both judge models failed, so verdict and agreement defaulted to 50/100 alongside source signals.",
                "confidence": "low",
                "risk_level": risk_level,
                "current_info": "Could not judge freshness automatically because both judge models failed.",
                "current_value": None,
                "effective_date": None,
                "change_summary": None,
                "stale_since": None,
                "stale_for_months": None,
                "stale_duration_label": None,
                "reason": combined_err,
                "sources": top_sources,
                "entity_domains": resolved_entity_domains,
                "sources_agreeing": 0,
                "sources_total": total_cnt,
                "agreement_summary": f"0 of {total_cnt} sources agree",
                "conflict_detected": False,
                "conflict_note": None,
                "dual_judge_used": True,
                "judge_models": judge_models,
                "judge_verdicts": [],
                "model_disagreement": False,
                "model_agreement_summary": None,
                "judge_telemetry": judge_telemetry,
                "tavily_credits_used": round(total_credits, 2),
                "cache_hit": overall_cache_hit,
                "search_strategy": search_strategy,
                "extracted_url": extract_info["url"] if extract_info.get("content") else best_url,
                "gateway_metrics": gateway_metrics,
                "fallback_used": any(bool(m.get("fallback_used")) for m in judge_telemetry),
                "primary_model": gateway_metrics.get("primary_model") if gateway_metrics else None,
                "model": model_used,
                "error": combined_err,
            }

        if j1_judgement is not None and j2_judgement is not None:
            judge_verdicts = [
                {
                    "model": j1_label,
                    "status": j1_judgement["status"],
                    "score": j1_judgement["score"],
                    "current_info": j1_judgement["current_info"],
                    "reason": j1_judgement["reason"],
                },
                {
                    "model": j2_label,
                    "status": j2_judgement["status"],
                    "score": j2_judgement["score"],
                    "current_info": j2_judgement["current_info"],
                    "reason": j2_judgement["reason"],
                },
            ]
            s1 = j1_judgement["status"]
            s2 = j2_judgement["status"]
            agg_vals = [
                v for v in (j1_judgement["agreeing_sources"], j2_judgement["agreeing_sources"]) if v is not None
            ]
            merged_agreeing = round(sum(agg_vals) / len(agg_vals)) if agg_vals else None
            merged_conflicting = bool(
                j1_judgement["conflicting_sources"] or j2_judgement["conflicting_sources"]
            )
            default_band_scores = {"fresh": 90, "uncertain": 55, "outdated": 15, "unverifiable": 50}
            sc1 = j1_judgement["score"] if j1_judgement["score"] is not None else default_band_scores[s1]
            sc2 = j2_judgement["score"] if j2_judgement["score"] is not None else default_band_scores[s2]

            if s1 == s2:
                model_disagreement = False
                model_agreement_summary = "2/2 models agree"
                merged_score = None if s1 == "unverifiable" else round((sc1 + sc2) / 2)
                merged_reason = j1_judgement["reason"]
                if (
                    j2_judgement["reason"]
                    and j2_judgement["reason"].strip().lower() != j1_judgement["reason"].strip().lower()
                ):
                    merged_reason = f"{j1_judgement['reason']} (2/2 judges agreed: {j1_label}, {j2_label}.)"
                judgement = {
                    "status": s1,
                    "score": merged_score,
                    "agreeing_sources": merged_agreeing,
                    "conflicting_sources": merged_conflicting,
                    "current_info": j1_judgement["current_info"] or j2_judgement["current_info"],
                    "current_value": j1_judgement["current_value"] or j2_judgement["current_value"],
                    "effective_date": j1_judgement["effective_date"] or j2_judgement["effective_date"],
                    "change_summary": j1_judgement["change_summary"] or j2_judgement["change_summary"],
                    "stale_since": j1_judgement["stale_since"] or j2_judgement["stale_since"],
                    "reason": merged_reason,
                }
            else:
                model_disagreement = True
                model_agreement_summary = "Models disagree"
                merged_score = max(40, min(74, round((sc1 + sc2) / 2)))
                judgement = {
                    "status": "uncertain",
                    "score": merged_score,
                    "agreeing_sources": merged_agreeing,
                    "conflicting_sources": merged_conflicting,
                    "current_info": (
                        f"Models disagree — {j1_label} ({s1.upper()}): {j1_judgement['current_info']} | "
                        f"{j2_label} ({s2.upper()}): {j2_judgement['current_info']}"
                    ),
                    "current_value": j1_judgement["current_value"] or j2_judgement["current_value"],
                    "effective_date": j1_judgement["effective_date"] or j2_judgement["effective_date"],
                    "change_summary": j1_judgement["change_summary"] or j2_judgement["change_summary"],
                    "stale_since": j1_judgement["stale_since"] or j2_judgement["stale_since"],
                    "reason": (
                        f"Models disagree ({j1_label} judged '{s1}', {j2_label} judged '{s2}'). "
                        f"{j1_label}: {j1_judgement['reason']} | {j2_label}: {j2_judgement['reason']}"
                    ),
                }
        else:
            # Exactly one judge succeeded while the other failed: mark Uncertain with note.
            ok_j = j1_judgement or j2_judgement
            ok_label = j1_label if j1_judgement is not None else j2_label
            fail_label = j2_label if j1_judgement is not None else j1_label
            fail_msg = j2_err if j1_judgement is not None else j1_err
            model_disagreement = True
            model_agreement_summary = "Models disagree"
            partial_judge_error = f"Second judge ({fail_label}) failed: {fail_msg}"
            ok_score = ok_j["score"] if ok_j["score"] is not None else 55
            clamped_partial = max(40, min(74, ok_score))
            judge_verdicts = [
                {
                    "model": ok_label,
                    "status": ok_j["status"],
                    "score": ok_j["score"],
                    "current_info": ok_j["current_info"],
                    "reason": ok_j["reason"],
                },
                {
                    "model": fail_label,
                    "status": "error",
                    "score": None,
                    "current_info": "Judge request failed.",
                    "reason": fail_msg or "Request failed.",
                },
            ]
            judgement = {
                "status": "uncertain",
                "score": clamped_partial,
                "agreeing_sources": ok_j["agreeing_sources"],
                "conflicting_sources": True,
                "current_info": f"Marked Uncertain because {fail_label} failed while {ok_label} judged '{ok_j['status']}'. {ok_j['current_info']}",
                "current_value": ok_j["current_value"],
                "effective_date": ok_j["effective_date"],
                "change_summary": ok_j["change_summary"],
                "stale_since": ok_j["stale_since"],
                "reason": f"Models disagree ({ok_label} judged '{ok_j['status']}', {fail_label} failed: {fail_msg}). {ok_j['reason']}",
            }

    # Step 4: Compute status-banded score, conflict override, evidence breakdown, & confidence.
    scoring = compute_freshness_score(
        status=judgement["status"],
        llm_score=judgement["score"],
        current_info=judgement["current_info"],
        search_results=search_results,
        llm_agreeing_count=judgement["agreeing_sources"],
        today=today_date,
        reason=judgement["reason"],
        llm_conflicting=judgement["conflicting_sources"],
        entity_domains=resolved_entity_domains,
    )

    effective_status = scoring["resolved_status"]
    effective_confidence = scoring["confidence"]
    effective_why = scoring["why_this_score"]
    effective_reason = judgement["reason"]
    effective_conflict_detected = scoring["conflict_detected"]
    effective_conflict_note = scoring["conflict_note"]

    if model_disagreement:
        effective_status = "uncertain"
        effective_confidence = "low"
        if len(judge_verdicts) == 2:
            v1, v2 = judge_verdicts[0], judge_verdicts[1]
            effective_why = (
                f"Models disagree ({v1['model']}: '{v1['status']}' vs {v2['model']}: '{v2['status']}'); "
                f"marked 'uncertain' (40-74, low confidence). {effective_why}"
            )
    elif dual_judge and model_agreement_summary == "2/2 models agree":
        effective_why = f"2/2 models agree ({' vs '.join(judge_models)}). {effective_why}"

    if not model_disagreement and effective_conflict_detected and "disagree" not in effective_reason.lower():
        effective_reason = (
            f"Sources disagree ({scoring['agreement_summary']}). {effective_reason}"
        )

    stale_since, stale_for_months, stale_duration_label = estimate_stale_since_from_sources(
        status=effective_status,
        claim_text=claim_text,
        search_results=search_results,
        current_info=judgement["current_info"],
        current_value=judgement["current_value"],
        change_summary=judgement["change_summary"],
        effective_date=judgement["effective_date"],
        llm_stale_since=judgement["stale_since"],
        agreeing_indices=scoring["agreeing_indices"],
        today=today_date,
    )

    all_fallback_used = any(bool(m.get("fallback_used")) for m in judge_telemetry) if judge_telemetry else False

    out_dict = {
        "claim": claim_text,
        "span": span_text,
        "category": category,
        "search_query": search_query,
        "status": effective_status,
        "score": scoring["score"],
        "score_breakdown": scoring["score_breakdown"],
        "why_this_score": effective_why,
        "confidence": effective_confidence,
        "risk_level": risk_level,
        "current_info": judgement["current_info"],
        "current_value": judgement["current_value"],
        "effective_date": judgement["effective_date"],
        "change_summary": judgement["change_summary"],
        "stale_since": stale_since,
        "stale_for_months": stale_for_months,
        "stale_duration_label": stale_duration_label,
        "reason": effective_reason,
        "sources": top_sources,
        "entity_domains": resolved_entity_domains,
        "sources_agreeing": scoring["sources_agreeing"],
        "sources_total": scoring["sources_total"],
        "agreement_summary": scoring["agreement_summary"],
        "conflict_detected": effective_conflict_detected,
        "conflict_note": effective_conflict_note,
        "dual_judge_used": bool(dual_judge),
        "judge_models": judge_models,
        "judge_verdicts": judge_verdicts,
        "model_disagreement": model_disagreement,
        "model_agreement_summary": model_agreement_summary,
        "judge_telemetry": judge_telemetry,
        "tavily_credits_used": round(total_credits, 2),
        "cache_hit": overall_cache_hit,
        "search_strategy": search_strategy,
        "extracted_url": extract_info["url"] if extract_info.get("content") else best_url,
        "search_warning": search_meta.get("partial_error"),
        "gateway_metrics": gateway_metrics,
        "fallback_used": all_fallback_used,
        "primary_model": gateway_metrics.get("primary_model") if gateway_metrics else None,
        "model": model_used,
    }
    if partial_judge_error:
        out_dict["error"] = partial_judge_error
    return out_dict


def _make_worker_tavily_client(shared_client: TavilyClient) -> tuple[TavilyClient, bool]:
    """
    Create an isolated TavilyClient (separate requests.Session) for a worker thread
    when `shared_client` is a real TavilyClient without monkeypatched methods.
    Mocked or custom-session clients pass through untouched.
    """
    if (
        isinstance(shared_client, TavilyClient)
        and isinstance(getattr(shared_client, "session", None), requests.Session)
        and not getattr(shared_client, "_external_session", False)
        and "search" not in vars(shared_client)
        and "extract" not in vars(shared_client)
        and getattr(shared_client, "api_key", None)
    ):
        worker_client = TavilyClient(
            api_key=shared_client.api_key,
            proxies=getattr(shared_client, "proxies", None),
            api_base_url=getattr(shared_client, "base_url", None),
        )
        return worker_client, True
    return shared_client, False


def verify_claims(
    claims: list[dict[str, str]],
    progress_callback: Callable[[int, int, dict], None] | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
    entity_domains: dict[str, list[str] | str] | None = None,
    dual_judge: bool = False,
    secondary_judge_model: str | None = None,
    max_workers: int = PARALLEL_VERIFICATION_WORKERS,
) -> list[dict]:
    """
    Verify a list of claims against live web sources in parallel.

    Runs up to `max_workers` (default 3) claims concurrently using a
    ThreadPoolExecutor with thread-safe SQLite caching, per-worker Tavily HTTP
    sessions, and ordered result assembly. `progress_callback(completed, total,
    result)` is always invoked sequentially on the submitting thread so callers
    (like Streamlit UI state and credit counters) remain race-free.

    Args:
        claims: List of claim dicts from extractor.extract_claims().
        progress_callback: Optional callback invoked as callback(completed, total, result).
        use_cache: Whether to use the 6-hour on-disk Tavily cache.
        cache_path: Optional override path for the SQLite cache file.
        entity_domains: Optional override mapping entity keys to official domains.
        dual_judge: When True, run two judge models per claim via Bifrost.
        secondary_judge_model: Optional explicit override for Judge 2.
        max_workers: Maximum parallel verification workers (default 3).

    Returns:
        List of verification result dicts in the same order as `claims`.

    Raises:
        ValueError: If TAVILY_API_KEY is missing.
    """
    if not claims:
        return []

    # Validate Tavily configuration upfront on the calling thread.
    tavily_client = get_tavily_client()
    total = len(claims)
    worker_count = max(1, min(total, int(max_workers or PARALLEL_VERIFICATION_WORKERS)))

    # Propagate thread-local outage toggle state into worker threads cleanly.
    parent_outage_raw = getattr(_llm_mod._thread_state, "simulate_outage", None)

    def _run_single(item: dict[str, str]) -> dict:
        pop_last_gateway_metrics()
        if parent_outage_raw is not None:
            set_simulate_outage(parent_outage_raw)
        else:
            _llm_mod._thread_state.simulate_outage = None

        worker_client, should_close = _make_worker_tavily_client(tavily_client)
        try:
            call_kwargs: dict = {
                "tavily_client": worker_client,
                "use_cache": use_cache,
                "cache_path": cache_path,
                "entity_domains": entity_domains,
            }
            if dual_judge:
                call_kwargs["dual_judge"] = True
            if secondary_judge_model:
                call_kwargs["secondary_judge_model"] = secondary_judge_model
            return verify_claim(item, **call_kwargs)
        finally:
            if should_close:
                try:
                    worker_client.session.close()
                except Exception:
                    pass
            pop_last_gateway_metrics()
            _llm_mod._thread_state.simulate_outage = None

    verified_results: list[dict | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_idx = {
            executor.submit(_run_single, claim_item): idx
            for idx, claim_item in enumerate(claims)
        }
        for future in as_completed(future_to_idx):
            orig_idx = future_to_idx[future]
            result = future.result()
            verified_results[orig_idx] = result
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, result)

    return [res for res in verified_results if res is not None]


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
        score_val = item.get("score")
        score_str = "N/A" if score_val is None else f"{score_val}/100"
        extras = []
        if item.get("current_value"):
            extras.append(f"Current Value: {item['current_value']}")
        if item.get("effective_date"):
            extras.append(f"Effective Date: {item['effective_date']}")
        if item.get("change_summary"):
            extras.append(f"Change Summary: {item['change_summary']}")
        if item.get("stale_since"):
            dur = f" ({item['stale_duration_label']})" if item.get("stale_duration_label") else ""
            extras.append(f"Stale Since: {item['stale_since']}{dur}")
        if item.get("conflict_note"):
            extras.append(f"Source Agreement Note: {item['conflict_note']}")
        if item.get("model_agreement_summary"):
            extras.append(f"Dual-Judge Consensus: {item['model_agreement_summary']}")
        extra_block = ("\n   " + "\n   ".join(extras)) if extras else ""

        findings_lines.append(
            f"{idx}. Claim: {item.get('claim', '')}\n"
            f"   Status: {item.get('status', 'uncertain')} (Score: {score_str})\n"
            f"   Current Info: {item.get('current_info', '')}"
            f"{extra_block}\n"
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
    rewritten, model_used = _invoke_call_llm(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.2,
        task="rewrite",
    )
    _verifier_state.last_metrics = _capture_call_metrics(
        "Rewrite answer",
        model_used,
        (time.perf_counter() - llm_t0) * 1000.0,
        task="rewrite",
    )
    return rewritten, model_used
