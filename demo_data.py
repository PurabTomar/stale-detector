"""
demo_data.py - Offline cached sample audits for Demo mode.

Allows Stale-Answer Detector to work smoothly with zero live API or network calls
during judging, demos, or API outages by returning pre-verified cached audits
stored in `demo_cache.json`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEMO_CACHE_PATH = Path(__file__).resolve().parent / "demo_cache.json"


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def load_demo_cache(cache_path: str | Path | None = None) -> dict:
    """Load the offline cached demo audits bundle from `demo_cache.json`."""
    path = Path(cache_path) if cache_path is not None else DEMO_CACHE_PATH
    if not path.exists():
        return {"pasted_examples": [], "ask_sample": {}, "url_sample": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"pasted_examples": [], "ask_sample": {}, "url_sample": {}}
    except (OSError, json.JSONDecodeError):
        return {"pasted_examples": [], "ask_sample": {}, "url_sample": {}}


def get_pasted_demo_examples(cache_path: str | Path | None = None) -> list[dict]:
    """Return all cached pasted-answer sample definitions."""
    data = load_demo_cache(cache_path)
    items = data.get("pasted_examples") or []
    return [copy.deepcopy(item) for item in items if isinstance(item, dict)]


def find_cached_pasted_audit(
    answer_text: str | None = None,
    *,
    example_index: int | None = None,
    cache_path: str | Path | None = None,
) -> tuple[dict, bool]:
    """
    Look up a cached pasted-answer audit for Demo mode.

    Returns `(sample_dict, exact_match_bool)`:
    - If `example_index` or `answer_text` matches one of the 4 preset demos (by exact
      text or overlapping key terms), returns that exact sample.
    - If the judge enters custom text while Demo mode is enabled, falls back to the
      first cached sample audit (`exact_match=False`) so the UI never fails offline.
    """
    samples = get_pasted_demo_examples(cache_path)
    if not samples:
        raise FileNotFoundError("Offline demo_cache.json is missing pasted_examples.")

    if example_index is not None and 0 <= example_index < len(samples):
        return samples[example_index], True

    norm_in = _normalize_text(answer_text or "")
    if not norm_in:
        return samples[0], False

    for item in samples:
        if _normalize_text(item.get("answer_text", "")) == norm_in:
            return item, True

    keyword_groups = [
        ("iphone", ("iphone", "a17", "ios 17", "1,199", "$1,199")),
        ("gpt", ("gpt-4", "turbo", "128k", "million input", "$30")),
        ("gst", ("gst", "slab", "28%", "20 lakh", "goods and services tax")),
        ("ceo", ("yaccarino", "altman", "openai", "$86 billion", "twitter")),
    ]
    for sample_id, keywords in keyword_groups:
        if any(kw in norm_in for kw in keywords):
            for item in samples:
                if item.get("id") == sample_id:
                    return item, True

    return samples[0], False


def get_cached_ask_audit(question_text: str | None = None, cache_path: str | Path | None = None) -> dict:
    """Return the cached Ask & Audit sample payload for Demo mode."""
    data = load_demo_cache(cache_path)
    sample = copy.deepcopy(data.get("ask_sample") or {})
    if not sample:
        raise FileNotFoundError("Offline demo_cache.json is missing ask_sample.")
    clean_q = str(question_text or "").strip()
    sample["question"] = clean_q or str(sample.get("default_question") or "Sample fact-check question")
    return sample


def get_cached_url_audit(url_text: str | None = None, cache_path: str | Path | None = None) -> dict:
    """Return the cached Audit-a-URL sample payload for Demo mode."""
    data = load_demo_cache(cache_path)
    sample = copy.deepcopy(data.get("url_sample") or {})
    if not sample:
        raise FileNotFoundError("Offline demo_cache.json is missing url_sample.")
    clean_url = str(url_text or "").strip()
    sample["url"] = clean_url or str(sample.get("default_url") or "https://example.org/blog/2026-tech-and-policy-brief")
    return sample
