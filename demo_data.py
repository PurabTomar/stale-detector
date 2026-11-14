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
SAVED_RESULTS_DATE = "Nov 13, 2026"


def _tag_saved_results(sample_dict: dict) -> dict:
    if not isinstance(sample_dict, dict):
        return sample_dict
    sample_dict.setdefault("source_origin", "saved")
    sample_dict.setdefault("saved_date", SAVED_RESULTS_DATE)
    results = sample_dict.get("results")
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                r.setdefault("source_origin", "saved")
                r.setdefault("saved_date", SAVED_RESULTS_DATE)
    return sample_dict


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
    return [_tag_saved_results(copy.deepcopy(item)) for item in items if isinstance(item, dict)]


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
    return _tag_saved_results(sample)


def get_cached_url_audit(url_text: str | None = None, cache_path: str | Path | None = None) -> dict:
    """Return the cached Audit-a-URL sample payload for Demo mode."""
    data = load_demo_cache(cache_path)
    sample = copy.deepcopy(data.get("url_sample") or {})
    if not sample:
        raise FileNotFoundError("Offline demo_cache.json is missing url_sample.")
    clean_url = str(url_text or "").strip()
    sample["url"] = clean_url or str(sample.get("default_url") or "https://example.org/blog/2026-tech-and-policy-brief")
    return _tag_saved_results(sample)


GUIDED_DEMO_STEPS = [
    {
        "step": 0,
        "title": "1. iPhone (Fresh)",
        "badge": "Fresh",
        "badge_color": "#16A34A",
        "caption": "Official Apple specs & pricing verified current today against apple.com.",
        "sample": {
            "id": "iphone_fresh",
            "label": "iPhone release & pricing (Fresh)",
            "answer_text": "Apple's latest flagship smartphone is the iPhone 16 Pro Max, starting at $1,199 in the United States. It is powered by the A18 Pro chip and ships with iOS 18 out of the box.",
            "extract_model": "gemini/gemini-2.5-flash-lite",
            "results": [
                {
                    "claim": "Apple's latest flagship smartphone is the iPhone 16 Pro Max, starting at $1,199 in the United States.",
                    "status": "fresh",
                    "score": 96,
                    "confidence": "high",
                    "risk_level": "low",
                    "category": "pricing",
                    "reason": "Apple's current flagship is indeed the iPhone 16 Pro Max, released in September 2024 starting at $1,199.",
                    "current_info": "iPhone 16 Pro Max starts at $1,199 (256GB). Current flagship as of 2025/2026.",
                    "entity_domains": ["apple.com"],
                    "sources": [
                        {"url": "https://www.apple.com/iphone-16-pro/", "title": "iPhone 16 Pro - Apple", "source_type": "Primary", "stance": "support", "published_date": "2024-09-09", "snippet": "iPhone 16 Pro and iPhone 16 Pro Max. Built for Apple Intelligence. Starting at $1,199."},
                        {"url": "https://www.theverge.com/2024/9/9/apple-iphone-16-pro-max-specs-price", "title": "Apple iPhone 16 Pro Max specs - The Verge", "source_type": "News", "stance": "support", "published_date": "2024-09-09", "snippet": "The iPhone 16 Pro Max starts at $1,199 for 256GB of storage."}
                    ],
                    "score_breakdown": {"evidence_agreement": 100, "authority": 95, "recency": 95, "model_agreement": 95},
                    "gateway_metrics": {"latency_ms": 320, "model": "gemini-2.5-flash"}
                },
                {
                    "claim": "The flagship iPhone is powered by the A18 Pro chip.",
                    "status": "fresh",
                    "score": 98,
                    "confidence": "high",
                    "risk_level": "low",
                    "category": "api_docs",
                    "reason": "The iPhone 16 Pro Max is powered by Apple's A18 Pro chip built on 3nm architecture.",
                    "current_info": "Powered by Apple A18 Pro with 6-core GPU and 16-core Neural Engine.",
                    "entity_domains": ["apple.com"],
                    "sources": [
                        {"url": "https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/", "title": "Apple debuts iPhone 16 Pro - Newsroom", "source_type": "Primary", "stance": "support", "published_date": "2024-09-09", "snippet": "Powered by the breakthrough A18 Pro chip with industry-leading CPU performance."}
                    ],
                    "score_breakdown": {"evidence_agreement": 100, "authority": 100, "recency": 95, "model_agreement": 95},
                    "gateway_metrics": {"latency_ms": 280, "model": "gemini-2.5-flash"}
                },
                {
                    "claim": "Apple's flagship iPhone ships with iOS 18 out of the box.",
                    "status": "fresh",
                    "score": 95,
                    "confidence": "high",
                    "risk_level": "low",
                    "category": "api_docs",
                    "reason": "iPhone 16 Pro models shipped with iOS 18 preinstalled.",
                    "current_info": "Ships with iOS 18 out of the box, with Apple Intelligence updates.",
                    "entity_domains": ["apple.com"],
                    "sources": [
                        {"url": "https://www.apple.com/ios/ios-18/", "title": "iOS 18 - Apple", "source_type": "Primary", "stance": "support", "published_date": "2024-09-16", "snippet": "iOS 18 is available now for iPhone 16 family and compatible devices."}
                    ],
                    "score_breakdown": {"evidence_agreement": 100, "authority": 100, "recency": 90, "model_agreement": 90},
                    "gateway_metrics": {"latency_ms": 290, "model": "gemini-2.5-flash"}
                }
            ],
            "corrected_answer": "Apple's latest flagship smartphone is the iPhone 16 Pro Max, starting at $1,199 in the United States. It is powered by the A18 Pro chip and ships with iOS 18 out of the box."
        }
    },
    {
        "step": 1,
        "title": "2. GST Slabs (Outdated)",
        "badge": "Outdated",
        "badge_color": "#EA580C",
        "caption": "Historical tax slabs replaced by recent GST Council rationalizations.",
        "sample": {
            "id": "gst_outdated",
            "label": "India GST slabs & threshold (Outdated)",
            "answer_text": "Under India's Goods and Services Tax (GST) regime, goods and services fall under four main tax slabs—5%, 12%, 18%, and 28%—and businesses with annual turnover above ₹20 lakh must register for GST.",
            "extract_model": "gemini/gemini-2.5-flash-lite",
            "results": [
                {
                    "claim": "Under India's GST regime, goods and services fall under four main tax slabs: 5%, 12%, 18%, and 28%.",
                    "status": "outdated",
                    "score": 12,
                    "confidence": "high",
                    "risk_level": "critical",
                    "category": "regulation",
                    "reason": "The GST Council has restructured tax rates with exemptions and rationalizations, and multiple ministerial groups have proposed merging the 12% and 18% slabs into a single standard rate.",
                    "current_info": "GST rate rationalization ongoing; special rates of 0%, 0.25%, 3% exist alongside standard slabs.",
                    "stale_since": "2024-06-22",
                    "stale_duration_label": "stale for ~18 months",
                    "sources": [
                        {"url": "https://cbic-gst.gov.in", "title": "CBIC GST Portal", "source_type": "Primary", "stance": "contradict", "published_date": "2024-06-22", "snippet": "GST Council continues rate rationalization exercises with updated notifications."}
                    ],
                    "score_breakdown": {"evidence_agreement": 10, "authority": 90, "recency": 85, "model_agreement": 90},
                    "gateway_metrics": {"latency_ms": 350, "model": "gemini-2.5-flash"}
                },
                {
                    "claim": "Businesses with annual turnover above ₹20 lakh must register for GST in India.",
                    "status": "fresh",
                    "score": 88,
                    "confidence": "high",
                    "risk_level": "low",
                    "category": "regulation",
                    "reason": "Threshold remains ₹20 lakh for services (and ₹40 lakh for supplier of goods in normal states).",
                    "current_info": "₹20 lakh threshold remains current for service providers.",
                    "sources": [
                        {"url": "https://www.gst.gov.in", "title": "Goods and Services Tax Portal", "source_type": "Primary", "stance": "support", "published_date": "2024-01-01", "snippet": "Mandatory registration threshold for services is ₹20 lakh turnover."}
                    ],
                    "score_breakdown": {"evidence_agreement": 90, "authority": 95, "recency": 80, "model_agreement": 90},
                    "gateway_metrics": {"latency_ms": 310, "model": "gemini-2.5-flash"}
                }
            ],
            "corrected_answer": "Under India's Goods and Services Tax (GST) regime, the threshold for mandatory registration for service providers remains ₹20 lakh, while GST rate structures continue to evolve under GST Council rationalizations."
        }
    },
    {
        "step": 2,
        "title": "3. World Leadership (Incorrect)",
        "badge": "Incorrect",
        "badge_color": "#DC2626",
        "caption": "Factually false statement contradicted by official authoritative records.",
        "sample": {
            "id": "leadership_incorrect",
            "label": "World leadership & geography (Incorrect)",
            "answer_text": "The Prime Minister of India is Rahul Gandhi, while the capital of Australia is Sydney. Meanwhile, the standard VAT rate in the United Kingdom remains 17.5%.",
            "extract_model": "gemini/gemini-2.5-flash-lite",
            "results": [
                {
                    "claim": "The Prime Minister of India is Rahul Gandhi.",
                    "status": "incorrect",
                    "score": 5,
                    "confidence": "high",
                    "risk_level": "critical",
                    "category": "leadership",
                    "correction": "Narendra Modi has served as the Prime Minister of India since May 2014.",
                    "reason": "Factually incorrect and never true. Narendra Modi is the incumbent Prime Minister of India, whereas Rahul Gandhi serves as the Leader of the Opposition in the Lok Sabha.",
                    "current_info": "Narendra Modi is the Prime Minister of India (assumed office May 2014, re-elected 2019 and 2024).",
                    "sources": [
                        {"url": "https://www.pmindia.gov.in", "title": "Prime Minister's Office - Government of India", "source_type": "Primary", "stance": "contradict", "published_date": "2024-06-09", "snippet": "Shri Narendra Modi took oath as the Prime Minister of India for a third consecutive term."}
                    ],
                    "score_breakdown": {"evidence_agreement": 0, "authority": 100, "recency": 95, "model_agreement": 95},
                    "gateway_metrics": {"latency_ms": 290, "model": "gemini-2.5-flash"}
                },
                {
                    "claim": "The capital of Australia is Sydney.",
                    "status": "incorrect",
                    "score": 8,
                    "confidence": "high",
                    "risk_level": "critical",
                    "category": "other",
                    "correction": "Canberra is the capital of Australia.",
                    "reason": "Sydney is Australia's most populous city, but Canberra was chosen as the national capital in 1908 and has served as capital since 1913.",
                    "current_info": "Canberra is the official capital of Australia.",
                    "sources": [
                        {"url": "https://www.australia.gov.au", "title": "Australian Government Official Portal", "source_type": "Primary", "stance": "contradict", "published_date": "2024-01-01", "snippet": "Canberra is the capital city of Australia, situated in the Australian Capital Territory."}
                    ],
                    "score_breakdown": {"evidence_agreement": 0, "authority": 100, "recency": 90, "model_agreement": 95},
                    "gateway_metrics": {"latency_ms": 260, "model": "gemini-2.5-flash"}
                },
                {
                    "claim": "The standard VAT rate in the United Kingdom remains 17.5%.",
                    "status": "outdated",
                    "score": 12,
                    "confidence": "high",
                    "risk_level": "moderate",
                    "category": "regulation",
                    "correction": "The standard rate of VAT in the UK is 20%.",
                    "stale_since": "2011-01-04",
                    "stale_duration_label": "stale for ~13 years",
                    "reason": "The UK standard VAT rate was raised from 17.5% to 20% on 4 January 2011 by HMRC.",
                    "current_info": "The standard rate of Value Added Tax (VAT) in the UK is 20%.",
                    "sources": [
                        {"url": "https://www.gov.uk/vat-rates", "title": "VAT rates - GOV.UK", "source_type": "Primary", "stance": "contradict", "published_date": "2024-01-01", "snippet": "Standard rate 20% - Most goods and services. The rate changed to 20% on 4 January 2011."}
                    ],
                    "score_breakdown": {"evidence_agreement": 10, "authority": 100, "recency": 85, "model_agreement": 90},
                    "gateway_metrics": {"latency_ms": 300, "model": "gemini-2.5-flash"}
                }
            ],
            "corrected_answer": "The Prime Minister of India is Narendra Modi, while the capital of Australia is Canberra. In the United Kingdom, the standard rate of VAT is 20%."
        }
    }
]


def get_guided_demo_step(step_idx: int = 0) -> dict:
    """Return a guided demo step definition (0=iPhone, 1=GST, 2=Leadership)."""
    idx = max(0, min(len(GUIDED_DEMO_STEPS) - 1, int(step_idx)))
    step_copy = copy.deepcopy(GUIDED_DEMO_STEPS[idx])
    if isinstance(step_copy.get("sample"), dict):
        step_copy["sample"] = _tag_saved_results(step_copy["sample"])
    return step_copy
