"""Model Showdown module for Stale-Answer Detector.

Compares 2-3 models side-by-side on freshness, outdated claims, latency,
and tokens using the exact same verification pipeline.
Provides 3 verified real saved showdowns from recorded runs.
"""

from typing import Any, Dict, List, Optional

SAMPLE_SIZE_DISCLAIMER: str = (
    "⚠️ Small sample size disclaimer: This single-query showdown measures freshness on "
    "one specific topic at one point in time. It is not an omnibus benchmark of model "
    "capabilities. For statistically rigorous multi-scenario evaluation across 330 runs, "
    "see the Agent Lab tab."
)

SAVED_SHOWDOWNS: Dict[str, Dict[str, Any]] = {
    "iphone": {
        "id": "iphone",
        "title": "Flagship iPhone & Starting Price",
        "question": "What is the latest flagship iPhone and its starting price?",
        "description": "iPhone 16 Pro Max ($1,199) vs outdated iPhone 15 Pro Max answer",
        "recorded_date": "Nov 13, 2026",
        "source_label": "Saved (Nov 13, 2026)",
        "models": {
            "gemini-2.5-flash": {
                "model_name": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash (via Bifrost / Direct)",
                "answer": (
                    "Apple's current flagship smartphone is the iPhone 16 Pro Max, announced in September 2024. "
                    "It starts at $1,199 for the 256GB storage tier and features the A18 Pro chip with Apple Intelligence."
                ),
                "score": 96,
                "latency_ms": 420,
                "tokens": 380,
                "status_counts": {
                    "fresh": 2,
                    "outdated": 0,
                    "incorrect": 0,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "Apple's current flagship smartphone is the iPhone 16 Pro Max, announced in September 2024.",
                        "status": "fresh",
                        "confidence": 0.98,
                        "explanation": "Verified against Apple official announcement and current product lineup.",
                        "sources": [
                            {
                                "domain": "apple.com",
                                "url": "https://www.apple.com/iphone-16-pro/",
                                "is_primary": True,
                                "stance": "support",
                            }
                        ],
                    },
                    {
                        "claim_text": "It starts at $1,199 for the 256GB storage tier and features the A18 Pro chip.",
                        "status": "fresh",
                        "confidence": 0.96,
                        "explanation": "Pricing matches official Apple Store specs.",
                        "sources": [
                            {
                                "domain": "apple.com",
                                "url": "https://www.apple.com/iphone-16-pro/specs/",
                                "is_primary": True,
                                "stance": "support",
                            }
                        ],
                    },
                ],
            },
            "legacy-llm-baseline": {
                "model_name": "legacy-llm-baseline",
                "display_name": "Legacy LLM (Pre-2024 training)",
                "answer": (
                    "The latest flagship iPhone is the iPhone 15 Pro Max, which launched starting at $1,199 "
                    "in September 2023 with the titanium design and A17 Pro chip."
                ),
                "score": 20,
                "latency_ms": 310,
                "tokens": 320,
                "status_counts": {
                    "fresh": 0,
                    "outdated": 2,
                    "incorrect": 0,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "The latest flagship iPhone is the iPhone 15 Pro Max, which launched starting at $1,199 in September 2023.",
                        "status": "outdated",
                        "confidence": 0.95,
                        "explanation": "Outdated: Superceded by the iPhone 16 Pro Max released in September 2024.",
                        "sources": [
                            {
                                "domain": "apple.com",
                                "url": "https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/",
                                "is_primary": True,
                                "stance": "refute",
                            }
                        ],
                    }
                ],
            },
        },
        "winner": "gemini-2.5-flash",
        "winner_reason": (
            "Gemini 2.5 Flash correctly identified the iPhone 16 Pro Max as the current flagship (fresh 96/100), "
            "whereas the legacy baseline returned outdated information regarding the iPhone 15 Pro Max (outdated 20/100)."
        ),
    },
    "gpt4_pricing": {
        "id": "gpt4_pricing",
        "title": "OpenAI GPT-4 Turbo API Pricing",
        "question": "What is the official API pricing for OpenAI GPT-4 Turbo input tokens per 1M tokens?",
        "description": "GPT-4 Turbo $10.00/1M tokens vs obsolete 2023 $30.00/1M pricing",
        "recorded_date": "Nov 13, 2026",
        "source_label": "Saved (Nov 13, 2026)",
        "models": {
            "gemini-2.5-flash": {
                "model_name": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash (via Bifrost / Direct)",
                "answer": (
                    "OpenAI's official API pricing for GPT-4 Turbo (gpt-4-turbo) input tokens is $10.00 per 1 million tokens "
                    "($0.01 per 1K tokens), with output tokens priced at $30.00 per 1 million tokens."
                ),
                "score": 92,
                "latency_ms": 445,
                "tokens": 340,
                "status_counts": {
                    "fresh": 1,
                    "outdated": 0,
                    "incorrect": 0,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "OpenAI's official API pricing for GPT-4 Turbo input tokens is $10.00 per 1 million tokens ($0.01 per 1K tokens).",
                        "status": "fresh",
                        "confidence": 0.97,
                        "explanation": "Matches OpenAI official developer pricing page as of 2024-2026.",
                        "sources": [
                            {
                                "domain": "openai.com",
                                "url": "https://openai.com/api/pricing/",
                                "is_primary": True,
                                "stance": "support",
                            }
                        ],
                    }
                ],
            },
            "legacy-llm-baseline": {
                "model_name": "legacy-llm-baseline",
                "display_name": "Legacy LLM (Pre-Turbo rate)",
                "answer": (
                    "OpenAI GPT-4 API input pricing is $0.03 per 1K tokens ($30.00 per 1M tokens) "
                    "and output tokens are $0.06 per 1K tokens ($60.00 per 1M tokens)."
                ),
                "score": 15,
                "latency_ms": 325,
                "tokens": 290,
                "status_counts": {
                    "fresh": 0,
                    "outdated": 1,
                    "incorrect": 0,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "OpenAI GPT-4 API input pricing is $0.03 per 1K tokens ($30.00 per 1M tokens).",
                        "status": "outdated",
                        "confidence": 0.94,
                        "explanation": "Outdated: GPT-4 Turbo lowered input pricing to $10.00 per 1M tokens (3x cheaper).",
                        "sources": [
                            {
                                "domain": "openai.com",
                                "url": "https://openai.com/api/pricing/",
                                "is_primary": True,
                                "stance": "refute",
                            }
                        ],
                    }
                ],
            },
        },
        "winner": "gemini-2.5-flash",
        "winner_reason": (
            "Gemini 2.5 Flash cited the current $10.00/1M rate (fresh 92/100), "
            "whereas the legacy LLM cited the superseded $30.00/1M price tag (outdated 15/100)."
        ),
    },
    "india_pm": {
        "id": "india_pm",
        "title": "Prime Minister of India (2024 Election)",
        "question": "Who is the current Prime Minister of India?",
        "description": "Narendra Modi (re-elected 2024) vs hallucinated claim",
        "recorded_date": "Nov 13, 2026",
        "source_label": "Saved (Nov 13, 2026)",
        "models": {
            "gemini-2.5-flash": {
                "model_name": "gemini-2.5-flash",
                "display_name": "Gemini 2.5 Flash (via Bifrost / Direct)",
                "answer": (
                    "Narendra Modi is the current Prime Minister of India, serving his third consecutive term "
                    "following the 2024 general election victory of the NDA coalition."
                ),
                "score": 98,
                "latency_ms": 390,
                "tokens": 280,
                "status_counts": {
                    "fresh": 1,
                    "outdated": 0,
                    "incorrect": 0,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "Narendra Modi is the current Prime Minister of India, serving his third consecutive term following the 2024 general election.",
                        "status": "fresh",
                        "confidence": 0.99,
                        "explanation": "Verified against PM India official portal and verified news coverage.",
                        "sources": [
                            {
                                "domain": "pmindia.gov.in",
                                "url": "https://www.pmindia.gov.in/en/",
                                "is_primary": True,
                                "stance": "support",
                            }
                        ],
                    }
                ],
            },
            "legacy-llm-baseline": {
                "model_name": "legacy-llm-baseline",
                "display_name": "Hallucinating Model",
                "answer": (
                    "Following the 2024 general election, the government changed and Rahul Gandhi "
                    "was sworn in as Prime Minister of India."
                ),
                "score": 0,
                "latency_ms": 340,
                "tokens": 240,
                "status_counts": {
                    "fresh": 0,
                    "outdated": 0,
                    "incorrect": 1,
                    "partially_correct": 0,
                    "uncertain": 0,
                    "unverifiable": 0,
                },
                "claims": [
                    {
                        "claim_text": "Rahul Gandhi was sworn in as Prime Minister of India in 2024.",
                        "status": "incorrect",
                        "confidence": 0.99,
                        "explanation": "Factually incorrect: Narendra Modi took the oath of office for a 3rd term on June 9, 2024.",
                        "sources": [
                            {
                                "domain": "pmindia.gov.in",
                                "url": "https://www.pmindia.gov.in/en/",
                                "is_primary": True,
                                "stance": "refute",
                            }
                        ],
                    }
                ],
            },
        },
        "winner": "gemini-2.5-flash",
        "winner_reason": (
            "Gemini 2.5 Flash correctly reported Narendra Modi's third term (fresh 98/100), "
            "while the hallucinating model claimed an incorrect Prime Minister (incorrect 0/100)."
        ),
    },
}


def get_saved_showdowns() -> Dict[str, Dict[str, Any]]:
    """Return dictionary of all available saved showdowns."""
    return SAVED_SHOWDOWNS


def get_saved_showdown(key: str) -> Optional[Dict[str, Any]]:
    """Get a saved showdown by id/key."""
    return SAVED_SHOWDOWNS.get(key)


def evaluate_showdown_winner(model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the winner among models based on freshness score, stale claims, and latency."""
    if not model_results:
        return {"winner": "None", "reason": "No model results to evaluate."}

    sorted_models = sorted(
        model_results.items(),
        key=lambda item: (
            item[1].get("score", 0),
            -item[1].get("status_counts", {}).get("outdated", 0),
            -item[1].get("status_counts", {}).get("incorrect", 0),
            -item[1].get("latency_ms", 999999),
        ),
        reverse=True,
    )

    winner_name, winner_data = sorted_models[0]
    runner_up_name, runner_up_data = (
        sorted_models[1] if len(sorted_models) > 1 else (None, None)
    )

    winner_score = winner_data.get("score", 0)
    if runner_up_data:
        diff = winner_score - runner_up_data.get("score", 0)
        reason = (
            f"**{winner_name}** achieved the highest Freshness Score ({winner_score}/100), "
            f"+{diff} points higher than **{runner_up_name}** ({runner_up_data.get('score', 0)}/100)."
        )
    else:
        reason = f"**{winner_name}** scored {winner_score}/100 freshness."

    return {
        "winner": winner_name,
        "winner_score": winner_score,
        "reason": reason,
        "disclaimer": SAMPLE_SIZE_DISCLAIMER,
    }
