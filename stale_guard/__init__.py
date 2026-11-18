"""stale_guard - Freshness Guard SDK for AI chatbots and RAG pipelines.

Public entry point:
    from stale_guard import guard

    result = guard(question="...", answer="...", context=None, max_claims=6)
"""

from stale_guard.guard import (
    DEFAULT_MAX_CLAIMS,
    MAX_GUARD_ANSWER_CHARS,
    MAX_GUARD_CONTEXT_CHARS,
    MAX_GUARD_QUESTION_CHARS,
    compute_guard_verdict,
    guard,
)

__all__ = [
    "DEFAULT_MAX_CLAIMS",
    "MAX_GUARD_ANSWER_CHARS",
    "MAX_GUARD_CONTEXT_CHARS",
    "MAX_GUARD_QUESTION_CHARS",
    "compute_guard_verdict",
    "guard",
]
