"""Minimal ~15-line example wrapping any chatbot reply with stale_guard."""

from stale_guard import guard


def safe_chatbot_reply(question: str, raw_bot_reply: str, rag_context: str | None = None) -> str:
    """Return the bot's reply if fresh, or a cited correction if stale/incorrect."""
    result = guard(question=question, answer=raw_bot_reply, context=rag_context, max_claims=6)
    if result["verdict"] in {"stale", "incorrect"}:
        return result["corrected_answer"]
    if result["verdict"] in {"needs_review", "verification_failed"}:
        return f"{result['corrected_answer']}\n\n(Note: some facts could not be verified live.)"
    return raw_bot_reply


if __name__ == "__main__":
    print(safe_chatbot_reply("What is the latest Python release?", "Python 3.11 is the latest stable release."))
