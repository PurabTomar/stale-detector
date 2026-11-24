"""
heuristic.py - Pure-Python regex heuristic for time-sensitive phrase detection.

No API calls; zero I/O. Called on every text-area blur/Ctrl+Enter to give the
user a fast "pre-flight" preview of which phrases are likely to go stale —
clearly labelled as heuristic only, not verified.

Returns a list of dicts: {start, end, text, reason, category}
sorted by start offset.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Pattern table: (category, reason_label, compiled_regex)
# Order matters: earlier patterns take priority when spans overlap.
# ---------------------------------------------------------------------------

_RAW_PATTERNS: list[tuple[str, str, str]] = [
    # Price / currency amounts
    (
        "price",
        "price / currency",
        r"\b(?:USD|GBP|EUR|INR|JPY|AUD|CAD|CHF|SGD|HKD|CNY|KRW)\s*[\d,]+(?:\.\d+)?"
        r"|\$[\d,]+(?:\.\d+)?"
        r"|£[\d,]+(?:\.\d+)?"
        r"|€[\d,]+(?:\.\d+)?"
        r"|₹[\d,]+(?:\.\d+)?"
        r"|¥[\d,]+(?:\.\d+)?"
        r"|\b[\d,]+(?:\.\d+)?\s+(?:USD|GBP|EUR|INR|JPY|AUD|CAD|CHF|SGD|HKD|CNY|KRW)\b",
    ),
    # Percentage figures
    (
        "statistic",
        "percentage figure",
        r"\b\d+(?:\.\d+)?\s*%",
    ),
    # Year / full date
    (
        "other",
        "year / date",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
        r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}\b",
    ),
    # Software version numbers
    (
        "version",
        "software version",
        r"\bv?\d+\.\d+(?:\.\d+){0,2}\b",
    ),
    # Temporal / recency words
    (
        "records",
        "recency keyword",
        r"\b(?:latest|current|newest|most\s+recent|up-to-date|up\s+to\s+date"
        r"|today|now|this\s+year|this\s+month|as\s+of|at\s+present|at\s+the\s+time"
        r"|recently|just\s+released|just\s+launched)\b",
    ),
    # Office / leadership titles followed by a name-like sequence
    (
        "leadership",
        "office / leadership title",
        r"\b(?:CEO|CFO|CTO|COO|CMO|CPO|CIO|CISO|President|Vice\s+President|VP"
        r"|Prime\s+Minister|PM|Chancellor|Governor|Mayor|Director|Head\s+of"
        r"|Chief\s+Executive|Chief\s+Financial|Chief\s+Technology|Managing\s+Director)\b",
    ),
    # "Number N" / count claims
    (
        "statistic",
        "numeric count claim",
        r"\b(?:over|more\s+than|at\s+least|nearly|approximately|around|about)\s+\d[\d,]*"
        r"|\b\d[\d,]*\s+(?:billion|million|trillion|thousand|hundred|users|customers|employees|stores|locations)\b",
    ),
]

_COMPILED: list[tuple[str, str, re.Pattern[str]]] = [
    (cat, reason, re.compile(pattern, re.IGNORECASE))
    for cat, reason, pattern in _RAW_PATTERNS
]


def detect_heuristic_highlights(text: str) -> list[dict]:
    """
    Scan *text* for time-sensitive phrases using lightweight regex patterns.

    Returns a list of dicts, each with:
      - start (int): character offset of match start
      - end (int): character offset of match end
      - text (str): matched substring
      - reason (str): human-readable label shown in the chip
      - category (str): category name matching extractor.VALID_CATEGORIES

    Results are sorted by start offset; overlapping matches are deduplicated
    so the first (highest-priority) pattern wins.
    """
    if not text or not text.strip():
        return []

    hits: list[dict] = []
    occupied: list[tuple[int, int]] = []  # (start, end) of accepted spans

    for cat, reason, pattern in _COMPILED:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            # Skip if this span overlaps with an already-accepted span
            if any(s < oe and e > os for os, oe in occupied):
                continue
            occupied.append((s, e))
            hits.append(
                {
                    "start": s,
                    "end": e,
                    "text": m.group(),
                    "reason": reason,
                    "category": cat,
                }
            )

    hits.sort(key=lambda h: h["start"])
    return hits


def render_heuristic_preview_html(highlights: list[dict], limit: int = 12) -> str:
    """
    Render heuristic highlight chips as a compact HTML block.

    Clearly labelled "Heuristic preview — not verified" so users understand
    this is pattern-matching, not live API verification.
    """
    import html as _html

    if not highlights:
        return ""

    shown = highlights[:limit]
    more = len(highlights) - len(shown)

    _CATEGORY_ICONS: dict[str, str] = {
        "price": "💰",
        "version": "🔢",
        "statistic": "📊",
        "leadership": "👤",
        "records": "🏷️",
        "other": "📅",
    }

    chips: list[str] = []
    for h in shown:
        icon = _CATEGORY_ICONS.get(h["category"], "📌")
        txt = _html.escape(h["text"])
        reason = _html.escape(h["reason"])
        chips.append(
            f'<span class="sad-heuristic-chip" title="{reason}">'
            f'<span class="sad-heuristic-icon" aria-hidden="true">{icon}</span>'
            f'<span class="sad-heuristic-text">{txt}</span>'
            f'</span>'
        )

    more_html = (
        f'<span class="sad-heuristic-more">+{more} more</span>' if more > 0 else ""
    )

    return (
        '<div class="sad-heuristic-preview" role="note" '
        'aria-label="Heuristic preview — these phrases may go stale">'
        '<div class="sad-heuristic-label">'
        '<span class="sad-heuristic-label-icon" aria-hidden="true">⚡</span>'
        '<span>Heuristic preview</span>'
        '<span class="sad-heuristic-label-note">not verified — simple pattern matching</span>'
        '</div>'
        f'<div class="sad-heuristic-chips">{"".join(chips)}{more_html}</div>'
        '</div>'
    )
