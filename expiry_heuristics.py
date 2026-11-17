"""
expiry_heuristics.py - Transparent category-based heuristic change cycle and review date estimation.

Rules are stored in `expiry_rules.json`.
Transparently labels all outputs as heuristics, not deterministic predictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

_RULES_FILE = Path(__file__).resolve().parent / "expiry_rules.json"


def load_expiry_rules(path: Path | str | None = None) -> dict[str, Any]:
    """Load the transparent expiry rules dictionary from JSON."""
    p = Path(path) if path else _RULES_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"rules": {}, "metadata": {}}


def get_likely_expiry_info(
    category: str = "other",
    volatility_class: str | None = None,
    check_date: datetime | str | None = None,
    rules_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Compute typical change cycle and estimated review date for a factual claim.

    Returns:
        Dict with keys:
        - volatility_class: canonical class (e.g. 'price')
        - typical_cycle: human-readable cycle (e.g. '3 months', 'none')
        - estimated_review_date: formatted date string (e.g. 'Feb 15, 2027' or 'None (Indefinite)')
        - heuristic_tag: 'Typical change cycle: <volatility_class> (heuristic)'
        - review_label: 'Estimated review: <date> (heuristic, not a prediction)'
        - disclaimer: 'Heuristic estimate, not a prediction'
    """
    rules_data = load_expiry_rules(rules_path)
    rules = rules_data.get("rules", {})

    # Determine matched rule key
    matched_key = "other"
    if volatility_class and str(volatility_class).lower() in rules:
        matched_key = str(volatility_class).lower()
    elif category and str(category).lower() in rules:
        matched_key = str(category).lower()

    rule = rules.get(matched_key, rules.get("other", {
        "volatility_class": matched_key,
        "typical_cycle": "6 months",
        "days_until_review": 180,
    }))

    vol_class = rule.get("volatility_class", matched_key)
    cycle = rule.get("typical_cycle", "6 months")
    days = rule.get("days_until_review")

    # Parse base date
    base_dt: datetime
    if isinstance(check_date, datetime):
        base_dt = check_date
    elif isinstance(check_date, str) and check_date.strip():
        try:
            base_dt = datetime.fromisoformat(check_date.strip())
        except Exception:
            base_dt = datetime.now()
    else:
        base_dt = datetime.now()

    if days is None:
        review_date_str = "None (Stable fact)"
        review_label = "No routine review required (heuristic, not a prediction)"
    else:
        target_dt = base_dt + timedelta(days=int(days))
        review_date_str = target_dt.strftime("%b %d, %Y")
        review_label = f"Estimated review: {review_date_str} (heuristic, not a prediction)"

    heuristic_tag = f"Typical change cycle: {vol_class} (heuristic)"

    return {
        "volatility_class": vol_class,
        "typical_cycle": cycle,
        "days_until_review": days,
        "estimated_review_date": review_date_str,
        "heuristic_tag": heuristic_tag,
        "review_label": review_label,
        "disclaimer": "Heuristic category cycle, not a deterministic prediction.",
    }
