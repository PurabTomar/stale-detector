"""
source_credibility.py - Transparent rule-based source credibility scoring.

Maps source types (Primary, News, Blog, Forum) to credibility scores, tier badges,
visual progress indicators, and confidence contributions based on `source_credibility_rules.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_RULES_FILE = Path(__file__).resolve().parent / "source_credibility_rules.json"


def load_credibility_rules(path: Path | str | None = None) -> dict[str, Any]:
    """Load credibility rules dictionary from JSON file."""
    p = Path(path) if path else _RULES_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"rules": {}}


def get_source_credibility(
    source_type: str | None = None,
    url: str | None = None,
    rules_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    Get credibility score, tier label, and confidence rule contribution for a source.

    Returns:
        Dict with keys:
        - source_type: normalized canonical type ('Primary', 'News', 'Blog', 'Forum')
        - credibility_score: integer 0-100 (95, 80, 50, 30)
        - tier_label: human-readable authority tier
        - color: hex color code for UI badge/bar
        - rule_contribution: transparent statement explaining confidence impact
    """
    rules_data = load_credibility_rules(rules_path)
    rules = rules_data.get("rules", {})

    raw_type = str(source_type or "").strip().lower()
    matched_key = "news"
    if "primary" in raw_type or "official" in raw_type:
        matched_key = "primary"
    elif "news" in raw_type or "press" in raw_type:
        matched_key = "news"
    elif "blog" in raw_type:
        matched_key = "blog"
    elif "forum" in raw_type or "reddit" in raw_type or "community" in raw_type:
        matched_key = "forum"

    rule = rules.get(matched_key, {
        "source_type": matched_key.title(),
        "credibility_score": 70,
        "tier_label": "General Web Source",
        "color": "#94A3B8",
        "rule_contribution": "+10% standard web confidence contribution",
    })

    return {
        "source_type": rule.get("source_type", matched_key.title()),
        "credibility_score": int(rule.get("credibility_score", 70)),
        "tier_label": rule.get("tier_label", "Web Source"),
        "color": rule.get("color", "#94A3B8"),
        "rule_contribution": rule.get("rule_contribution", ""),
    }


def render_credibility_bar_html(cred_data: dict[str, Any]) -> str:
    """Render a compact glassmorphic credibility bar with rule contribution."""
    score = cred_data.get("credibility_score", 70)
    color = cred_data.get("color", "#10B981")
    tier = cred_data.get("tier_label", "Authority")
    contrib = cred_data.get("rule_contribution", "")

    return f"""
    <div style="margin-top: 6px; padding: 6px 10px; background: rgba(15, 23, 42, 0.4); border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-size: 0.75rem;">
            <span style="color: #F8FAFC; font-weight: 600;">{tier}</span>
            <span style="color: {color}; font-weight: 700;">{score}% Credibility</span>
        </div>
        <div style="height: 5px; width: 100%; background: rgba(51, 65, 85, 0.5); border-radius: 3px; overflow: hidden; margin-bottom: 4px;">
            <div style="height: 100%; width: {score}%; background: {color}; border-radius: 3px;"></div>
        </div>
        <div style="font-size: 0.72rem; color: #94A3B8; line-height: 1.2;">
            <strong>Rule:</strong> {contrib}
        </div>
    </div>
    """
