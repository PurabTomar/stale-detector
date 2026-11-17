"""
trust_badge.py - SVG Trust Badge generator and Markdown snippet generator.

Generates self-contained SVG freshness badges ('Checked <date>: N fresh, M stale')
with download capability, copy-markdown snippet, and explicit snapshot notice.
"""

from __future__ import annotations

import html
import re


def generate_trust_badge_svg(
    fresh_count: int = 0,
    stale_count: int = 0,
    check_date: str = "",
) -> str:
    """
    Generate an SVG trust badge.

    Format:
    [ Shield Checked <date> | N fresh, M stale ]
    """
    date_str = html.escape(str(check_date or "Nov 2026").strip())
    label_left = f"Checked {date_str}"
    label_right = f"{fresh_count} fresh, {stale_count} stale"

    # Color selection for the right status side
    if stale_count == 0 and fresh_count > 0:
        bg_right = "#059669"
        accent_right = "#10B981"
    elif stale_count > 0 and fresh_count > 0:
        bg_right = "#D97706"
        accent_right = "#F59E0B"
    elif stale_count > 0 and fresh_count == 0:
        bg_right = "#DC2626"
        accent_right = "#EF4444"
    else:
        bg_right = "#475569"
        accent_right = "#64748B"

    # Approximate width based on character count
    left_chars = len(label_left)
    right_chars = len(label_right)
    width_left = max(110, 30 + left_chars * 7)
    width_right = max(110, 20 + right_chars * 7)
    total_width = width_left + width_right

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="28" viewBox="0 0 {total_width} 28" role="img" aria-label="Freshness: {label_right}">
  <title>Freshness Audit: {label_right} on {date_str}</title>
  <defs>
    <linearGradient id="grad_left" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1E293B" />
      <stop offset="100%" stop-color="#0F172A" />
    </linearGradient>
    <linearGradient id="grad_right" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{accent_right}" />
      <stop offset="100%" stop-color="{bg_right}" />
    </linearGradient>
    <clipPath id="rounded">
      <rect width="{total_width}" height="28" rx="6" fill="#fff" />
    </clipPath>
  </defs>
  <g clip-path="url(#rounded)">
    <!-- Left Background -->
    <rect x="0" y="0" width="{width_left}" height="28" fill="url(#grad_left)" />
    <!-- Right Background -->
    <rect x="{width_left}" y="0" width="{width_right}" height="28" fill="url(#grad_right)" />
    <!-- Subtle divider line -->
    <line x1="{width_left}" y1="0" x2="{width_left}" y2="28" stroke="rgba(255,255,255,0.15)" stroke-width="1" />
  </g>
  <!-- Shield Icon -->
  <g transform="translate(8, 6) scale(0.65)" fill="none" stroke="#818CF8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    <path d="m9 12 2 2 4-4" stroke="#A5B4FC"/>
  </g>
  <!-- Text Labels -->
  <g fill="#F8FAFC" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif" font-size="11" font-weight="600" text-rendering="geometricPrecision">
    <text x="{30 + (width_left - 30) / 2}" y="18" text-anchor="middle" fill="#E2E8F0">{label_left}</text>
    <text x="{width_left + width_right / 2}" y="18" text-anchor="middle" fill="#FFFFFF">{label_right}</text>
  </g>
</svg>"""
    return svg.strip()


def generate_trust_badge_markdown(
    fresh_count: int = 0,
    stale_count: int = 0,
    check_date: str = "",
    badge_path: str = "freshness-badge.svg",
) -> str:
    """Generate Markdown embed snippet for the trust badge."""
    date_clean = check_date or "Nov 2026"
    return (
        f'[![Freshness Audit: {fresh_count} fresh, {stale_count} stale]({badge_path})]'
        f'(# "Audited with Stale-Answer Detector on {date_clean}. Note: Snapshot in time.")'
    )


def get_trust_badge_snapshot_notice(check_date: str = "") -> str:
    """Return explicit snapshot in time disclaimer."""
    date_clean = check_date or "the check date"
    return (
        f"⚠️ Snapshot Notice: This badge reflects verified freshness as of {date_clean}. "
        f"Live web facts, product versions, and prices change over time; "
        f"always run a fresh audit for mission-critical applications."
    )
