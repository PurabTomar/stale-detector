"""
calibration.py - Score calibration calculation and analysis for Benchmark tab.

Plots predicted score (binned: 0-25, 26-50, 51-75, 76-100) against empirical accuracy (share correct).
Transparently reports 'not enough data' if fewer than 5 evaluated claims exist.
"""

from __future__ import annotations

from typing import Any

from benchmark import load_cached_results


def compute_calibration(results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Compute calibration curve bins and plain-language interpretation.

    Returns:
        Dict with keys:
        - status: 'ok' or 'not_enough_data'
        - bins: list of bin dicts with 'bin_label', 'n', 'correct', 'accuracy_pct'
        - reading: plain-language explanation of calibration results
        - total_evaluated: count of scorable claims analyzed
    """
    items = results
    if items is None:
        cached = load_cached_results()
        if isinstance(cached, dict):
            items = cached.get("results") or []

    if not items or len(items) < 5:
        return {
            "status": "not_enough_data",
            "message": "Not enough benchmark data to compute calibration curve (minimum 5 evaluated claims required).",
            "bins": [],
            "reading": "Not enough benchmark evaluation data available to calculate empirical calibration.",
            "total_evaluated": len(items) if items else 0,
        }

    # Bins definition: 4 quartiles of freshness score (0-25, 26-50, 51-75, 76-100)
    bin_specs = [
        {"label": "0–25", "min": 0, "max": 25, "desc": "Low Freshness"},
        {"label": "26–50", "min": 26, "max": 50, "desc": "Moderate-Low"},
        {"label": "51–75", "min": 51, "max": 75, "desc": "Moderate-High"},
        {"label": "76–100", "min": 76, "max": 100, "desc": "High Freshness"},
    ]

    bins_data: list[dict[str, Any]] = []
    scorable_count = 0

    for b in bin_specs:
        b_items = []
        for it in items:
            raw_sc = it.get("score")
            if raw_sc is None:
                continue
            try:
                sc = float(raw_sc)
            except (ValueError, TypeError):
                continue
            if b["min"] <= sc <= b["max"]:
                b_items.append(it)

        n = len(b_items)
        scorable_count += n
        correct_cnt = sum(1 for it in b_items if bool(it.get("correct")))
        acc = round((correct_cnt / n) * 100.0, 1) if n > 0 else 0.0

        bins_data.append({
            "bin": b["label"],
            "description": b["desc"],
            "min_score": b["min"],
            "max_score": b["max"],
            "n": n,
            "correct": correct_cnt,
            "accuracy_pct": acc,
        })

    if scorable_count < 5:
        return {
            "status": "not_enough_data",
            "message": "Not enough scored benchmark items to compute calibration curve.",
            "bins": [],
            "reading": "Not enough data",
            "total_evaluated": scorable_count,
        }

    # Generate plain-language reading
    top_bin = bins_data[-1]  # 76-100
    bottom_bin = bins_data[0]  # 0-25

    readings = []
    if top_bin["n"] > 0:
        readings.append(
            f"Scores above 75 were right {top_bin['accuracy_pct']}% of the time (n={top_bin['n']})."
        )
    if bottom_bin["n"] > 0:
        readings.append(
            f"Claims scored 0–25 had {bottom_bin['accuracy_pct']}% classification accuracy (n={bottom_bin['n']})."
        )

    reading_text = " ".join(readings) if readings else "Calibration data computed successfully."

    return {
        "status": "ok",
        "bins": bins_data,
        "reading": reading_text,
        "total_evaluated": scorable_count,
    }
