"""
batch_audit.py - Multi-item batch audit processor for Stale-Answer Detector.

Capabilities:
- Parse batch input from pasted text (split on blank lines) or uploaded .txt, .md, or .csv files.
- Enforce strict safety constraints: max 20 items, max 200 KB file upload.
- Cap claims per item and total claims across batch to protect session quotas.
- Compute concise item-level summary metrics (overall score, status counts, worst claim).
- Export batch results as standardized CSV and JSON formats.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

# Safety limits for batch audits
MAX_BATCH_ITEMS = 20
MAX_BATCH_FILE_SIZE_BYTES = 200 * 1024  # 200 KB
DEFAULT_BATCH_CLAIMS_PER_ITEM = 3
MAX_BATCH_TOTAL_CLAIMS = 30


def parse_batch_input(
    raw_text: str = "",
    file_bytes: bytes | None = None,
    filename: str = "",
) -> tuple[list[str], str | None]:
    """
    Parse multiple answers from text or uploaded file bytes.

    Accepts:
    - .csv file: reads 'answer', 'text', 'content', 'response' column, or first column.
    - .txt / .md file: splits on double newline / blank line.
    - raw_text: splits on blank line.

    Enforces:
    - Max 200 KB file size.
    - Max 20 items (truncates to 20 with clean slicing).

    Returns:
        `(items, error_message)`
    """
    # 1. Enforce file size check if file bytes provided
    if file_bytes is not None:
        if len(file_bytes) > MAX_BATCH_FILE_SIZE_BYTES:
            return (
                [],
                f"File size exceeds maximum limit of 200 KB (received {len(file_bytes) / 1024:.1f} KB). "
                f"Please upload a smaller file.",
            )
        try:
            decoded_text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                decoded_text = file_bytes.decode("latin-1")
            except Exception:
                return [], "Could not decode file content. Please upload a valid UTF-8 text, markdown, or CSV file."
    else:
        decoded_text = raw_text or ""

    clean_content = decoded_text.strip()
    if not clean_content:
        return [], "No content provided to batch audit. Please upload a file or paste answers."

    items: list[str] = []
    lower_filename = filename.lower()

    # 2. Parse based on file type
    if lower_filename.endswith(".csv"):
        try:
            reader = csv.reader(io.StringIO(clean_content))
            rows = [row for row in reader if row and any(c.strip() for c in row)]
            if not rows:
                return [], "The uploaded CSV file is empty."

            # Check if first row has a header like 'answer', 'text', 'content'
            first_row_lower = [str(c).strip().lower() for c in rows[0]]
            target_col = 0
            has_header = False
            for cand in ("answer", "text", "content", "response", "claim"):
                if cand in first_row_lower:
                    target_col = first_row_lower.index(cand)
                    has_header = True
                    break

            data_rows = rows[1:] if has_header else rows
            for row in data_rows:
                if len(row) > target_col:
                    val = str(row[target_col]).strip()
                    if val:
                        items.append(val)
        except Exception as exc:
            return [], f"Failed to parse CSV file: {exc}"
    else:
        # Plain text or Markdown: split by blank line (one or more empty lines)
        splits = re.split(r"\n\s*\n+", clean_content)
        for chunk in splits:
            val = chunk.strip()
            if val:
                items.append(val)

    if not items:
        return [], "No non-empty answers could be parsed from the provided input."

    # 3. Truncate to MAX_BATCH_ITEMS
    if len(items) > MAX_BATCH_ITEMS:
        items = items[:MAX_BATCH_ITEMS]

    return items, None


def summarize_batch_item(
    index: int,
    answer_text: str,
    results: list[dict],
    overall_score: int | float | None = None,
    extract_model: str | None = None,
    source_origin: str = "live",
) -> dict[str, Any]:
    """
    Build a structured summary row for one audited batch item.
    """
    clean_text = " ".join(answer_text.split()).strip()
    snippet = clean_text[:75] + ("..." if len(clean_text) > 75 else "")

    fresh_cnt = sum(1 for r in results if r.get("status") == "fresh")
    outdated_cnt = sum(1 for r in results if r.get("status") in {"outdated", "incorrect"})
    unverifiable_cnt = sum(1 for r in results if r.get("status") in {"unverifiable", "uncertain", "partially_correct"})

    # Compute or format overall score
    if overall_score is not None:
        try:
            eff_score = int(round(float(overall_score)))
        except (TypeError, ValueError):
            eff_score = 0
    else:
        if results:
            scores = [r.get("score") for r in results if r.get("score") is not None]
            eff_score = int(round(sum(scores) / len(scores))) if scores else 0
        else:
            eff_score = 100  # No claims = 100 default

    # Identify worst claim
    worst_claim_text = "None"
    worst_status = "none"
    worst_score = 100
    if results:
        # Sort so that outdated/incorrect comes first, then lowest score
        sorted_results = sorted(
            results,
            key=lambda r: (
                0 if r.get("status") in {"outdated", "incorrect"} else 1,
                r.get("score", 100) if r.get("score") is not None else 100,
            ),
        )
        worst_item = sorted_results[0]
        worst_claim_text = str(worst_item.get("claim", ""))[:70]
        worst_status = str(worst_item.get("status", "unknown"))
        worst_score = worst_item.get("score", 100)

    # Determine source label
    source_label = "Live"
    if source_origin == "cached" or any(r.get("cache_hit") for r in results):
        source_label = "Cached"
    elif source_origin == "saved" or any(r.get("source_origin") == "saved" for r in results):
        source_label = "Saved"

    return {
        "index": index,
        "snippet": snippet,
        "overall_score": eff_score,
        "fresh_count": fresh_cnt,
        "outdated_count": outdated_cnt,
        "unverifiable_count": unverifiable_cnt,
        "total_claims": len(results),
        "worst_claim": worst_claim_text,
        "worst_status": worst_status,
        "worst_score": worst_score,
        "source_label": source_label,
        "extract_model": extract_model or "gemini-2.5-flash",
    }


def export_batch_summary_csv(summary_rows: list[dict[str, Any]]) -> str:
    """Export batch summary table to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Item #",
        "Answer Snippet",
        "Freshness Score",
        "Fresh Claims",
        "Outdated Claims",
        "Unverifiable Claims",
        "Total Claims",
        "Worst Claim",
        "Worst Claim Status",
        "Source Origin",
    ])
    for row in summary_rows:
        writer.writerow([
            row.get("index", 1),
            row.get("snippet", ""),
            row.get("overall_score", 0),
            row.get("fresh_count", 0),
            row.get("outdated_count", 0),
            row.get("unverifiable_count", 0),
            row.get("total_claims", 0),
            row.get("worst_claim", ""),
            row.get("worst_status", ""),
            row.get("source_label", "Live"),
        ])
    return output.getvalue()


def export_batch_summary_json(
    summary_rows: list[dict[str, Any]],
    batch_items_data: list[dict[str, Any]] | None = None,
) -> str:
    """Export batch summary and item claims to formatted JSON string."""
    payload = {
        "batch_summary": summary_rows,
        "total_items": len(summary_rows),
        "items_data": batch_items_data or [],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
