"""
scripts/generate_showdown.py - Record real Model Showdown results to evidence/showdown_results.json.

Runs 3 questions against 2-3 REAL configured models through the active LLM routing
(Bifrost gateway when USE_BIFROST=true, or Direct Gemini when USE_BIFROST=false),
verifies every model's answer via the real Tavily + LLM pipeline, and saves:
- timestamp (`recorded_at`, `recorded_date`)
- routing mode (`routing_mode`)
- configured models (`models`)
- per-question raw responses, measured latencies, returned tokens, claims, and scores.

Usage:
    python scripts/generate_showdown.py
    python scripts/generate_showdown.py --output evidence/showdown_results.json --models model1,model2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Ensure repository root is importable when run as `python scripts/generate_showdown.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm import is_use_bifrost
from showdown_data import (
    DEFAULT_SHOWDOWN_QUESTIONS,
    SHOWDOWN_RESULTS_FILE,
    get_configured_showdown_models,
    run_live_showdown,
)


def generate_showdown_file(
    output_path: Path | str = SHOWDOWN_RESULTS_FILE,
    models: list[str] | None = None,
    questions: list[dict[str, str]] | None = None,
) -> dict:
    """Run all showdown questions against real configured models and persist JSON."""
    selected_models = list(models) if models else get_configured_showdown_models()
    if len(selected_models) < 2:
        raise RuntimeError(
            "At least 2 configured models are required to generate a Model Showdown. "
            "Configure LLM_MODEL, LLM_MODEL_EXTRACT, LLM_MODEL_JUDGE, or LLM_MODEL_FALLBACKS."
        )

    q_list = list(questions) if questions else list(DEFAULT_SHOWDOWN_QUESTIONS)
    now_utc = datetime.now(timezone.utc)
    routing_mode = "bifrost" if is_use_bifrost() else "direct_gemini"

    showdowns: dict[str, dict] = {}
    for q_item in q_list:
        qid = q_item["id"]
        print(f"[generate_showdown] Running '{qid}': {q_item['question']}")
        res = run_live_showdown(q_item["question"], models=selected_models)
        res["id"] = qid
        res["title"] = q_item.get("title") or res["title"]
        res["description"] = q_item.get("description") or res["description"]
        showdowns[qid] = res

    payload = {
        "recorded_at": now_utc.isoformat(),
        "recorded_date": now_utc.strftime("%b %d, %Y"),
        "routing_mode": routing_mode,
        "models": selected_models,
        "showdowns": showdowns,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[generate_showdown] Saved {len(showdowns)} real showdowns to {out}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate real recorded Model Showdown results.")
    parser.add_argument(
        "--output",
        default=str(SHOWDOWN_RESULTS_FILE),
        help="Destination JSON file (default: evidence/showdown_results.json)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated list of 2-3 configured model IDs to compare.",
    )
    args = parser.parse_args()
    models_arg = [m.strip() for m in args.models.split(",") if m.strip()] or None
    generate_showdown_file(output_path=args.output, models=models_arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
