"""
benchmark.py - Evaluation dataset loader, live detector benchmark runner, metrics
calculator, disk cache (`benchmark/results.json`), and README table formatter.

Usage (CLI):
    python benchmark.py [--no-cache] [--dual-judge] [--workers 3] [--update-readme]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from typing import Callable

from extractor import (
    VALID_CATEGORIES,
    extract_claims,
    get_last_gateway_metrics as get_extract_metrics,
)
import llm as _llm_mod
from llm import (
    _get_config_value,
    get_llm_config,
    is_use_bifrost,
    pop_last_gateway_metrics,
    set_simulate_outage,
)
from verifier import (
    PARALLEL_VERIFICATION_WORKERS,
    _make_worker_tavily_client,
    get_tavily_client,
    verify_claim,
)

ROOT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT_DIR / "benchmark"
DATASET_PATH = BENCHMARK_DIR / "dataset.json"
RESULTS_PATH = BENCHMARK_DIR / "results.json"
README_PATH = ROOT_DIR / "README.md"

BENCHMARK_README_START = "<!-- BENCHMARK_RESULTS_START -->"
BENCHMARK_README_END = "<!-- BENCHMARK_RESULTS_END -->"

EXPECTED_BENCHMARK_STATUSES = {"fresh", "outdated"}


def load_dataset(dataset_path: str | Path | None = None) -> list[dict]:
    """
    Load and validate labelled claims from `benchmark/dataset.json`.

    Each entry requires: `claim`, `category`, `expected_status` ('fresh' or 'outdated'),
    and `note`.
    """
    path = Path(dataset_path) if dataset_path is not None else DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path}.")

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Benchmark dataset must be a non-empty JSON array.")

    cleaned: list[dict] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Benchmark dataset entry #{idx} must be a JSON object.")

        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            raise ValueError(f"Benchmark dataset entry #{idx} is missing 'claim'.")

        raw_cat = str(item.get("category", "other")).strip().lower()
        category = raw_cat if raw_cat in VALID_CATEGORIES else "other"

        expected = str(item.get("expected_status", "")).strip().lower()
        if expected not in EXPECTED_BENCHMARK_STATUSES:
            raise ValueError(
                f"Benchmark dataset entry #{idx} has invalid expected_status={expected!r}; "
                "must be 'fresh' or 'outdated'."
            )

        note = str(item.get("note", "")).strip()
        cleaned.append(
            {
                "index": idx,
                "claim": claim_text,
                "category": category,
                "expected_status": expected,
                "note": note,
            }
        )

    return cleaned


def _extract_tokens_and_models(
    verify_result: dict,
    extract_metrics: dict | None = None,
) -> tuple[int, int, int, list[str]]:
    """Aggregate prompt, completion, total tokens and unique model names for one evaluated item."""
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    models: list[str] = []

    if isinstance(extract_metrics, dict):
        ep = int(extract_metrics.get("prompt_tokens") or 0)
        ec = int(extract_metrics.get("completion_tokens") or 0)
        et = int(extract_metrics.get("total_tokens") or (ep + ec))
        prompt_tokens += ep
        completion_tokens += ec
        total_tokens += et
        em = str(extract_metrics.get("model_used") or "").strip()
        if em and em not in models:
            models.append(em)

    judge_tel = [
        entry
        for entry in (verify_result.get("judge_telemetry") or [])
        if isinstance(entry, dict)
    ]
    if judge_tel:
        for jm in judge_tel:
            jp = int(jm.get("prompt_tokens") or 0)
            jc = int(jm.get("completion_tokens") or 0)
            jt = int(jm.get("total_tokens") or (jp + jc))
            prompt_tokens += jp
            completion_tokens += jc
            total_tokens += jt
            m_name = str(jm.get("model_used") or "").strip()
            if m_name and m_name not in models:
                models.append(m_name)
    else:
        gw = verify_result.get("gateway_metrics")
        if isinstance(gw, dict):
            gp = int(gw.get("prompt_tokens") or 0)
            gc = int(gw.get("completion_tokens") or 0)
            gt = int(gw.get("total_tokens") or (gp + gc))
            prompt_tokens += gp
            completion_tokens += gc
            total_tokens += gt
            m_name = str(gw.get("model_used") or "").strip()
            if m_name and m_name not in models:
                models.append(m_name)

    for jm in verify_result.get("judge_models") or []:
        s_jm = str(jm or "").strip()
        if s_jm and s_jm not in models:
            models.append(s_jm)

    single_model = str(verify_result.get("model") or "").strip()
    if single_model and single_model not in models:
        models.append(single_model)

    return prompt_tokens, completion_tokens, total_tokens, models


def compute_benchmark_metrics(
    rows: list[dict],
    total_duration_ms: float | None = None,
) -> dict:
    """
    Compute classification metrics (accuracy, stale precision/recall/F1,
    confusion matrix, per-category breakdown) and efficiency telemetry
    (average latency, total tokens, Tavily credits) across benchmark rows.

    Stale detection treats `outdated` as the positive class:
    - TP: expected='outdated' and predicted='outdated'
    - FP: expected='fresh'    and predicted='outdated'
    - FN: expected='outdated' and predicted!='outdated'
    - TN: expected='fresh'    and predicted!='outdated'
    Strict accuracy counts exact matches (`predicted_status == expected_status`).
    """
    total = len(rows)
    if total == 0:
        return {
            "total_claims": 0,
            "correct": 0,
            "incorrect": 0,
            "accuracy": 0.0,
            "precision_stale": 0.0,
            "recall_stale": 0.0,
            "f1_stale": 0.0,
            "avg_latency_ms": 0.0,
            "total_duration_ms": round(float(total_duration_ms or 0.0), 1),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_tavily_credits": 0.0,
            "error_count": 0,
            "confusion_matrix": {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
                "matrix": {
                    "outdated": {"outdated": 0, "fresh": 0, "other": 0},
                    "fresh": {"outdated": 0, "fresh": 0, "other": 0},
                },
            },
            "by_category": {},
        }

    correct = 0
    tp = fp = fn = tn = 0
    matrix = {
        "outdated": {"outdated": 0, "fresh": 0, "other": 0},
        "fresh": {"outdated": 0, "fresh": 0, "other": 0},
    }

    total_latency_ms = 0.0
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_credits = 0.0
    error_count = 0

    by_category: dict[str, dict] = {}

    for row in rows:
        exp = str(row.get("expected_status", "")).strip().lower()
        pred = str(row.get("predicted_status", "unverifiable")).strip().lower()
        cat = str(row.get("category", "other")).strip().lower()

        is_correct = bool(row.get("correct")) if "correct" in row else (pred == exp)
        if is_correct:
            correct += 1

        if exp == "outdated":
            if pred == "outdated":
                tp += 1
                matrix["outdated"]["outdated"] += 1
            else:
                fn += 1
                if pred == "fresh":
                    matrix["outdated"]["fresh"] += 1
                else:
                    matrix["outdated"]["other"] += 1
        elif exp == "fresh":
            if pred == "outdated":
                fp += 1
                matrix["fresh"]["outdated"] += 1
            else:
                tn += 1
                if pred == "fresh":
                    matrix["fresh"]["fresh"] += 1
                else:
                    matrix["fresh"]["other"] += 1

        total_latency_ms += float(row.get("latency_ms") or 0.0)
        p_tok = int(row.get("prompt_tokens") or 0)
        c_tok = int(row.get("completion_tokens") or 0)
        t_tok = int(row.get("total_tokens") or (p_tok + c_tok))
        total_prompt += p_tok
        total_completion += c_tok
        total_tokens += t_tok
        total_credits += float(row.get("tavily_credits_used") or 0.0)
        if row.get("error"):
            error_count += 1

        cat_bucket = by_category.setdefault(
            cat,
            {
                "total": 0,
                "correct": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            },
        )
        cat_bucket["total"] += 1
        if is_correct:
            cat_bucket["correct"] += 1
        if exp == "outdated" and pred == "outdated":
            cat_bucket["tp"] += 1
        elif exp == "outdated":
            cat_bucket["fn"] += 1
        elif exp == "fresh" and pred == "outdated":
            cat_bucket["fp"] += 1
        else:
            cat_bucket["tn"] += 1

    for cat_stats in by_category.values():
        cat_total = cat_stats["total"]
        cat_stats["accuracy"] = round(cat_stats["correct"] / cat_total, 4) if cat_total else 0.0

    accuracy = round(correct / total, 4)
    precision_stale = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall_stale = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1_stale = (
        round((2.0 * precision_stale * recall_stale) / (precision_stale + recall_stale), 4)
        if (precision_stale + recall_stale) > 0
        else 0.0
    )

    avg_latency_ms = round(total_latency_ms / total, 1)
    resolved_duration_ms = round(
        float(total_duration_ms) if total_duration_ms is not None else total_latency_ms,
        1,
    )

    return {
        "total_claims": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "precision_stale": precision_stale,
        "recall_stale": recall_stale,
        "f1_stale": f1_stale,
        "avg_latency_ms": avg_latency_ms,
        "total_duration_ms": resolved_duration_ms,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "total_tavily_credits": round(total_credits, 2),
        "error_count": error_count,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "matrix": matrix,
        },
        "by_category": by_category,
    }


def load_cached_results(results_path: str | Path | None = None) -> dict | None:
    """
    Load previously cached benchmark results from `benchmark/results.json`.
    Returns None if the cache file does not exist or cannot be parsed cleanly.
    """
    path = Path(results_path) if results_path is not None else RESULTS_PATH
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return None
    return data


def save_cached_results(payload: dict, results_path: str | Path | None = None) -> Path:
    """Persist benchmark results JSON to disk cleanly."""
    path = Path(results_path) if results_path is not None else RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _evaluate_single_benchmark_item(
    entry: dict,
    base_tavily_client,
    *,
    use_cache: bool,
    cache_path: str | None,
    dual_judge: bool,
    use_extractor_query: bool,
    parent_outage_raw,
) -> dict:
    """Evaluate one labelled claim through optional query extraction + live claim verification."""
    pop_last_gateway_metrics()
    if parent_outage_raw is not None:
        set_simulate_outage(parent_outage_raw)
    else:
        _llm_mod._thread_state.simulate_outage = None

    worker_client, should_close = _make_worker_tavily_client(base_tavily_client)
    t0 = time.perf_counter()
    extract_metrics: dict | None = None
    search_query = entry["claim"]
    claim_text = entry["claim"]
    category = entry["category"]
    expected = entry["expected_status"]
    note = entry.get("note", "")

    try:
        if use_extractor_query:
            try:
                extracted_items, _ext_model = extract_claims(claim_text)
                extract_metrics = get_extract_metrics()
                if extracted_items:
                    candidate_q = str(extracted_items[0].get("search_query") or "").strip()
                    if candidate_q:
                        search_query = candidate_q
            except Exception:
                # Fall back to the raw claim string if query extraction fails for one item.
                extract_metrics = get_extract_metrics()

        verify_res = verify_claim(
            {
                "claim": claim_text,
                "category": category,
                "search_query": search_query,
            },
            tavily_client=worker_client,
            use_cache=use_cache,
            cache_path=cache_path,
            dual_judge=dual_judge,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        pred_status = str(verify_res.get("status", "unverifiable")).strip().lower()
        raw_score = verify_res.get("score")
        safe_score = None if (raw_score is None or pred_status == "unverifiable") else int(raw_score)
        p_tok, c_tok, t_tok, row_models = _extract_tokens_and_models(verify_res, extract_metrics)
        credits_used = round(float(verify_res.get("tavily_credits_used") or 0.0), 2)

        compact_sources: list[dict] = []
        for src in (verify_res.get("sources") or [])[:3]:
            if isinstance(src, dict) and src.get("url"):
                compact_sources.append(
                    {
                        "title": str(src.get("title") or "").strip(),
                        "url": str(src.get("url") or "").strip(),
                        "source_type": str(src.get("source_type") or "").strip(),
                    }
                )

        return {
            "index": int(entry["index"]),
            "claim": claim_text,
            "category": category,
            "expected_status": expected,
            "predicted_status": pred_status,
            "correct": pred_status == expected,
            "score": safe_score,
            "confidence": str(verify_res.get("confidence", "low")).strip().lower(),
            "note": note,
            "current_value": verify_res.get("current_value"),
            "current_info": str(verify_res.get("current_info") or "").strip(),
            "reason": str(verify_res.get("reason") or "").strip(),
            "search_query": str(verify_res.get("search_query") or search_query),
            "cache_hit": bool(verify_res.get("cache_hit", False)),
            "model": verify_res.get("model") or (row_models[-1] if row_models else None),
            "models": row_models,
            "sources": compact_sources,
            "latency_ms": latency_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": t_tok,
            "tavily_credits_used": credits_used,
            "error": verify_res.get("error"),
        }
    finally:
        if should_close:
            try:
                worker_client.session.close()
            except Exception:
                pass
        pop_last_gateway_metrics()
        _llm_mod._thread_state.simulate_outage = None


def run_benchmark(
    dataset_path: str | Path | None = None,
    results_path: str | Path | None = None,
    *,
    progress_callback: Callable[[int, int, dict], None] | None = None,
    use_cache: bool = True,
    cache_path: str | None = None,
    max_workers: int = PARALLEL_VERIFICATION_WORKERS,
    use_extractor_query: bool = True,
    dual_judge: bool = False,
    save_results: bool = True,
    raise_on_all_failed: bool = True,
) -> dict:
    """
    Run the stale-claim detector on all labelled claims in `benchmark/dataset.json`
    using parallel worker threads, compute accuracy/precision/recall/confusion-matrix
    and resource metrics, and cache the resulting payload in `benchmark/results.json`.
    """
    entries = load_dataset(dataset_path)
    total = len(entries)

    # Validate Tavily + LLM credentials upfront on the calling thread before fanning out.
    tavily_client = get_tavily_client()
    if use_extractor_query:
        get_llm_config(task="extract")
    else:
        get_llm_config(task="judge")

    parent_outage_raw = getattr(_llm_mod._thread_state, "simulate_outage", None)
    worker_count = max(1, min(total, int(max_workers or PARALLEL_VERIFICATION_WORKERS)))

    bench_t0 = time.perf_counter()
    ordered_rows: list[dict | None] = [None] * total
    completed = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_pos = {
            executor.submit(
                _evaluate_single_benchmark_item,
                entry,
                tavily_client,
                use_cache=use_cache,
                cache_path=cache_path,
                dual_judge=dual_judge,
                use_extractor_query=use_extractor_query,
                parent_outage_raw=parent_outage_raw,
            ): pos
            for pos, entry in enumerate(entries)
        }
        for future in as_completed(future_to_pos):
            pos = future_to_pos[future]
            row = future.result()
            ordered_rows[pos] = row
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total, row)

    total_duration_ms = round((time.perf_counter() - bench_t0) * 1000.0, 1)
    rows = [r for r in ordered_rows if r is not None]

    # Guard against overwriting an existing cached benchmark if every claim failed before reaching an LLM.
    if raise_on_all_failed and rows:
        total_tok_check = sum(int(r.get("total_tokens") or 0) for r in rows)
        if all(bool(r.get("error")) for r in rows) and total_tok_check == 0:
            first_err = next((str(r["error"]) for r in rows if r.get("error")), "All benchmark checks failed.")
            raise RuntimeError(first_err)

    metrics = compute_benchmark_metrics(rows, total_duration_ms=total_duration_ms)

    models_used: list[str] = []
    for row in rows:
        for m in row.get("models") or []:
            if m and m not in models_used:
                models_used.append(m)

    routing_mode = "bifrost" if is_use_bifrost() else "direct_gemini"
    payload = {
        "app": "Stale-Answer Detector",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "routing_mode": routing_mode,
        "models_used": models_used,
        "use_tavily_cache": bool(use_cache),
        "dual_judge": bool(dual_judge),
        "total_claims": len(rows),
        "metrics": metrics,
        "results": rows,
    }

    if save_results:
        save_cached_results(payload, results_path=results_path)

    return payload


def format_results_markdown(payload: dict) -> str:
    """Format a benchmark results payload into crisp Markdown summary + confusion matrix + per-claim tables."""
    metrics = payload.get("metrics") or {}
    rows = payload.get("results") or []
    cm = metrics.get("confusion_matrix") or {}
    cm_mat = cm.get("matrix") or {}

    acc_pct = f"{float(metrics.get('accuracy', 0.0)) * 100:.1f}%"
    prec_pct = f"{float(metrics.get('precision_stale', 0.0)) * 100:.1f}%"
    rec_pct = f"{float(metrics.get('recall_stale', 0.0)) * 100:.1f}%"
    f1_pct = f"{float(metrics.get('f1_stale', 0.0)) * 100:.1f}%"
    avg_lat = f"{float(metrics.get('avg_latency_ms', 0.0)):,.1f} ms"
    tot_tok = f"{int(metrics.get('total_tokens', 0)):,}"
    tot_cred = f"{float(metrics.get('total_tavily_credits', 0.0)):g}"
    correct_cnt = int(metrics.get("correct", 0))
    total_cnt = int(metrics.get("total_claims", len(rows)))

    models_str = ", ".join(f"`{m}`" for m in (payload.get("models_used") or [])) or "N/A"
    lines: list[str] = [
        f"*Last run:* `{payload.get('generated_at', 'N/A')}` | "
        f"*Mode:* `{payload.get('routing_mode', 'N/A')}` | "
        f"*Models:* {models_str}",
        "",
        "### Benchmark Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Accuracy** | **{acc_pct}** ({correct_cnt}/{total_cnt}) |",
        f"| **Precision (stale)** | {prec_pct} (TP={cm.get('tp', 0)}, FP={cm.get('fp', 0)}) |",
        f"| **Recall (stale)** | {rec_pct} (TP={cm.get('tp', 0)}, FN={cm.get('fn', 0)}) |",
        f"| **F1 (stale)** | {f1_pct} |",
        f"| **Avg Latency / Claim** | {avg_lat} |",
        f"| **Total Tokens Used** | {tot_tok} |",
        f"| **Total Tavily Credits** | {tot_cred} |",
        "",
        "### Confusion Matrix (Expected vs. Predicted)",
        "",
        "| Expected \\ Predicted | Outdated (Stale) | Fresh | Uncertain / Unverifiable |",
        "| :--- | :---: | :---: | :---: |",
        (
            f"| **Outdated (Stale)** | {cm_mat.get('outdated', {}).get('outdated', cm.get('tp', 0))} "
            f"| {cm_mat.get('outdated', {}).get('fresh', 0)} "
            f"| {cm_mat.get('outdated', {}).get('other', 0)} |"
        ),
        (
            f"| **Fresh** | {cm_mat.get('fresh', {}).get('outdated', cm.get('fp', 0))} "
            f"| {cm_mat.get('fresh', {}).get('fresh', cm.get('tn', 0))} "
            f"| {cm_mat.get('fresh', {}).get('other', 0)} |"
        ),
        "",
        f"### Per-Claim Results ({total_cnt} Claims)",
        "",
        "| # | Mark | Category | Claim | Expected | Predicted | Score | Latency | Tokens | Credits |",
        "| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for row in rows:
        idx = row.get("index", "")
        mark = "Correct" if row.get("correct") else "Incorrect"
        mark_sym = f"✓ {mark}" if row.get("correct") else f"✗ {mark}"
        cat = str(row.get("category", "")).replace("|", "\\|")
        claim_clean = str(row.get("claim", "")).replace("\n", " ").replace("|", "\\|")
        exp = str(row.get("expected_status", ""))
        pred = str(row.get("predicted_status", ""))
        sc = row.get("score")
        sc_str = "N/A" if sc is None else f"{sc}/100"
        lat_str = f"{float(row.get('latency_ms') or 0.0):,.0f} ms"
        tok_str = f"{int(row.get('total_tokens') or 0):,}"
        cred_str = f"{float(row.get('tavily_credits_used') or 0.0):g}"
        lines.append(
            f"| {idx} | {mark_sym} | `{cat}` | {claim_clean} | `{exp}` | `{pred}` | "
            f"{sc_str} | {lat_str} | {tok_str} | {cred_str} |"
        )

    return "\n".join(lines)


def update_readme_benchmark_section(
    payload: dict,
    readme_path: str | Path | None = None,
) -> bool:
    """Replace the block between BENCHMARK_RESULTS markers inside README.md."""
    path = Path(readme_path) if readme_path is not None else README_PATH
    if not path.exists():
        return False

    content = path.read_text(encoding="utf-8")
    if BENCHMARK_README_START not in content or BENCHMARK_README_END not in content:
        return False

    before, rest = content.split(BENCHMARK_README_START, 1)
    _, after = rest.split(BENCHMARK_README_END, 1)
    rendered = format_results_markdown(payload)
    updated = f"{before}{BENCHMARK_README_START}\n{rendered}\n{BENCHMARK_README_END}{after}"
    path.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the 20-claim Stale-Answer Detector benchmark.")
    parser.add_argument("--no-cache", action="store_true", help="Disable the 6h Tavily SQLite cache.")
    parser.add_argument("--dual-judge", action="store_true", help="Run benchmark with dual-judge consensus.")
    parser.add_argument("--workers", type=int, default=PARALLEL_VERIFICATION_WORKERS, help="Parallel worker count.")
    parser.add_argument("--no-extract-query", action="store_true", help="Skip extract_claims query refinement.")
    parser.add_argument("--update-readme", action="store_true", help="Write formatted results table into README.md.")
    args = parser.parse_args(argv)

    print_lock = threading.Lock()

    def _cli_progress(done: int, total: int, row: dict) -> None:
        mark = "OK  " if row.get("correct") else "MISS"
        with print_lock:
            print(
                f"[{done:02d}/{total:02d}] {mark} | expected={row['expected_status']:8s} | "
                f"predicted={row['predicted_status']:12s} | {row['latency_ms']:7.1f} ms | "
                f"{row['claim'][:58]}"
            )

    payload = run_benchmark(
        use_cache=not args.no_cache,
        dual_judge=args.dual_judge,
        max_workers=args.workers,
        use_extractor_query=not args.no_extract_query,
        progress_callback=_cli_progress,
    )
    m = payload["metrics"]
    print(
        f"\nSaved cached run to {RESULTS_PATH}\n"
        f"Accuracy: {m['accuracy'] * 100:.1f}% ({m['correct']}/{m['total_claims']}) | "
        f"Precision (stale): {m['precision_stale'] * 100:.1f}% | "
        f"Recall (stale): {m['recall_stale'] * 100:.1f}% | "
        f"Avg latency: {m['avg_latency_ms']:.1f} ms | "
        f"Tokens: {m['total_tokens']:,} | Tavily credits: {m['total_tavily_credits']:g}"
    )

    if args.update_readme:
        if update_readme_benchmark_section(payload):
            print(f"Updated benchmark results table in {README_PATH}")
        else:
            print(f"Note: benchmark markers not found in {README_PATH}; skipped README update.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
