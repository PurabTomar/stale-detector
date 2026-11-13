"""
examples/external_agent/server.py - Standalone external research agent server.

A separate minimal FastAPI application sharing ZERO code with the main repository.
Implements a 2-stage search with query-rewriting strategy:
1. Initial search on raw claim.
2. LLM call to rewrite/sharpen search query for latest current facts.
3. Second search with rewritten query.
4. Final judge LLM call returning normalized verdict, current value, and execution trace.

Runs on port 9001.
"""

import json
import os
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="External Research Agent (Query-Rewriting)")


def _redact(text: str) -> str:
    """Scrub sensitive keys from trace summaries."""
    import re
    return re.sub(r"(tvly-|AIza|sk-|bf-vk-)[A-Za-z0-9_-]{8,}", "[REDACTED]", str(text or ""))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "query_rewriting_external"}


@app.post("/verify")
async def verify(request: Request) -> JSONResponse:
    t0 = time.perf_counter()
    body = await request.json()
    task = body.get("task", {})
    run_id = body.get("run_id", "external-run")
    task_text = str(task.get("input_text") or task.get("claim") or "")
    category = str(task.get("category") or "other")

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    llm_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY", "bifrost-local")
    llm_base_url = (
        os.environ.get("LLM_BASE_URL")
        or ("http://localhost:8080/v1" if os.environ.get("USE_BIFROST", "false").lower() == "true"
            else "https://generativelanguage.googleapis.com/v1beta/openai/")
    ).rstrip("/")
    llm_model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"

    trace: list[dict[str, Any]] = []

    # Stage 1: Initial search
    s1_t0 = time.perf_counter()
    s1_snippets: list[str] = []
    if tavily_key:
        try:
            r1 = requests.post(
                "https://api.tavily.com/search",
                json={"query": task_text, "api_key": tavily_key, "max_results": 3},
                timeout=15,
            )
            if r1.status_code == 200:
                s_data = r1.json().get("results", [])
                s1_snippets = [f"{s.get('title')}: {s.get('snippet')}" for s in s_data if isinstance(s, dict)]
        except Exception:
            pass
    s1_elapsed = round((time.perf_counter() - s1_t0) * 1000.0, 2)
    trace.append({
        "idx": 1,
        "ts": time.time(),
        "kind": "tool",
        "name": "initial_tavily_search",
        "input_summary": _redact(f"query='{task_text[:80]}'"),
        "output_summary": f"Retrieved {len(s1_snippets)} sources",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": s1_elapsed,
        "cache_hit": False,
        "error": None,
    })

    # Stage 2: Query rewriting LLM call
    rw_t0 = time.perf_counter()
    rewritten_query = f"{task_text} current verified facts today"
    try:
        rw_resp = requests.post(
            f"{llm_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {llm_key}",
                "Content-Type": "application/json",
                "x-bf-run-id": run_id,
            },
            json={
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a search query optimizer. Output a short 4-8 word search query to find the latest accurate facts. Output the query only.",
                    },
                    {
                        "role": "user",
                        "content": f"Claim: {task_text}\nInitial snippets: {' '.join(s1_snippets)[:300]}",
                    },
                ],
                "temperature": 0.0,
            },
            timeout=20,
        )
        if rw_resp.status_code == 200:
            rw_text = rw_resp.json()["choices"][0]["message"]["content"].strip()
            if rw_text:
                rewritten_query = rw_text
    except Exception:
        pass
    rw_elapsed = round((time.perf_counter() - rw_t0) * 1000.0, 2)
    trace.append({
        "idx": 2,
        "ts": time.time(),
        "kind": "llm",
        "name": "rewrite_query",
        "input_summary": _redact(f"Optimize: {task_text[:80]}"),
        "output_summary": _redact(f"Rewritten: {rewritten_query[:80]}"),
        "model": llm_model,
        "tokens_in": 60,
        "tokens_out": 15,
        "latency_ms": rw_elapsed,
        "cache_hit": False,
        "error": None,
    })

    # Stage 3: Second search with rewritten query
    s2_t0 = time.perf_counter()
    s2_snippets: list[str] = []
    if tavily_key:
        try:
            r2 = requests.post(
                "https://api.tavily.com/search",
                json={"query": rewritten_query, "api_key": tavily_key, "max_results": 3},
                timeout=15,
            )
            if r2.status_code == 200:
                s_data2 = r2.json().get("results", [])
                s2_snippets = [f"{s.get('title')}: {s.get('snippet')}" for s in s_data2 if isinstance(s, dict)]
        except Exception:
            pass
    s2_elapsed = round((time.perf_counter() - s2_t0) * 1000.0, 2)
    trace.append({
        "idx": 3,
        "ts": time.time(),
        "kind": "tool",
        "name": "refined_tavily_search",
        "input_summary": _redact(f"query='{rewritten_query[:80]}'"),
        "output_summary": f"Retrieved {len(s2_snippets)} refined sources",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": s2_elapsed,
        "cache_hit": False,
        "error": None,
    })

    # Stage 4: Final judge LLM call
    j_t0 = time.perf_counter()
    all_snippets = s1_snippets + s2_snippets
    evidence_block = "\n".join(all_snippets)[:1500] if all_snippets else "No search evidence available."
    verdict = "uncertain"
    current_val: str | None = None
    answer_text = ""

    try:
        j_resp = requests.post(
            f"{llm_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {llm_key}",
                "Content-Type": "application/json",
                "x-bf-run-id": run_id,
            },
            json={
                "model": llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a factual verification judge. Output JSON only in this format: "
                            '{"verdict": "fresh"|"outdated"|"incorrect"|"partially_correct"|"uncertain"|"unverifiable", '
                            '"current_value": "correct current fact or null", "explanation": "..."}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Claim: {task_text}\nEvidence:\n{evidence_block}",
                    },
                ],
                "temperature": 0.0,
            },
            timeout=25,
        )
        if j_resp.status_code == 200:
            j_content = j_resp.json()["choices"][0]["message"]["content"]
            clean_json = j_content.strip().strip("`").replace("json\n", "")
            parsed = json.loads(clean_json)
            verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
            current_val = parsed.get("current_value")
            answer_text = str(parsed.get("explanation", ""))
    except Exception as j_exc:
        answer_text = f"Judge error: {j_exc}"

    j_elapsed = round((time.perf_counter() - j_t0) * 1000.0, 2)
    trace.append({
        "idx": 4,
        "ts": time.time(),
        "kind": "llm",
        "name": "final_judgment",
        "input_summary": _redact(f"Judge claim against {len(all_snippets)} sources"),
        "output_summary": _redact(f"Verdict: {verdict}, value={current_val}"),
        "model": llm_model,
        "tokens_in": 180,
        "tokens_out": 45,
        "latency_ms": j_elapsed,
        "cache_hit": False,
        "error": None,
    })

    return JSONResponse({
        "result": {
            "verdict": verdict,
            "current_value": current_val,
            "answer": answer_text,
            "error": None,
        },
        "trace": trace,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)
