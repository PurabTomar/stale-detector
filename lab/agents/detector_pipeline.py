"""
lab/agents/detector_pipeline.py - Research agent wrapping the Stale-Answer Detector pipeline
(extract claims -> category-aware search -> judge).
"""

from typing import Any

from extractor import extract_claims
from lab.instrument import InstrumentedLLMClient, InstrumentedTavilyClient
from lab.schema import AgentResult, RunContext, Task
import llm
import verifier


def run_detector_pipeline_agent(task: Task, ctx: RunContext) -> AgentResult:
    """Execute Stale-Answer Detector's standard pipeline within the instrumented lab."""
    llm_client = InstrumentedLLMClient(ctx)
    tavily_client = InstrumentedTavilyClient(ctx)

    try:
        # Step 1: Claim extraction
        claims, _ = extract_claims(task.input_text)
        if not claims:
            claims = [
                {
                    "claim": task.input_text,
                    "category": task.category,
                    "risk_level": "medium",
                    "text_span": task.input_text,
                }
            ]

        # Record extraction step in telemetry
        if ctx:
            from lab.schema import StepRecord
            import time
            ctx.add_step(
                StepRecord(
                    idx=0,
                    ts=time.time(),
                    kind="decision",
                    name="claim_extraction",
                    input_summary=task.input_text[:120],
                    output_summary=f"Extracted {len(claims)} claim(s)",
                )
            )

        # Step 2 & 3: Category-aware search & Judgement
        claim_verdicts: list[dict[str, Any]] = []
        for claim_dict in claims[:3]:  # audit up to top 3 claims
            c_text = claim_dict.get("claim") or task.input_text
            cat = claim_dict.get("category") or task.category
            risk = claim_dict.get("risk_level") or "medium"

            # Instrumented Tavily search
            search_data = tavily_client.search_claim(
                query=c_text,
                category=cat,
                risk_level=risk,
                max_results=5,
                claim=c_text,
            )
            sources = search_data.get("results") or []

            # Format snippets for judge
            snippets = []
            for s in sources[:5]:
                if isinstance(s, dict):
                    title = s.get("title", "")
                    url = s.get("url", "")
                    snip = s.get("snippet", "")
                    snippets.append(f"[{title} - {url}]: {snip}")
            evidence_text = "\n\n".join(snippets) if snippets else "No sources found."

            # Instrumented judge call
            judge_prompt = (
                f"Claim: {c_text}\n"
                f"Category: {cat}\n\n"
                f"Web Evidence:\n{evidence_text}\n\n"
                "Evaluate with evidence: was_ever_true (yes/no/unknown), change_date (only if was_ever_true=yes), "
                "current_value, and correction. If was_ever_true is 'no', status must be 'incorrect', not 'outdated'.\n"
                "Output JSON: {\"status\": \"fresh\"|\"outdated\"|\"incorrect\"|\"partially_correct\"|\"uncertain\"|\"unverifiable\", "
                "\"current_value\": \"...\", \"correction\": \"...\"}"
            )
            raw_judge, _ = llm_client.call(
                system_prompt="You are an expert fact-checking judge. Output JSON only.",
                user_prompt=judge_prompt,
                temperature=0.1,
                task="pipeline_judge",
            )
            parsed_judge = llm.parse_json_response(raw_judge)
            status_val = str(parsed_judge.get("status") or parsed_judge.get("verdict") or "uncertain").strip().lower()
            curr_val = parsed_judge.get("current_value")
            claim_verdicts.append(
                {
                    "status": status_val,
                    "current_value": str(curr_val) if curr_val is not None else None,
                    "claim": c_text,
                }
            )

        # Severity resolution
        severity_order = ["incorrect", "outdated", "partially_correct", "uncertain", "unverifiable", "fresh"]
        resolved_status = "fresh"
        resolved_current_val: str | None = None

        for sev in severity_order:
            matching = [cv for cv in claim_verdicts if cv.get("status") == sev]
            if matching:
                resolved_status = sev
                resolved_current_val = matching[0].get("current_value")
                break

        if not resolved_current_val and claim_verdicts:
            resolved_current_val = claim_verdicts[0].get("current_value")

        return AgentResult(
            verdict=resolved_status,
            current_value=resolved_current_val,
            answer=f"Detector pipeline audited {len(claims)} claim(s): {resolved_status}.",
        )
    except Exception as exc:
        return AgentResult(verdict="uncertain", error=str(exc))
