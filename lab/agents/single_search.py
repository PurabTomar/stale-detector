"""
lab/agents/single_search.py - Research agent doing one Tavily search followed by one LLM judgment.
"""

from typing import Any
import json

from lab.instrument import InstrumentedLLMClient, InstrumentedTavilyClient
from lab.schema import AgentResult, RunContext, Task
import llm
import verifier

SYSTEM_PROMPT = """You are an expert fact verification judge. You are given a claim and live web search results.
Judge the claim against the web sources. Output valid JSON in exactly this format:
{
  "verdict": "fresh" | "outdated" | "incorrect" | "partially_correct" | "uncertain" | "unverifiable",
  "current_value": "the correct current value, person, price, or fact established by sources",
  "explanation": "brief justification citing sources"
}
Rules:
- "outdated": was once true and changed.
- "incorrect": never true or fundamentally wrong.
- "partially_correct": partially true and partially inaccurate.
- "fresh": entirely true today.
- "uncertain" or "unverifiable": sources conflict or are insufficient.
Output JSON only. Do not include markdown code fences."""


def run_single_search_agent(task: Task, ctx: RunContext) -> AgentResult:
    """Execute single_search agent: 1 Tavily search -> 1 LLM judge."""
    tavily_client = InstrumentedTavilyClient(ctx)
    llm_client = InstrumentedLLMClient(ctx)

    try:
        # Step 1: Single web search
        search_res = tavily_client.search_claim(
            query=task.input_text,
            category=task.category,
            max_results=5,
            claim=task.input_text,
        )
        sources = search_res.get("results") or []

        # Format snippets
        snippets = []
        for s in sources[:5]:
            if isinstance(s, dict):
                title = s.get("title", "")
                url = s.get("url", "")
                snip = s.get("snippet", "")
                snippets.append(f"Source [{title} - {url}]:\n{snip}")

        evidence_text = "\n\n".join(snippets) if snippets else "No sources found."

        # Step 2: Single LLM judgment
        user_prompt = (
            f"Claim: {task.input_text}\n"
            f"Category: {task.category}\n\n"
            f"Live Search Evidence:\n{evidence_text}"
        )

        raw_text, _ = llm_client.call(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            task="single_search_judge",
        )
        parsed = llm.parse_json_response(raw_text)
        verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
        curr_val = parsed.get("current_value")
        curr_val_str = str(curr_val) if curr_val is not None else None
        expl = str(parsed.get("explanation", ""))
        return AgentResult(verdict=verdict, current_value=curr_val_str, answer=expl)
    except Exception as exc:
        return AgentResult(verdict="uncertain", current_value=None, error=str(exc))
