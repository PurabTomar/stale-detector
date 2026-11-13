"""
lab/agents/react_loop.py - ReAct research agent (Thought -> Action -> Observation loop).
Max 4 tool calls, stops itself, with tool error retry.
"""

import json
import re
from typing import Any

from lab.instrument import InstrumentedLLMClient, InstrumentedTavilyClient
from lab.schema import AgentResult, RunContext, StepRecord, Task
import llm
import verifier

REACT_SYSTEM_PROMPT = """You are a ReAct research agent verifying a factual claim.
You cycle through Thought, Action, Observation until you have enough evidence to reach a verdict.

Available Actions:
- search[<search query>]: Search live web sources using Tavily.
- finish[<verdict>, <current_value>]: Conclude the verification.
  <verdict> must be one of: fresh, outdated, incorrect, partially_correct, uncertain, unverifiable.
  <current_value> is the verified actual fact, person, price, or date today.

Rules:
1. You may execute at most 4 tool calls.
2. If a search fails or returns an error, retry with a refined query.
3. Once you find decisive evidence, immediately output finish[<verdict>, <current_value>].
4. Format every response strictly as:
Thought: <your reasoning>
Action: <search[...] or finish[...]>"""

_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.*)\]", re.DOTALL | re.IGNORECASE)


def run_react_loop_agent(task: Task, ctx: RunContext) -> AgentResult:
    """Execute ReAct tool-calling loop (max 4 tool calls)."""
    tavily_client = InstrumentedTavilyClient(ctx)
    llm_client = InstrumentedLLMClient(ctx)

    history = [f"Claim to verify: {task.input_text}\nCategory: {task.category}"]
    max_tool_calls = 4
    tool_calls_count = 0
    final_verdict = "uncertain"
    final_current_val: str | None = None
    final_answer = ""

    for step_num in range(1, max_tool_calls + 2):
        user_prompt = "\n\n".join(history)
        try:
            response_text, _ = llm_client.call(
                system_prompt=REACT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
                task=f"react_step_{step_num}",
            )
        except Exception as exc:
            return AgentResult(verdict="uncertain", error=f"LLM error: {exc}")

        history.append(response_text)
        action_match = _ACTION_RE.search(response_text)

        if not action_match:
            # If model didn't format an action, prompt it to finalize
            history.append("Observation: Invalid format. Please specify Action: search[...] or finish[...].")
            continue

        action_name = action_match.group(1).strip().lower()
        action_arg = action_match.group(2).strip()

        if action_name == "finish":
            # finish[verdict, current_value]
            parts = [p.strip() for p in action_arg.split(",", 1)]
            final_verdict = parts[0].strip().lower() if parts else "uncertain"
            if len(parts) > 1 and parts[1].strip():
                final_current_val = parts[1].strip()
            final_answer = response_text
            break

        elif action_name == "search":
            tool_calls_count += 1
            if tool_calls_count > max_tool_calls:
                history.append("Observation: Maximum tool calls reached. Please output finish[<verdict>, <current_value>].")
                continue

            query = action_arg.strip(" '\"")
            try:
                raw_results = tavily_client.search_raw(
                    query=query or task.input_text,
                    max_results=3,
                    category=task.category,
                )
                snippets = []
                for s in raw_results[:3]:
                    if isinstance(s, dict):
                        title = s.get("title", "")
                        url = s.get("url", "")
                        snip = s.get("snippet", "")
                        snippets.append(f"[{title} - {url}]: {snip}")
                obs_text = "Found sources:\n" + "\n".join(snippets) if snippets else "No sources found."
            except Exception as search_err:
                obs_text = f"Tool error: {search_err}. Please retry with an alternative query or action."

            history.append(f"Observation: {obs_text}")

        else:
            history.append(f"Observation: Unknown action '{action_name}'. Use search[...] or finish[...].")

    # If the loop ended without explicit finish action, run a final fallback summary
    if final_verdict not in {"fresh", "outdated", "incorrect", "partially_correct", "uncertain", "unverifiable"}:
        try:
            summary_prompt = (
                f"Conversation history:\n{user_prompt}\n\n"
                f"Based on the evidence gathered above, determine the final verdict for the claim: '{task.input_text}'.\n"
                f"Output JSON: {{\"verdict\": \"fresh\"|\"outdated\"|\"incorrect\"|\"partially_correct\"|\"uncertain\"|\"unverifiable\", \"current_value\": \"...\"}}"
            )
            raw_summary, _ = llm_client.call(
                system_prompt="You are a decisive verification assistant. Output JSON only.",
                user_prompt=summary_prompt,
                temperature=0.1,
                task="react_finalize",
            )
            parsed = llm.parse_json_response(raw_summary)
            final_verdict = str(parsed.get("verdict", "uncertain")).strip().lower()
            curr = parsed.get("current_value")
            if curr is not None:
                final_current_val = str(curr)
        except Exception:
            final_verdict = "uncertain"

    return AgentResult(
        verdict=final_verdict,
        current_value=final_current_val,
        answer=final_answer or f"ReAct completed in {tool_calls_count} tool calls.",
    )
