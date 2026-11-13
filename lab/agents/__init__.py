"""
lab/agents - Registry of research agents evaluated in Agent Lab.
"""

from typing import Any

from lab.adapters import AgentAdapter, HttpAdapter, PythonAdapter
from lab.agents.detector_pipeline import run_detector_pipeline_agent
from lab.agents.llm_only import run_llm_only_agent
from lab.agents.react_loop import run_react_loop_agent
from lab.agents.single_search import run_single_search_agent

_IN_PROCESS_AGENTS = {
    "llm_only": run_llm_only_agent,
    "single_search": run_single_search_agent,
    "react_loop": run_react_loop_agent,
    "detector_pipeline": run_detector_pipeline_agent,
}


def list_agents() -> list[str]:
    """Return list of standard registered agent identifiers."""
    return list(_IN_PROCESS_AGENTS.keys()) + ["external_http", "external_agent"]


def get_agent(name: str, **kwargs: Any) -> AgentAdapter:
    """Instantiate and return an AgentAdapter by identifier."""
    name_clean = name.strip().lower()
    if name_clean in _IN_PROCESS_AGENTS:
        return PythonAdapter(_IN_PROCESS_AGENTS[name_clean], name=name_clean)

    if name_clean in {"external_http", "http", "external_agent"}:
        url = kwargs.get("url") or kwargs.get("endpoint_url") or "http://localhost:9001/verify"
        return HttpAdapter(endpoint_url=url, timeout_seconds=float(kwargs.get("timeout_seconds", 90.0)))

    raise ValueError(
        f"Unknown agent '{name}'. Available agents: {list_agents()}"
    )
