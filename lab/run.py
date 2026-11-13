"""
lab/run.py - CLI entry point for Agent Lab benchmarking.

Usage:
    python -m lab.run [OPTIONS]

Examples:
    python -m lab.run --dry-run
    python -m lab.run --agents llm_only,single_search --repeats 1
"""

import argparse
from pathlib import Path
import sys

from lab.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Agent Lab: Framework-independent research agent benchmarking platform."
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="llm_only,single_search,react_loop,detector_pipeline",
        help="Comma-separated list of agents to run (or 'all').",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=str(Path(__file__).parent / "tasks" / "research_freshness.json"),
        help="Path to task suite JSON file.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of evaluation repetitions per task (default: 3).",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="clean",
        help="Comma-separated test scenarios (default: 'clean').",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Maximum concurrent task workers (default: 2).",
    )
    parser.add_argument(
        "--timeout-per-run",
        type=float,
        default=90.0,
        help="Timeout in seconds per agent execution (default: 90.0).",
    )
    parser.add_argument(
        "--budget-tavily-credits",
        type=float,
        default=None,
        help="Maximum cumulative Tavily credits allowed before stopping.",
    )
    parser.add_argument(
        "--budget-llm-calls",
        type=int,
        default=None,
        help="Maximum cumulative LLM calls allowed before stopping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print estimated Tavily credits and LLM calls, then exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompts.",
    )
    parser.add_argument(
        "--cache-tavily",
        action="store_true",
        help="Enable Tavily cache (default: cache bypassed for benchmark runs).",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(Path(__file__).parent / "results" / "runs.jsonl"),
        help="Path to output append-only results JSONL file.",
    )

    args = parser.parse_args()

    # Parse agents list
    if args.agents.strip().lower() == "all":
        agent_names = ["llm_only", "single_search", "react_loop", "detector_pipeline", "external_http"]
    else:
        agent_names = [a.strip() for a in args.agents.split(",") if a.strip()]

    scenario_list = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    bypass_cache = not args.cache_tavily

    records = run_benchmark(
        agents=agent_names,
        suite_path=args.suite,
        results_path=args.results,
        repeats=args.repeats,
        scenarios=scenario_list,
        concurrency=args.concurrency,
        timeout_per_run=args.timeout_per_run,
        budget_tavily_credits=args.budget_tavily_credits,
        budget_llm_calls=args.budget_llm_calls,
        dry_run=args.dry_run,
        bypass_cache=bypass_cache,
    )

    if not args.dry_run:
        print(f"\n[Agent Lab] Finished benchmark run. Processed {len(records)} new run(s).")
        print(f"[Agent Lab] Results saved to: {args.results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
