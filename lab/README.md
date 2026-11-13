# Agent Lab Trace Format
External agents POST/stdout JSON containing `result` and optional `trace`:
```json
{
  "result": {"verdict": "outdated", "current_value": "$1,199", "answer": "...", "error": null},
  "trace": [
    {"idx": 1, "kind": "tool", "name": "search", "input_summary": "iPhone price",
     "output_summary": "found 3 sources", "tokens_in": 0, "tokens_out": 0, "latency_ms": 120.5},
    {"idx": 2, "kind": "llm", "name": "judge", "input_summary": "eval snippets",
     "output_summary": "verdict=outdated", "tokens_in": 120, "tokens_out": 45, "latency_ms": 450.2}
  ]
}
```
If `trace` is omitted, the run is automatically recorded as black-box.
