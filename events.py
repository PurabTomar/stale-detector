"""
events.py - Pure-pipeline event model and reducer for real-time live verification.

Driven strictly by real pipeline events with real timestamps (no simulated progress
or fake timers). Unit testable with offline events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from audit_animations import (
    AuditUIState,
    ClaimLiveState,
    apply_pipeline_event as _apply_pipeline_event_raw,
    create_initial_ui_state,
    render_action_status_pill_html,
    render_activity_feed_html,
    render_claim_discovery_chips_html,
    render_freshness_strip_from_results,
    render_freshness_strip_html,
    render_live_counters_html,
    render_live_marked_text_html,
    render_provisional_score_meter_html,
    render_scan_effect_html,
    render_stepper_check_svg,
    render_verify_live_board_html,
)


@dataclass
class PipelineEvent:
    """An immutable event emitted by the verification pipeline with real timestamps."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def name(self) -> str:
        return self.event_type

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


def audit_started_event(**payload: Any) -> PipelineEvent:
    return PipelineEvent(event_type="audit_started", payload=payload)


def claims_detected_event(claims: list[dict[str, Any]], threshold: int = 40, **kwargs: Any) -> PipelineEvent:
    p = {"claims": claims, "threshold": threshold}
    p.update(kwargs)
    return PipelineEvent(event_type="claims_detected", payload=p)


def claim_started_event(claim_id: int, **kwargs: Any) -> PipelineEvent:
    p = {"claim_id": claim_id, "claim_index": claim_id}
    p.update(kwargs)
    return PipelineEvent(event_type="claim_started", payload=p)


def source_found_event(claim_id: int, domain: str, type: str = "News", **kwargs: Any) -> PipelineEvent:
    p = {"claim_id": claim_id, "claim_index": claim_id, "domain": domain, "type": type}
    p.update(kwargs)
    return PipelineEvent(event_type="source_found", payload=p)


def claim_verified_event(
    claim_id: int,
    status: str,
    score: int | None = None,
    duration_ms: float | None = None,
    sources: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> PipelineEvent:
    p = {
        "claim_id": claim_id,
        "claim_index": claim_id,
        "status": status,
        "score": score,
        "duration_ms": duration_ms,
        "sources": sources or [],
    }
    p.update(kwargs)
    return PipelineEvent(event_type="claim_verified", payload=p)


def claim_failed_event(claim_id: int, reason: str, **kwargs: Any) -> PipelineEvent:
    p = {"claim_id": claim_id, "claim_index": claim_id, "reason": reason}
    p.update(kwargs)
    return PipelineEvent(event_type="claim_failed", payload=p)


def audit_finished_event(results: list[dict[str, Any]] | None = None, **kwargs: Any) -> PipelineEvent:
    p = {"results": results or []}
    p.update(kwargs)
    return PipelineEvent(event_type="audit_finished", payload=p)


def audit_cancelled_event(completed: int = 0, total: int = 0, **kwargs: Any) -> PipelineEvent:
    p = {"completed": completed, "total": total}
    p.update(kwargs)
    return PipelineEvent(event_type="audit_cancelled", payload=p)


def apply_pipeline_event(
    state: AuditUIState,
    event: str | PipelineEvent,
    payload: dict[str, Any] | None = None,
) -> AuditUIState:
    """Apply a single pipeline event to the UI state."""
    if isinstance(event, PipelineEvent):
        ev_name = event.event_type
        ev_payload = event.payload
    else:
        ev_name = str(event)
        ev_payload = payload or {}
    return _apply_pipeline_event_raw(state, ev_name, ev_payload)


def reduce_events(
    events: list[PipelineEvent | dict[str, Any] | tuple[str, dict[str, Any]]],
    initial_state: AuditUIState | None = None,
) -> AuditUIState:
    """Pure reducer: folds an event sequence into final AuditUIState."""
    state = initial_state if initial_state is not None else create_initial_ui_state()
    for ev in events:
        if isinstance(ev, PipelineEvent):
            state = apply_pipeline_event(state, ev)
        elif isinstance(ev, tuple) and len(ev) == 2:
            state = apply_pipeline_event(state, ev[0], ev[1])
        elif isinstance(ev, dict):
            ev_type = str(ev.get("event_type") or ev.get("type") or "unknown")
            ev_payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else ev
            state = apply_pipeline_event(state, ev_type, ev_payload)
    return state
