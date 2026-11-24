"""
audit_animations.py - Real-event-driven state machine and HTML renderers for the
"Check Freshness" animation sequence.

Design constraints:
- Strictly driven by real pipeline events (no fake timers or invented progress).
- CSS-first animations (< 600ms except loading loops), GPU-composited (transform,
  opacity, SVG stroke-dashoffset), reserved space to avoid layout shift.
- Full `prefers-reduced-motion: reduce` fallback (static progress bar, no motion).
- All user text, claims, and source metadata are escaped with `html.escape`.
- Requirements 1-12: live freshness strip, provisional score meter, live counters,
  activity feed, heuristic preview, accessibility aria-live, 150ms throttle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import time as _time
from typing import Any


STEP_ORDER: tuple[str, ...] = ("input", "detect", "verify", "score", "report")

STAGE_TO_BUTTON_LABEL: dict[str, str] = {
    "detect": "Detecting claims",
    "search": "Searching live sources",
    "extract": "Searching live sources",
    "verify": "Searching live sources",
    "judge": "Judging",
    "score": "Scoring",
    "report": "Scoring",
}

# Strip segment status to CSS class and icon
_STRIP_STATUS: dict[str, tuple[str, str]] = {
    "pending":   ("sad-strip-pending",   "⋯"),
    "running":   ("sad-strip-running",   "⟳"),
    "fresh":     ("sad-strip-fresh",     "✓"),
    "outdated":  ("sad-strip-outdated",  "⚠"),
    "incorrect": ("sad-strip-incorrect", "✗"),
    "partially_correct": ("sad-strip-partial",  "~"),
    "partial":   ("sad-strip-partial",   "~"),
    "uncertain": ("sad-strip-uncertain", "?"),
    "unverifiable": ("sad-strip-uncertain", "—"),
    "verification_failed": ("sad-strip-failed", "!"),
    "failed":    ("sad-strip-failed",    "!"),
    "skipped":   ("sad-strip-skipped",   "·"),
    "cancelled": ("sad-strip-skipped",   "·"),
}

# Statuses that contribute a verified count (not failed/skipped/pending)
_VERIFIED_STATUSES = frozenset({
    "fresh", "outdated", "incorrect", "partially_correct", "partial",
    "uncertain", "unverifiable",
})

# Statuses that contribute to the risk-weighted provisional score
_SCORABLE_STATUSES = frozenset({
    "fresh", "outdated", "incorrect", "partially_correct", "partial",
    "uncertain", "unverifiable",
})

# Activity feed event type to CSS class
_FEED_CLS: dict[str, str] = {
    "audit_started":     "sad-feed-info",
    "claims_detected":   "sad-feed-info",
    "claim_started":     "sad-feed-search",
    "source_found":      "sad-feed-source",
    "claim_verified":    "sad-feed-verified",
    "claim_failed":      "sad-feed-error",
    "audit_finished":    "sad-feed-info",
    "audit_cancelled":   "sad-feed-warn",
    "search":            "sad-feed-search",
    "extract":           "sad-feed-search",
    "judge":             "sad-feed-search",
    "done":              "sad-feed-verified",
    "error":             "sad-feed-error",
}


def _esc(val: object) -> str:
    return html.escape("" if val is None else str(val), quote=True)


@dataclass
class ClaimLiveState:
    """Real-time verification state for one claim during the Verify phase."""

    index: int
    claim: str
    category: str = "other"
    time_sensitivity: int = 50
    flagged: bool = True
    substep: str = "pending"  # pending | search | extract | judge | score | done | error
    status: str | None = None
    score: int | None = None
    sources: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None
    started_at: float = field(default_factory=_time.monotonic)


@dataclass
class AuditUIState:
    """Deterministic UI state derived exclusively from real pipeline events."""

    phase: str = "idle"  # idle | running | done | error | stopped
    active_stage: str = "idle"  # idle | detect | search | extract | judge | score | done | error | stopped
    button_state: str = "idle"  # idle | loading | success | error
    button_label: str = "Check Freshness"
    button_disabled: bool = False
    scan_active: bool = False
    input_glow: bool = False
    stepper: dict[str, str] = field(
        default_factory=lambda: {
            "input": "pending",
            "detect": "pending",
            "verify": "pending",
            "score": "pending",
            "report": "pending",
        }
    )
    failed_step: str | None = None
    error_reason: str | None = None
    progress_ratio: float = 0.0
    detected_claims: list[dict[str, Any]] = field(default_factory=list)
    claim_states: dict[int, ClaimLiveState] = field(default_factory=dict)
    sources_read: int = 0
    seen_source_urls: set[str] = field(default_factory=set)
    completed_claims: int = 0
    total_flagged_claims: int = 0
    stopped: bool = False
    reveal_ready: bool = False

    # --- NEW: Live Freshness Strip (Requirement 3) ---
    # strip_segments mirrors claim_states but keyed 0..N in original text order
    strip_segments: list[dict[str, Any]] = field(default_factory=list)

    # --- NEW: Provisional Score Meter (Requirement 5) ---
    provisional_score: int | None = None
    verified_count: int = 0      # claims with a real status (not pending/failed/skipped)
    failed_count: int = 0

    # --- NEW: Live Counters (Requirement 6) ---
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "fresh": 0,
            "outdated": 0,
            "incorrect": 0,
            "partial": 0,
            "uncertain": 0,
            "failed": 0,
        }
    )

    # --- NEW: Live Activity Feed (Requirement 7) ---
    # Newest-first list of feed entries.
    activity_feed: list[dict[str, Any]] = field(default_factory=list)
    # Last aria-live announcement text.
    last_announcement: str = ""

    # --- NEW: Throttle state (Requirement 10) ---
    last_render_monotonic: float = 0.0


def create_initial_ui_state() -> AuditUIState:
    """Return a clean idle AuditUIState (all stepper stages pending)."""
    return AuditUIState()


# ---------------------------------------------------------------------------
# Private helpers for the reducer
# ---------------------------------------------------------------------------

def _status_to_counter_key(status: str) -> str | None:
    """Map a claim status to a live-counter bucket key."""
    s = str(status or "").lower().strip()
    if s == "fresh":
        return "fresh"
    if s == "outdated":
        return "outdated"
    if s == "incorrect":
        return "incorrect"
    if s in ("partially_correct", "partial"):
        return "partial"
    if s in ("uncertain", "unverifiable"):
        return "uncertain"
    if s in ("verification_failed", "failed"):
        return "failed"
    return None


def _recompute_provisional_score(state: AuditUIState) -> None:
    """Recompute state.provisional_score using risk weighting across scorable claims (failed excluded)."""
    cat_weights = {
        "price": 1.5,
        "leadership": 1.5,
        "regulation": 1.4,
        "version": 1.2,
        "statistic": 1.2,
        "geography": 1.0,
        "records": 1.0,
        "other": 1.0,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for cs in state.claim_states.values():
        if cs.status in _SCORABLE_STATUSES and cs.score is not None:
            w = cat_weights.get(cs.category, 1.0)
            weighted_sum += float(cs.score) * w
            total_weight += w
    state.provisional_score = max(0, min(100, round(weighted_sum / total_weight))) if total_weight > 0 else None


def _add_feed_entry(
    state: AuditUIState,
    event_type: str,
    message: str,
    claim_idx: int | None = None,
    duration_ms: float | None = None,
) -> None:
    """Prepend a new activity-feed entry (newest-first). Caps at 60 entries."""
    entry: dict[str, Any] = {
        "event_type": event_type,
        "message": message,
        "claim_idx": claim_idx,
        "duration_ms": duration_ms,
        "ts": _time.monotonic(),
    }
    state.activity_feed.insert(0, entry)
    if len(state.activity_feed) > 60:
        state.activity_feed = state.activity_feed[:60]


def _sync_strip_from_states(state: AuditUIState) -> None:
    """Rebuild strip_segments list from claim_states (preserves text order by index)."""
    new_strip: list[dict[str, Any]] = []
    for idx in sorted(state.claim_states.keys()):
        cs = state.claim_states[idx]
        # Determine strip status
        if cs.substep in ("done", "error"):
            strip_status = cs.status or ("failed" if cs.substep == "error" else "uncertain")
        elif cs.substep == "pending":
            strip_status = "pending"
        else:
            strip_status = "running"
        new_strip.append({
            "idx": idx,
            "text": cs.claim,
            "status": strip_status,
            "duration_ms": cs.duration_ms,
            "sources": cs.sources,
        })
    state.strip_segments = new_strip


def apply_pipeline_event(
    state: AuditUIState,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> AuditUIState:
    """Pure event-to-UI-state reducer driven strictly by real pipeline events.

    Supported `event_type` values:
    - "reset"
    - "detect_start"
    - "detect_done"   (payload: claims, threshold)
    - "verify_start"  (payload: flagged_claims)
    - "claim_substep" (payload: claim_index, substep in search|extract|judge|score, sources?)
    - "claim_done"    (payload: claim_index, result)
    - "score_start"
    - "complete"      (payload: results?)
    - "error"         (payload: step in detect|verify|score|report, reason?)
    - "stopped"       (payload: completed?, total?)
    """
    data = payload or {}
    ev = str(event_type or "").strip().lower()

    if ev == "reset":
        return create_initial_ui_state()

    if ev in ("detect_start", "audit_started"):
        state.phase = "running"
        state.active_stage = "detect"
        state.button_state = "loading"
        state.button_label = STAGE_TO_BUTTON_LABEL["detect"]
        state.button_disabled = True
        state.scan_active = True
        state.input_glow = True
        state.failed_step = None
        state.error_reason = None
        state.stopped = False
        state.reveal_ready = False
        state.progress_ratio = 0.12
        state.strip_segments = []
        state.provisional_score = None
        state.verified_count = 0
        state.failed_count = 0
        state.counters = {k: 0 for k in state.counters}
        state.activity_feed = []
        state.last_announcement = "Starting claim detection…"
        state.stepper = {
            "input": "done",
            "detect": "running",
            "verify": "pending",
            "score": "pending",
            "report": "pending",
        }
        _add_feed_entry(state, "audit_started", "Audit started — detecting claims")
        return state

    if ev in ("detect_done", "claims_detected"):
        claims = list(data.get("claims") or [])
        threshold = int(data.get("threshold", 40))
        state.detected_claims = claims
        state.scan_active = False
        state.input_glow = False
        state.claim_states = {}
        flagged_idx = 0
        for idx, c in enumerate(claims):
            ts_raw = c.get("time_sensitivity", 50)
            try:
                ts_val = max(0, min(100, int(ts_raw)))
            except (TypeError, ValueError):
                ts_val = 50
            is_flagged = ts_val >= threshold
            if is_flagged:
                state.claim_states[flagged_idx] = ClaimLiveState(
                    index=flagged_idx,
                    claim=str(c.get("claim") or "").strip(),
                    category=str(c.get("category") or "other").strip().lower(),
                    time_sensitivity=ts_val,
                    flagged=True,
                    substep="pending",
                )
                flagged_idx += 1
        state.total_flagged_claims = flagged_idx
        state.completed_claims = 0
        state.progress_ratio = 0.25 if flagged_idx > 0 else 0.90
        _sync_strip_from_states(state)
        n_total = len(claims)
        _add_feed_entry(
            state,
            "claims_detected",
            f"Detected {n_total} claim(s) — {flagged_idx} flagged for live verification",
        )
        state.last_announcement = f"Detected {n_total} claims, {flagged_idx} to verify"
        if flagged_idx > 0:
            state.phase = "running"
            state.active_stage = "search"
            state.button_state = "loading"
            state.button_label = STAGE_TO_BUTTON_LABEL["search"]
            state.button_disabled = True
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "running",
                "score": "pending",
                "report": "pending",
            }
        else:
            state.phase = "running"
            state.active_stage = "score"
            state.button_state = "loading"
            state.button_label = STAGE_TO_BUTTON_LABEL["score"]
            state.button_disabled = True
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "done",
                "score": "running",
                "report": "pending",
            }
        return state

    if ev == "verify_start":
        flagged = list(data.get("flagged_claims") or [])
        if flagged and not state.claim_states:
            for idx, c in enumerate(flagged):
                ts_raw = c.get("time_sensitivity", 50)
                try:
                    ts_val = max(0, min(100, int(ts_raw)))
                except (TypeError, ValueError):
                    ts_val = 50
                state.claim_states[idx] = ClaimLiveState(
                    index=idx,
                    claim=str(c.get("claim") or "").strip(),
                    category=str(c.get("category") or "other").strip().lower(),
                    time_sensitivity=ts_val,
                    flagged=True,
                    substep="search",
                )
            state.total_flagged_claims = len(flagged)
        state.phase = "running"
        state.active_stage = "search"
        state.button_state = "loading"
        state.button_label = STAGE_TO_BUTTON_LABEL["search"]
        state.button_disabled = True
        state.scan_active = False
        state.input_glow = False
        state.stepper = {
            "input": "done",
            "detect": "done",
            "verify": "running",
            "score": "pending",
            "report": "pending",
        }
        _sync_strip_from_states(state)
        return state

    if ev == "claim_substep":
        claim_idx = int(data.get("claim_index", 0))
        substep = str(data.get("substep") or "search").strip().lower()
        if claim_idx not in state.claim_states:
            state.claim_states[claim_idx] = ClaimLiveState(
                index=claim_idx,
                claim=str(data.get("claim") or f"Claim #{claim_idx + 1}"),
                category=str(data.get("category") or "other").strip().lower(),
            )
        c_state = state.claim_states[claim_idx]
        if c_state.substep not in {"done", "error"}:
            c_state.substep = substep

        new_sources = data.get("sources") or []
        for src in new_sources:
            if not isinstance(src, dict):
                continue
            url = str(src.get("url") or "").strip()
            domain = str(src.get("domain") or "").strip()
            if not url and not domain:
                continue
            dedupe_key = url or domain
            if dedupe_key not in state.seen_source_urls:
                state.seen_source_urls.add(dedupe_key)
                state.sources_read += 1
                # Emit source feed entry
                _add_feed_entry(
                    state,
                    "source_found",
                    f"Source: {domain or url}",
                    claim_idx=claim_idx,
                )
            if not any(s.get("url") == url and s.get("domain") == domain for s in c_state.sources):
                c_state.sources.append(
                    {
                        "url": url,
                        "domain": domain or url,
                        "source_type": str(src.get("source_type") or "News"),
                    }
                )

        if substep == "search" and c_state.substep == "search":
            claim_short = c_state.claim[:60] + ("…" if len(c_state.claim) > 60 else "")
            _add_feed_entry(
                state,
                "claim_started",
                f'Searching: "{claim_short}"',
                claim_idx=claim_idx,
            )

        if state.phase == "running":
            if substep == "score":
                state.active_stage = "score"
                state.button_label = STAGE_TO_BUTTON_LABEL["score"]
            elif substep == "judge":
                state.active_stage = "judge"
                state.button_label = STAGE_TO_BUTTON_LABEL["judge"]
            else:
                state.active_stage = substep
                state.button_label = STAGE_TO_BUTTON_LABEL.get(substep, "Searching live sources")
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "running",
                "score": "pending",
                "report": "pending",
            }
        _sync_strip_from_states(state)
        return state

    if ev == "claim_started":
        claim_idx = int(data.get("claim_id", data.get("claim_index", data.get("id", 0))))
        if claim_idx not in state.claim_states:
            state.claim_states[claim_idx] = ClaimLiveState(
                index=claim_idx,
                claim=str(data.get("claim") or f"Claim #{claim_idx + 1}"),
                category=str(data.get("category") or "other").strip().lower(),
            )
        c_state = state.claim_states[claim_idx]
        if c_state.substep not in {"done", "error"}:
            c_state.substep = "search"
        c_state.started_at = _time.monotonic()
        claim_short = c_state.claim[:60] + ("…" if len(c_state.claim) > 60 else "")
        _add_feed_entry(
            state,
            "claim_started",
            f'Searching: "{claim_short}"',
            claim_idx=claim_idx,
        )
        _sync_strip_from_states(state)
        return state

    if ev == "source_found":
        claim_idx = int(data.get("claim_id", data.get("claim_index", data.get("id", 0))))
        domain = str(data.get("domain") or "").strip()
        url = str(data.get("url") or "").strip()
        src_type = str(data.get("type", data.get("source_type", "News"))).strip()
        key = url or domain
        if key and key not in state.seen_source_urls:
            state.seen_source_urls.add(key)
            state.sources_read += 1
            _add_feed_entry(
                state,
                "source_found",
                f"Source: {domain or url}",
                claim_idx=claim_idx,
            )
        if claim_idx in state.claim_states and key:
            c_state = state.claim_states[claim_idx]
            if not any(s.get("url") == url and s.get("domain") == domain for s in c_state.sources):
                c_state.sources.append(
                    {
                        "url": url,
                        "domain": domain or url,
                        "source_type": src_type,
                    }
                )
        _sync_strip_from_states(state)
        return state

    if ev == "claim_failed":
        claim_idx = int(data.get("claim_id", data.get("claim_index", data.get("id", 0))))
        reason = str(data.get("reason") or "Verification failed").strip()
        if claim_idx not in state.claim_states:
            state.claim_states[claim_idx] = ClaimLiveState(
                index=claim_idx,
                claim=str(data.get("claim") or f"Claim #{claim_idx + 1}"),
                category=str(data.get("category") or "other").strip().lower(),
            )
        c_state = state.claim_states[claim_idx]
        was_finished = c_state.substep in {"done", "error"}
        c_state.status = "verification_failed"
        c_state.error = reason
        c_state.score = None
        c_state.substep = "error"
        if not was_finished:
            state.completed_claims += 1
            state.failed_count += 1
            if "failed" in state.counters:
                state.counters["failed"] += 1
            _recompute_provisional_score(state)
            _add_feed_entry(
                state,
                "claim_failed",
                f"Verification failed: Claim #{claim_idx + 1} — {reason}",
                claim_idx=claim_idx,
            )
            state.last_announcement = (
                f"Claim {state.verified_count + state.failed_count} of "
                f"{state.total_flagged_claims} failed: {reason}"
            )
        total_f = max(state.total_flagged_claims, len(state.claim_states), 1)
        state.progress_ratio = round(0.25 + 0.65 * min(1.0, state.completed_claims / total_f), 3)
        _sync_strip_from_states(state)
        return state

    if ev in ("claim_done", "claim_verified"):
        claim_idx = int(data.get("claim_id", data.get("claim_index", data.get("id", 0))))
        raw_res = data.get("result")
        result = raw_res if isinstance(raw_res, dict) else data
        if claim_idx not in state.claim_states:
            state.claim_states[claim_idx] = ClaimLiveState(
                index=claim_idx,
                claim=str(result.get("claim") or f"Claim #{claim_idx + 1}"),
                category=str(result.get("category") or "other").strip().lower(),
            )
        c_state = state.claim_states[claim_idx]
        was_finished = c_state.substep in {"done", "error"}
        res_status = str(result.get("status") or "uncertain").strip().lower()

        # Duration from result or elapsed since started_at
        dur_ms = result.get("duration_ms")
        if dur_ms is None and hasattr(c_state, "started_at"):
            dur_ms = round((_time.monotonic() - c_state.started_at) * 1000.0, 1)
        c_state.duration_ms = dur_ms

        c_state.status = res_status
        c_state.score = result.get("score")
        c_state.error = result.get("error")
        c_state.substep = "error" if res_status == "verification_failed" else "done"

        for src in (result.get("sources") or []):
            if not isinstance(src, dict):
                continue
            url = str(src.get("url") or "").strip()
            domain = str(src.get("domain") or "").strip()
            key = url or domain
            if key and key not in state.seen_source_urls:
                state.seen_source_urls.add(key)
                state.sources_read += 1
            if key and not any(s.get("url") == url and s.get("domain") == domain for s in c_state.sources):
                c_state.sources.append(
                    {
                        "url": url,
                        "domain": domain or url,
                        "source_type": str(src.get("source_type") or "News"),
                    }
                )

        if not was_finished:
            state.completed_claims += 1
            # Update live counters
            ck = _status_to_counter_key(res_status)
            if ck and ck in state.counters:
                state.counters[ck] += 1
            # Update verified / failed counts
            if res_status in _VERIFIED_STATUSES:
                state.verified_count += 1
            elif res_status in ("verification_failed", "failed"):
                state.failed_count += 1
            # Recompute provisional score (excludes failed)
            _recompute_provisional_score(state)
            # Activity feed
            status_label = res_status.replace("_", " ").title()
            dur_str = f" in {dur_ms:.0f}ms" if dur_ms is not None else ""
            n_srcs = len(c_state.sources)
            _add_feed_entry(
                state,
                "claim_verified",
                f"Verified: {status_label}{dur_str} · {n_srcs} source(s)",
                claim_idx=claim_idx,
                duration_ms=dur_ms,
            )
            # aria-live announcement
            state.last_announcement = (
                f"Claim {state.verified_count + state.failed_count} of "
                f"{state.total_flagged_claims} verified: {status_label}"
            )

        total_f = max(state.total_flagged_claims, len(state.claim_states), 1)
        state.progress_ratio = round(0.25 + 0.65 * min(1.0, state.completed_claims / total_f), 3)
        if state.completed_claims >= total_f:
            state.active_stage = "score"
            state.button_label = STAGE_TO_BUTTON_LABEL["score"]
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "done",
                "score": "running",
                "report": "pending",
            }
        _sync_strip_from_states(state)
        return state

    if ev == "score_start":
        state.phase = "running"
        state.active_stage = "score"
        state.button_state = "loading"
        state.button_label = STAGE_TO_BUTTON_LABEL["score"]
        state.button_disabled = True
        state.scan_active = False
        state.input_glow = False
        state.progress_ratio = 0.92
        state.stepper = {
            "input": "done",
            "detect": "done",
            "verify": "done",
            "score": "running",
            "report": "pending",
        }
        return state

    if ev in ("complete", "audit_finished"):
        results = data.get("results") or []
        has_failed = (
            any(
                isinstance(r, dict) and str(r.get("status", "")).lower() == "verification_failed"
                for r in results
            )
            or state.failed_count > 0
            or any(cs.status == "verification_failed" for cs in state.claim_states.values())
        )
        state.scan_active = False
        state.input_glow = False
        state.button_disabled = False
        state.stopped = False
        state.reveal_ready = True
        # Final sync of strip segments
        _sync_strip_from_states(state)
        _add_feed_entry(state, "audit_finished", "Audit complete")
        if has_failed:
            state.phase = "error"
            state.active_stage = "error"
            state.failed_step = "verify"
            state.button_state = "error"
            state.button_label = "Retry"
            state.progress_ratio = 0.60
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "error",
                "score": "pending",
                "report": "pending",
            }
        else:
            state.phase = "done"
            state.active_stage = "done"
            state.failed_step = None
            state.button_state = "success"
            state.button_label = "Check Freshness"
            state.progress_ratio = 1.0
            state.stepper = {
                "input": "done",
                "detect": "done",
                "verify": "done",
                "score": "done",
                "report": "done",
            }
        return state

    if ev == "error":
        step = str(data.get("step") or "detect").strip().lower()
        if step not in {"detect", "verify", "score", "report"}:
            step = "detect"
        state.phase = "error"
        state.active_stage = "error"
        state.failed_step = step
        state.error_reason = str(data.get("reason") or "").strip() or None
        state.button_state = "error"
        state.button_label = "Retry"
        state.button_disabled = False
        state.scan_active = False
        state.input_glow = False
        state.reveal_ready = False
        reason_txt = state.error_reason or "Unknown error"
        _add_feed_entry(state, "error", f"Error at {step}: {reason_txt}")

        stepper: dict[str, str] = {}
        hit_error = False
        for s_name in STEP_ORDER:
            if s_name == step:
                stepper[s_name] = "error"
                hit_error = True
            elif not hit_error:
                stepper[s_name] = "done"
            else:
                stepper[s_name] = "pending"
        state.stepper = stepper
        return state

    if ev in ("stopped", "audit_cancelled"):
        state.phase = "stopped"
        state.active_stage = "stopped"
        state.stopped = True
        state.button_state = "idle"
        state.button_label = "Check Freshness"
        state.button_disabled = False
        state.scan_active = False
        state.input_glow = False
        if "completed" in data:
            state.completed_claims = int(data["completed"])
        if "total" in data:
            state.total_flagged_claims = int(data["total"])
        state.reveal_ready = state.completed_claims > 0
        state.stepper = {
            "input": "done",
            "detect": "done" if state.detected_claims else "pending",
            "verify": "done" if state.completed_claims > 0 else "pending",
            "score": "pending",
            "report": "pending",
        }
        # Mark all still-pending strip segments as "skipped"
        for seg in state.strip_segments:
            if seg.get("status") in ("pending", "running"):
                seg["status"] = "skipped"
        _add_feed_entry(
            state,
            "audit_cancelled",
            f"Audit stopped — {state.completed_claims} of {state.total_flagged_claims} verified",
        )
        return state

    return state


# ---------------------------------------------------------------------------
# Pure HTML Renderers (CSS-first, GPU-composited, reduced-motion aware)
# ---------------------------------------------------------------------------


def render_reduced_motion_progress_html(progress_ratio: float, label: str) -> str:
    """Render a simple static progress bar shown when `prefers-reduced-motion: reduce` is active."""
    pct = max(0, min(100, int(round(float(progress_ratio) * 100.0))))
    return (
        f'<div class="sad-reduced-progress" role="progressbar" '
        f'aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100" '
        f'aria-label="{_esc(label)}">'
        f'<div class="sad-reduced-progress-head">'
        f'<span>{_esc(label)}</span><span>{pct}%</span>'
        f'</div>'
        f'<div class="sad-reduced-progress-track">'
        f'<div class="sad-reduced-progress-fill" style="transform: scaleX({pct / 100.0:.3f});"></div>'
        f'</div>'
        f'</div>'
    )


def render_action_status_pill_html(state: AuditUIState) -> str:
    """Render the live Check Freshness status surface (Requirement A):
    - loading: shimmering gradient, CSS spinner, real step label
    - success: quick green check pulse
    - error: red shake with Retry label
    - stopped: partial results banner
    """
    if state.button_state == "loading":
        pct = max(0, min(100, int(round(state.progress_ratio * 100))))
        return (
            f'<div class="sad-live-btn-status sad-btn-loading" role="status" aria-live="polite">'
            f'<span class="sad-btn-spinner" aria-hidden="true"></span>'
            f'<span class="sad-btn-status-label">{_esc(state.button_label)}</span>'
            f'<span class="sad-btn-status-pct">{pct}%</span>'
            f'</div>'
            f'{render_reduced_motion_progress_html(state.progress_ratio, state.button_label)}'
        )
    if state.button_state == "success":
        return (
            '<div class="sad-live-btn-status sad-btn-success" role="status" aria-live="polite">'
            '<svg class="sad-inline-check-svg" viewBox="0 0 20 20" width="16" height="16" fill="none" aria-hidden="true">'
            '<path class="sad-check-path" d="M4.5 10.5L8.2 14.2L15.5 6.5" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
            '<span class="sad-btn-status-label">Verification complete</span>'
            '</div>'
        )
    if state.button_state == "error":
        reason_part = f" — {_esc(state.error_reason)}" if state.error_reason else ""
        return (
            f'<div class="sad-live-btn-status sad-btn-error" role="alert">'
            f'<span class="sad-btn-err-icon" aria-hidden="true">!</span>'
            f'<span class="sad-btn-status-label">Retry{reason_part}</span>'
            f'</div>'
        )
    if state.stopped:
        return (
            f'<div class="sad-live-btn-status sad-btn-stopped" role="status" aria-live="polite">'
            f'<span class="sad-btn-stop-dot" aria-hidden="true"></span>'
            f'<span class="sad-btn-status-label">Audit stopped — showing {state.completed_claims} of {max(state.total_flagged_claims, state.completed_claims)} verified claim(s)</span>'
            f'</div>'
        )
    return ""


def render_scan_effect_html(answer_text: str, is_scanning: bool) -> str:
    """Render the soft gradient scan line sweeping across user text while detecting (Requirement B)."""
    if not is_scanning:
        return ""
    snippet = str(answer_text or "").strip()
    if len(snippet) > 420:
        snippet = snippet[:420].rstrip() + "..."
    return (
        '<div class="sad-scan-box sad-scan-active" role="status" aria-live="polite">'
        '<div class="sad-scan-header">'
        '<span class="sad-scan-dot" aria-hidden="true"></span>'
        '<span>Scanning text for time-sensitive claims...</span>'
        '</div>'
        f'<div class="sad-scan-text-wrap">'
        f'<div class="sad-scan-beam" aria-hidden="true"></div>'
        f'<p class="sad-scan-text">{_esc(snippet)}</p>'
        f'</div>'
        '</div>'
    )


def render_claim_discovery_chips_html(
    claims: list[dict[str, Any]],
    threshold: int = 40,
    category_icons: dict[str, str] | None = None,
) -> str:
    """Render popping claim discovery chips with category icon, amber/grey dot,
    and animated time-sensitivity fill bar (Requirement C).
    """
    if not claims:
        return ""
    icons = category_icons or {}
    chips_html: list[str] = []
    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        claim_txt = str(c.get("claim") or "").strip()
        if not claim_txt:
            continue
        short_txt = claim_txt if len(claim_txt) <= 84 else claim_txt[:81].rstrip() + "..."
        cat = str(c.get("category") or "other").strip().lower()
        cat_icon = icons.get(cat, "📌 ")
        ts_raw = c.get("time_sensitivity", 50)
        try:
            ts_val = max(0, min(100, int(ts_raw)))
        except (TypeError, ValueError):
            ts_val = 50
        is_flagged = ts_val >= int(threshold)
        dot_cls = "sad-disc-dot-amber" if is_flagged else "sad-disc-dot-grey"
        status_title = f"Flagged for live verification (sensitivity {ts_val}%)" if is_flagged else f"Stable fact (sensitivity {ts_val}%)"
        bar_cls = "sad-disc-bar-flagged" if is_flagged else "sad-disc-bar-stable"
        delay_ms = min(idx * 55, 330)
        scale_val = max(0.06, min(1.0, ts_val / 100.0))
        chips_html.append(
            f'<div class="sad-discovery-chip" style="animation-delay: {delay_ms}ms;" title="{_esc(status_title)}">'
            f'<div class="sad-disc-top">'
            f'<span class="sad-disc-dot {dot_cls}" aria-hidden="true"></span>'
            f'<span class="sad-disc-cat">{_esc(cat_icon)}</span>'
            f'<span class="sad-disc-text">{_esc(short_txt)}</span>'
            f'<span class="sad-disc-score">{ts_val}%</span>'
            f'</div>'
            f'<div class="sad-disc-bar-track">'
            f'<div class="sad-disc-bar-fill {bar_cls}" style="--sad-bar-scale: {scale_val:.3f}; animation-delay: {delay_ms + 80}ms;"></div>'
            f'</div>'
            f'</div>'
        )
    if not chips_html:
        return ""
    return (
        '<div class="sad-discovery-wrap">'
        '<div class="sad-discovery-header">'
        f'<span>Discovered {len(chips_html)} claim(s)</span>'
        '<span class="sad-disc-legend">'
        '<span class="sad-disc-dot sad-disc-dot-amber"></span> Flagged &nbsp;'
        '<span class="sad-disc-dot sad-disc-dot-grey"></span> Stable'
        '</span>'
        '</div>'
        f'<div class="sad-discovery-grid">{"".join(chips_html)}</div>'
        '</div>'
    )


def render_verify_live_board_html(state: AuditUIState) -> str:
    """Render the live Verify phase board (Requirement E):
    - Reserved space to avoid layout shift
    - Skeleton shimmer cards with mini substeps (Search, Extract, Judge)
    - Source chips popping in one by one as real sources arrive
    - Real 'Sources read: N' counter
    """
    if not state.claim_states:
        return ""

    substep_order = ("search", "extract", "judge")
    substep_labels = {"search": "Search", "extract": "Extract", "judge": "Judge"}

    cards_html: list[str] = []
    for idx in sorted(state.claim_states.keys()):
        cs = state.claim_states[idx]
        short_claim = cs.claim if len(cs.claim) <= 105 else cs.claim[:102].rstrip() + "..."
        is_done = cs.substep == "done"
        is_err = cs.substep == "error"
        card_state_cls = "sad-v-done" if is_done else ("sad-v-error" if is_err else "sad-v-running")

        # Determine each mini-substep state strictly from cs.substep
        sub_rank = {"pending": -1, "search": 0, "extract": 1, "judge": 2, "score": 3, "done": 3, "error": 2}
        curr_rank = sub_rank.get(cs.substep, -1)
        mini_pills: list[str] = []
        for s_i, s_key in enumerate(substep_order):
            if is_done:
                st_cls = "sub-done"
            elif is_err and s_i == min(curr_rank, 2):
                st_cls = "sub-error"
            elif s_i < curr_rank:
                st_cls = "sub-done"
            elif s_i == curr_rank:
                st_cls = "sub-running"
            else:
                st_cls = "sub-pending"
            mini_pills.append(
                f'<span class="sad-mini-substep {st_cls}">{substep_labels[s_key]}</span>'
            )

        # Real arriving source chips (one by one with staggered pop)
        src_chips: list[str] = []
        for s_idx, src in enumerate(cs.sources[:5]):
            dom = _esc(src.get("domain") or "source")
            s_type = str(src.get("source_type") or "News").upper()
            src_chips.append(
                f'<span class="sad-live-src-chip" style="animation-delay: {min(s_idx * 55, 275)}ms;">'
                f'<span class="sad-live-src-badge">{_esc(s_type)}</span>{dom}'
                f'</span>'
            )

        if is_done:
            st_lbl = str(cs.status or "verified").replace("_", " ").title()
            score_str = f" &bull; {cs.score}/100" if cs.score is not None else ""
            body_html = f'<div class="sad-v-done-pill">✓ {_esc(st_lbl)}{score_str}</div>'
        elif is_err:
            body_html = '<div class="sad-v-err-pill">! Verification failed</div>'
        else:
            body_html = (
                '<div class="sad-v-skel-lines" aria-hidden="true">'
                '<span class="sad-skel-bar sad-skel-w1"></span>'
                '<span class="sad-skel-bar sad-skel-w2"></span>'
                '</div>'
            )

        src_row_html = (
            f'<div class="sad-live-src-row">{"".join(src_chips)}</div>'
            if src_chips
            else ""
        )
        cards_html.append(
            f'<div class="sad-verify-skel-card {card_state_cls}">'
            f'<div class="sad-v-card-head">'
            f'<span class="sad-v-claim-txt">#{idx + 1} {_esc(short_claim)}</span>'
            f'<div class="sad-mini-substeps">{"".join(mini_pills)}</div>'
            f'</div>'
            f'{body_html}'
            f'{src_row_html}'
            f'</div>'
        )

    total_f = max(state.total_flagged_claims, len(state.claim_states))
    return (
        '<div class="sad-verify-board" role="status" aria-live="polite">'
        '<div class="sad-verify-board-head">'
        f'<span class="sad-v-progress-txt">Verifying {state.completed_claims}/{total_f} flagged claim(s)</span>'
        f'<span class="sad-sources-read-pill" data-sources-read="{state.sources_read}">'
        f'Sources read: <strong>{state.sources_read}</strong>'
        f'</span>'
        '</div>'
        f'{render_reduced_motion_progress_html(state.progress_ratio, f"Verified {state.completed_claims}/{total_f} claims")}'
        f'<div class="sad-verify-cards-list">{"".join(cards_html)}</div>'
        '</div>'
    )


def render_stepper_check_svg() -> str:
    """Inline SVG draw-on check mark for completed stepper steps (Requirement D)."""
    return (
        '<svg class="sad-pipe-check-svg" viewBox="0 0 16 16" width="12" height="12" fill="none" aria-hidden="true">'
        '<path class="sad-pipe-check-path" d="M3.2 8.4L6.4 11.6L12.8 4.8" '
        'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg>'
    )


# ---------------------------------------------------------------------------
# NEW: Live Freshness Strip (Requirement 3)
# ---------------------------------------------------------------------------


def render_freshness_strip_html(state: AuditUIState) -> str:
    """Render the Live Freshness Strip directly under Input Studio.

    One segment per claim (in text order): pending=grey/pulse, running=indigo/animated,
    fresh=green, outdated=orange, incorrect=red, partial=amber, uncertain=slate,
    failed=hatched red-grey. Click scrolls to claim card (JS anchor).
    Shows 'N of M verified' label. aria-live polite region.
    """
    if not state.strip_segments and state.claim_states:
        _sync_strip_from_states(state)

    segments = state.strip_segments
    if not segments:
        # During detect phase, we may not have segments yet — show placeholder
        if state.phase not in ("running", "done", "error", "stopped"):
            return ""
        total = state.total_flagged_claims
        if total == 0:
            return ""

    total = max(state.total_flagged_claims, len(segments))
    verified = state.verified_count
    failed = state.failed_count
    pending_count = total - verified - failed

    # Build label
    if state.phase in ("done",) and failed == 0:
        label_txt = f"All {verified} claim(s) verified"
    elif state.phase in ("done", "error") and failed > 0:
        label_txt = f"{verified} of {total} verified · {failed} failed"
    elif state.phase == "stopped":
        label_txt = f"{verified} of {total} verified (stopped)"
    elif total > 0:
        label_txt = f"{verified} of {total} verified…"
    else:
        label_txt = "Verifying…"

    segs_html: list[str] = []
    for seg in segments:
        idx = seg["idx"]
        status = seg.get("status", "pending")
        cls, icon = _STRIP_STATUS.get(status, ("sad-strip-pending", "⋯"))
        claim_txt = seg.get("text", f"Claim {idx + 1}")
        short = claim_txt[:80] + ("…" if len(claim_txt) > 80 else "")
        dur = seg.get("duration_ms")
        dur_str = f" ({dur:.0f}ms)" if dur is not None else ""
        tooltip = f"#{idx + 1}: {short} — {status.replace('_', ' ')}{dur_str}"
        # anchor id used by card scroll
        segs_html.append(
            f'<span class="sad-strip-seg {cls}" '
            f'role="listitem" '
            f'title="{_esc(tooltip)}" '
            f'aria-label="Claim {idx + 1}: {_esc(status.replace(chr(95), chr(32)))}" '
            f'onclick="document.getElementById(\'sad-claim-card-{idx}\')?.scrollIntoView({{behavior:\'smooth\'}})" '
            f'style="cursor:pointer;">'
            f'<span class="sad-strip-icon" aria-hidden="true">{icon}</span>'
            f'</span>'
        )

    segs_block = "".join(segs_html)
    return (
        f'<div class="sad-freshness-strip" role="list" '
        f'aria-label="Claim verification progress: {_esc(label_txt)}">'
        f'{segs_block}'
        f'<span class="sad-strip-label">{_esc(label_txt)}</span>'
        f'</div>'
        # aria-live announcement (visually hidden, screen-reader only)
        f'<div aria-live="polite" aria-atomic="true" class="sad-aria-announcer">'
        f'{_esc(state.last_announcement)}'
        f'</div>'
    )


def render_freshness_strip_from_results(
    results: list[dict],
    ui_state: AuditUIState | None = None,
) -> str:
    """Render the freshness strip from results (or existing ui_state) to stay in results header."""
    if ui_state is not None and (ui_state.strip_segments or ui_state.claim_states):
        return render_freshness_strip_html(ui_state)
    if not results:
        return ""
    state = create_initial_ui_state()
    for idx, r in enumerate(results):
        status = str(r.get("status") or "uncertain").lower()
        score = r.get("score")
        state.claim_states[idx] = ClaimLiveState(
            index=idx,
            claim=str(r.get("claim") or f"Claim #{idx + 1}"),
            status=status,
            substep="error" if status == "verification_failed" else "done",
            score=score,
            duration_ms=r.get("duration_ms"),
            sources=r.get("sources") or [],
        )
    state.phase = "done"
    state.total_flagged_claims = len(results)
    state.completed_claims = len(results)
    state.verified_count = sum(1 for r in results if str(r.get("status", "")).lower() in _VERIFIED_STATUSES)
    state.failed_count = sum(1 for r in results if str(r.get("status", "")).lower() in ("verification_failed", "failed"))
    _sync_strip_from_states(state)
    return render_freshness_strip_html(state)


# ---------------------------------------------------------------------------
# NEW: Provisional Score Meter (Requirement 5)
# ---------------------------------------------------------------------------


def render_provisional_score_meter_html(state: AuditUIState) -> str:
    """Render the Provisional Score Meter (SVG arc gauge).

    Never shows a final-looking score before the audit completes.
    At the end, the CSS animates the ring to the real score.
    Failed claims are excluded; label says 'X could not be verified'.
    """
    if state.phase == "idle":
        return ""

    score = state.provisional_score
    total = state.total_flagged_claims
    verified = state.verified_count
    failed = state.failed_count
    is_final = state.phase in ("done", "error", "stopped")

    if score is None:
        score_display = "—"
        ring_pct = 0
    else:
        score_display = str(score)
        ring_pct = max(0, min(100, score))

    # SVG ring: circumference ≈ 2π×44 ≈ 276.46 for r=44
    circ = 276.46
    offset = circ * (1 - ring_pct / 100.0)

    if is_final:
        label = f"Score: {score_display}/100" if score is not None else "Score unavailable"
    else:
        label = f"Provisional ({verified} of {total} verified)"

    if failed > 0:
        failed_note = (
            f'<span class="sad-meter-failed-note">{failed} could not be verified</span>'
        )
    else:
        failed_note = ""

    provisional_cls = "" if is_final else "sad-meter-provisional"

    # Pick ring colour based on score
    if score is None or score < 40:
        ring_color = "#DC2626"  # red
    elif score < 65:
        ring_color = "#EA580C"  # orange
    elif score < 80:
        ring_color = "#D97706"  # amber
    else:
        ring_color = "#16A34A"  # green

    return (
        f'<div class="sad-provisional-meter {provisional_cls}" '
        f'role="meter" aria-valuenow="{ring_pct}" aria-valuemin="0" aria-valuemax="100" '
        f'aria-label="{_esc(label)}">'
        f'<div class="sad-meter-ring-wrap">'
        f'<svg class="sad-meter-svg" viewBox="0 0 100 100" width="80" height="80" aria-hidden="true">'
        f'<circle class="sad-meter-track" cx="50" cy="50" r="44" fill="none" stroke-width="8"/>'
        f'<circle class="sad-meter-arc" cx="50" cy="50" r="44" fill="none" '
        f'stroke="{ring_color}" stroke-width="8" stroke-linecap="round" '
        f'stroke-dasharray="{circ:.2f}" stroke-dashoffset="{offset:.2f}" '
        f'transform="rotate(-90 50 50)"/>'
        f'</svg>'
        f'<span class="sad-meter-score">{_esc(score_display)}</span>'
        f'</div>'
        f'<div class="sad-meter-label">{_esc(label)}</div>'
        f'{failed_note}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# NEW: Live Status Counters (Requirement 6)
# ---------------------------------------------------------------------------


def render_live_counters_html(state: AuditUIState) -> str:
    """Render live status counter chips (Fresh / Outdated / Incorrect / Partial / Uncertain / Failed).

    Each chip increments with a 'bump' CSS animation class added dynamically via
    a data attribute change that the CSS transition responds to.
    """
    if state.phase == "idle":
        return ""

    counters = state.counters
    _COUNTER_CFG: list[tuple[str, str, str]] = [
        ("fresh",     "✓", "sad-counter-fresh"),
        ("outdated",  "⚠", "sad-counter-outdated"),
        ("incorrect", "✗", "sad-counter-incorrect"),
        ("partial",   "~", "sad-counter-partial"),
        ("uncertain", "?", "sad-counter-uncertain"),
        ("failed",    "!", "sad-counter-failed"),
    ]

    chips: list[str] = []
    for key, icon, cls in _COUNTER_CFG:
        count = counters.get(key, 0)
        label = key.replace("_", " ").title()
        chips.append(
            f'<span class="sad-counter-chip {cls}" '
            f'data-count="{count}" '
            f'title="{count} {label}" '
            f'aria-label="{count} {_esc(label)}">'
            f'<span class="sad-counter-icon" aria-hidden="true">{icon}</span>'
            f'<span class="sad-counter-num">{count}</span>'
            f'<span class="sad-counter-lbl">{_esc(label)}</span>'
            f'</span>'
        )

    return (
        '<div class="sad-live-counters" role="status" aria-label="Live verification counters">'
        + "".join(chips)
        + '</div>'
    )


# ---------------------------------------------------------------------------
# NEW: Live Activity Feed (Requirement 7)
# ---------------------------------------------------------------------------


def render_activity_feed_html(state: AuditUIState) -> str:
    """Render the collapsible activity feed (newest-first, real durations, real events).

    Click on a feed line scrolls to the corresponding claim card.
    Errors appear in red.
    """
    feed = state.activity_feed
    if not feed:
        return ""

    n = len(feed)
    items: list[str] = []
    for entry in feed:  # already newest-first
        ev_type = str(entry.get("event_type") or "")
        msg = str(entry.get("message") or "")
        dur = entry.get("duration_ms")
        claim_idx = entry.get("claim_idx")
        cls = _FEED_CLS.get(ev_type, "sad-feed-info")
        dur_html = (
            f'<span class="sad-feed-dur">{dur:.0f}ms</span>' if dur is not None else ""
        )
        onclick_attr = ""
        if claim_idx is not None:
            onclick_attr = (
                f'onclick="document.getElementById(\'sad-claim-card-{claim_idx}\')?.scrollIntoView({{behavior:\'smooth\'}})" '
                f'style="cursor:pointer;" '
            )
        items.append(
            f'<li class="sad-feed-entry {cls}" {onclick_attr}>'
            f'<span class="sad-feed-msg">{_esc(msg)}</span>'
            f'{dur_html}'
            f'</li>'
        )

    return (
        f'<details class="sad-activity-feed">'
        f'<summary class="sad-feed-summary">Activity <span class="sad-feed-count">({n})</span></summary>'
        f'<ul class="sad-feed-list">{"".join(items)}</ul>'
        f'</details>'
    )


# ---------------------------------------------------------------------------
# NEW: Live Marked Text in Answer (Requirement 4)
# ---------------------------------------------------------------------------


def render_live_marked_text_html(answer_text: str, state: AuditUIState) -> str:
    """Render user text with real-time status highlights and icons as claims complete (Requirement 4).

    - Exact phrases get status highlight + icon as each claim completes.
    - Pending or running claims get a dotted pulsing underline (sad-hl-pending).
    - Failed claims get a hatched underline (sad-hl-failed).
    - Uses exact/fuzzy span matching against answer_text.
    - All user text is escaped with html.escape.
    """
    if not answer_text or not answer_text.strip():
        return ""
    if not state.claim_states and not state.detected_claims:
        return ""

    from extractor import locate_span_in_text

    spans: list[dict[str, Any]] = []

    # Map each claim in claim_states
    for idx, cs in sorted(state.claim_states.items()):
        raw_claim = cs.claim
        loc = locate_span_in_text(answer_text, span_text=raw_claim, claim_text=raw_claim)
        if loc is None and len(raw_claim) > 40:
            loc = locate_span_in_text(answer_text, span_text=raw_claim[:40], claim_text=raw_claim[:40])
        if loc is not None:
            s_start, s_end = loc[0], loc[1]
            if cs.substep == "done":
                c_status = cs.status or "uncertain"
            elif cs.substep == "error" or cs.status == "verification_failed":
                c_status = "failed"
            elif cs.substep == "skipped":
                c_status = "skipped"
            else:
                c_status = "pending"
            spans.append({
                "start": s_start,
                "end": s_end,
                "claim_idx": idx,
                "status": c_status,
                "claim": cs.claim,
                "score": cs.score,
            })

    # Sort and remove overlaps
    spans.sort(key=lambda s: s["start"])
    non_overlap: list[dict[str, Any]] = []
    last_end = 0
    for sp in spans:
        if sp["start"] >= last_end:
            non_overlap.append(sp)
            last_end = sp["end"]

    _HL_ICONS: dict[str, tuple[str, str]] = {
        "pending":   ("sad-hl-pending",   "⋯"),
        "fresh":     ("sad-hl-fresh",     "✓"),
        "outdated":  ("sad-hl-outdated",  "⚠"),
        "incorrect": ("sad-hl-incorrect", "✗"),
        "partial":   ("sad-hl-partial",   "~"),
        "partially_correct": ("sad-hl-partial", "~"),
        "uncertain": ("sad-hl-uncertain", "?"),
        "unverifiable": ("sad-hl-uncertain", "—"),
        "failed":    ("sad-hl-failed",    "!"),
        "skipped":   ("sad-hl-skipped",   "·"),
    }

    parts: list[str] = []
    cur = 0
    for sp in non_overlap:
        if sp["start"] > cur:
            parts.append(_esc(answer_text[cur:sp["start"]]).replace("\n", "<br>"))
        st_key = sp["status"]
        cls, icon = _HL_ICONS.get(st_key, ("sad-hl-pending", "⋯"))
        score_info = f" ({sp['score']}/100)" if sp.get("score") is not None else ""
        tooltip = f"#{sp['claim_idx'] + 1}: {st_key.replace('_', ' ').title()}{score_info}"
        span_slice = _esc(answer_text[sp["start"]:sp["end"]]).replace("\n", "<br>")
        claim_target_idx = sp["claim_idx"]
        parts.append(
            f'<span class="sad-live-hl {cls}" title="{_esc(tooltip)}" '
            f'data-claim-idx="{claim_target_idx}">'
            f'<span class="sad-live-hl-icon" aria-hidden="true">{icon}</span>'
            f'{span_slice}'
            f'</span>'
        )
        cur = sp["end"]

    if cur < len(answer_text):
        parts.append(_esc(answer_text[cur:]).replace("\n", "<br>"))

    return (
        f'<div class="sad-live-text-card">'
        f'<div class="sad-live-text-header">'
        f'<span class="sad-live-text-title">⚡ Live text verification</span>'
        f'</div>'
        f'<div class="sad-live-text-content">{"".join(parts)}</div>'
        f'</div>'
    )
