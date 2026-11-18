"""
audit_animations.py - Real-event-driven state machine and HTML renderers for the
"Check Freshness" animation sequence.

Design constraints:
- Strictly driven by real pipeline events (no fake timers or invented progress).
- CSS-first animations (< 600ms except loading loops), GPU-composited (transform,
  opacity, SVG stroke-dashoffset), reserved space to avoid layout shift.
- Full `prefers-reduced-motion: reduce` fallback (static progress bar, no motion).
- All user text, claims, and source metadata are escaped with `html.escape`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import html
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


def create_initial_ui_state() -> AuditUIState:
    """Return a clean idle AuditUIState (all stepper stages pending)."""
    return AuditUIState()


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

    if ev == "detect_start":
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
        state.stepper = {
            "input": "done",
            "detect": "running",
            "verify": "pending",
            "score": "pending",
            "report": "pending",
        }
        return state

    if ev == "detect_done":
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
            if not any(s.get("url") == url and s.get("domain") == domain for s in c_state.sources):
                c_state.sources.append(
                    {
                        "url": url,
                        "domain": domain or url,
                        "source_type": str(src.get("source_type") or "News"),
                    }
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
        return state

    if ev == "claim_done":
        claim_idx = int(data.get("claim_index", 0))
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        if claim_idx not in state.claim_states:
            state.claim_states[claim_idx] = ClaimLiveState(
                index=claim_idx,
                claim=str(result.get("claim") or f"Claim #{claim_idx + 1}"),
                category=str(result.get("category") or "other").strip().lower(),
            )
        c_state = state.claim_states[claim_idx]
        was_finished = c_state.substep in {"done", "error"}
        res_status = str(result.get("status") or "uncertain").strip().lower()
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

    if ev == "complete":
        results = data.get("results") or []
        has_failed = any(
            isinstance(r, dict) and str(r.get("status", "")).lower() == "verification_failed"
            for r in results
        )
        state.scan_active = False
        state.input_glow = False
        state.button_disabled = False
        state.stopped = False
        state.reveal_ready = True
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

    if ev == "stopped":
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
