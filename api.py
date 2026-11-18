"""
api.py - FastAPI REST service for Stale-Answer Detector.

Exposes POST /verify so other AI applications, agents, or evaluation pipelines
can verify claims inside an LLM answer over HTTP and receive structured JSON.

Run locally with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from collections import deque
import hmac
import os
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from extractor import extract_claims, get_last_gateway_metrics as get_extract_metrics
from llm import check_gateway_health, is_use_bifrost
from reporting import build_report_payload
from settings import MAX_CLAIMS_PER_AUDIT
from stale_guard import (
    DEFAULT_MAX_CLAIMS,
    MAX_GUARD_ANSWER_CHARS,
    MAX_GUARD_CONTEXT_CHARS,
    MAX_GUARD_QUESTION_CHARS,
    guard as run_freshness_guard,
)
from verifier import verify_claims
from web_ui import get_web_ui_html

# ---------------------------------------------------------------------------
# Request & response schemas (powers OpenAPI docs at /docs)
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    """JSON body accepted by POST /verify."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": (
                        "Apple's latest iPhone is the iPhone 15 Pro Max starting at $1,199, "
                        "and GPT-4 Turbo costs $10 per million input tokens."
                    ),
                    "use_cache": True,
                }
            ]
        }
    )

    answer: str = Field(
        ...,
        description="LLM-generated answer text to extract and verify claim-by-claim.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to reuse the 6-hour on-disk SQLite Tavily cache.",
    )
    dual_judge: bool = Field(
        default=False,
        description="When true, two distinct judge models evaluate each claim through Bifrost.",
    )

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("The 'answer' field must contain non-empty text.")
        return value.strip()


class SourceLink(BaseModel):
    title: str
    url: str
    source_type: str | None = None
    published_date: str | None = None
    snippet: str | None = None


class JudgeVerdict(BaseModel):
    model: str
    status: str
    score: int | None = None
    current_info: str = ""
    reason: str = ""


class ClaimTokens(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class VerifiedClaim(BaseModel):
    index: int
    claim: str
    span: str | None = None
    category: str
    search_query: str
    status: str
    score: int | None = None
    confidence: str
    risk_level: str
    current_info: str
    current_value: str | None = None
    was_ever_true: str | None = None
    change_date: str | None = None
    correction: str | None = None
    effective_date: str | None = None
    change_summary: str | None = None
    stale_since: str | None = None
    stale_for_months: int | None = None
    stale_duration_label: str | None = None
    entity_domains: list[str] = Field(default_factory=list)
    sources_agreeing: int | None = None
    sources_total: int | None = None
    agreement_summary: str | None = None
    conflict_detected: bool = False
    conflict_note: str | None = None
    dual_judge_used: bool = False
    judge_models: list[str] = Field(default_factory=list)
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    model_disagreement: bool = False
    model_agreement_summary: str | None = None
    reason: str
    why_this_score: str
    score_breakdown: dict[str, Any] | None = None
    sources: list[SourceLink]
    extracted_url: str | None = None
    search_strategy: str | None = None
    cache_hit: bool
    tavily_credits_used: float
    model: str | None = None
    primary_model: str | None = None
    fallback_used: bool
    tokens: ClaimTokens
    latency_ms: float
    timestamp: str | None = None
    error: str | None = None


class VerifySummary(BaseModel):
    overall_score: int | None
    overall_tier: str
    claims_checked: int
    fresh: int
    outdated: int
    incorrect: int = 0
    partially_correct: int = 0
    uncertain: int
    unverifiable: int = 0
    tavily_credits_used: float


class TokenUsageSummary(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    requests_count: int
    fallback_count: int = 0
    total_latency_ms: float
    avg_latency_ms: float = 0.0
    audit_duration_ms: float = 0.0


class VerifyResponse(BaseModel):
    """Structured freshness verification JSON returned by POST /verify."""

    app: str
    routing_mode: str
    generated_at: str
    original_answer: str
    summary: VerifySummary
    models_used: list[str]
    fallback_routes: list[str]
    tokens_used: TokenUsageSummary
    claims: list[VerifiedClaim]


class GuardRequest(BaseModel):
    """JSON body accepted by POST /guard (Freshness Guard for RAG/chatbots)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "What is the latest stable Python release?",
                    "answer": "Python 3.11 is currently the latest stable release.",
                    "context": None,
                    "max_claims": 6,
                    "use_cache": True,
                    "auto_correct": True,
                }
            ]
        }
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_GUARD_QUESTION_CHARS,
        description=f"User prompt or question (max {MAX_GUARD_QUESTION_CHARS} chars).",
    )
    answer: str = Field(
        ...,
        min_length=1,
        max_length=MAX_GUARD_ANSWER_CHARS,
        description=f"Chatbot or RAG answer to audit for freshness (max {MAX_GUARD_ANSWER_CHARS} chars).",
    )
    context: str | None = Field(
        default=None,
        max_length=MAX_GUARD_CONTEXT_CHARS,
        description=f"Optional retrieved RAG context (max {MAX_GUARD_CONTEXT_CHARS} chars).",
    )
    max_claims: int = Field(
        default=DEFAULT_MAX_CLAIMS,
        ge=1,
        le=MAX_CLAIMS_PER_AUDIT,
        description=f"Maximum number of claims to verify (1 to {MAX_CLAIMS_PER_AUDIT}).",
    )
    use_cache: bool = Field(
        default=True,
        description="Reuse the 6-hour on-disk SQLite Tavily cache when true.",
    )
    dual_judge: bool = Field(
        default=False,
        description="Run two judge models per claim when true.",
    )
    auto_correct: bool = Field(
        default=True,
        description="Generate a cited corrected answer when stale or incorrect claims are found.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("The 'question' field must contain non-empty text.")
        return value.strip()

    @field_validator("answer")
    @classmethod
    def guard_answer_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("The 'answer' field must contain non-empty text.")
        return value.strip()


class GuardSource(BaseModel):
    index: int
    title: str
    url: str
    domain: str
    source_type: str
    published_date: str | None = None
    cited: bool = False


class GuardUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    llm_calls: int
    tavily_credits_used: float
    latency_ms: float
    routing_mode: str
    models_used: list[str] = Field(default_factory=list)


class GuardResponse(BaseModel):
    """Structured Freshness Guard verdict, claims, cited correction, sources, and usage."""

    verdict: str
    overall_score: int | None = None
    question: str
    answer: str
    context_provided: bool
    claims: list[dict[str, Any]]
    corrected_answer: str
    sources: list[GuardSource]
    usage: GuardUsage
    generated_at: str


# ---------------------------------------------------------------------------
# In-memory per-IP rate limiter & optional env API key guard for POST /guard
# ---------------------------------------------------------------------------

_GUARD_RATE_LOCK = threading.Lock()
_GUARD_IP_HITS: dict[str, deque[float]] = {}


def _reset_guard_rate_limiter() -> None:
    """Clear in-memory rate-limit counters (used by unit tests)."""
    with _GUARD_RATE_LOCK:
        _GUARD_IP_HITS.clear()


def _get_guard_rate_limit_config() -> tuple[int, float]:
    """Read per-IP rate limit (max_requests, window_seconds) from environment."""
    try:
        max_req = max(1, int(os.getenv("GUARD_RATE_LIMIT_REQUESTS", "30")))
    except ValueError:
        max_req = 30
    try:
        window_sec = max(1.0, float(os.getenv("GUARD_RATE_LIMIT_WINDOW_SECONDS", "60")))
    except ValueError:
        window_sec = 60.0
    return max_req, window_sec


def _get_client_ip(request: Request) -> str:
    """Extract client IP address from X-Forwarded-For or connection peer."""
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        first_hop = xff.split(",")[0].strip()
        if first_hop:
            return first_hop
    if request.client and request.client.host:
        return str(request.client.host)
    return "unknown"


def _enforce_guard_api_key(request: Request) -> None:
    """Validate optional GUARD_API_KEY (from env) via X-API-Key or Bearer token."""
    expected_key = (os.getenv("GUARD_API_KEY") or os.getenv("STALE_GUARD_API_KEY") or "").strip()
    if not expected_key:
        return
    provided = request.headers.get("x-api-key", "").strip()
    if not provided:
        auth = request.headers.get("authorization", "").strip()
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
    if not provided or not hmac.compare_digest(provided, expected_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Provide X-API-Key or Authorization: Bearer <key>.",
        )


def _enforce_guard_rate_limit(request: Request) -> None:
    """Apply a sliding-window per-IP in-memory rate limit to POST /guard."""
    max_req, window_sec = _get_guard_rate_limit_config()
    client_ip = _get_client_ip(request)
    now = time.monotonic()

    with _GUARD_RATE_LOCK:
        bucket = _GUARD_IP_HITS.setdefault(client_ip, deque())
        cutoff = now - window_sec
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= max_req:
            retry_after = max(1, int(window_sec - (now - bucket[0])) + 1)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({max_req} requests per {int(window_sec)}s per IP).",
                headers={"Retry-After": str(retry_after)},
            )
        bucket.append(now)


# ---------------------------------------------------------------------------
# FastAPI app & helpers
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stale-Answer Detector API",
    description=(
        "Extract time-sensitive claims from an LLM-generated answer, verify each claim "
        "against live web sources via Tavily, and return explainable freshness scores."
    ),
    version="1.0.0",
)


def _map_exception_to_http(exc: Exception) -> HTTPException:
    """
    Convert pipeline errors into clean HTTP status codes + friendly messages
    without ever exposing raw Python tracebacks.
    """
    raw = str(exc or "").strip()
    lower = raw.lower()

    # Invalid JSON returned by the LLM
    if any(tok in lower for tok in ("valid json", "json list", "json object", "parse json")):
        return HTTPException(
            status_code=502,
            detail=(
                "The LLM returned invalid JSON while extracting claims. "
                "Please retry the request or switch models in LLM_MODEL."
            ),
        )

    # Rate / usage limit
    if "429" in lower or "rate limit" in lower or "quota" in lower or "usage limit" in lower:
        return HTTPException(
            status_code=429,
            detail="Upstream rate or credit limit reached (429). Please retry shortly.",
        )

    # Invalid / unauthorized API keys
    if any(
        tok in lower
        for tok in ("401", "403", "invalid llm api key", "invalid gemini api key", "invalid bifrost virtual key")
    ):
        return HTTPException(
            status_code=502,
            detail=_first_clean_line(raw) or "Upstream API authentication failed (401/403).",
        )

    # Missing configuration keys (e.g. LLM_API_KEY / TAVILY_API_KEY)
    if isinstance(exc, ValueError) and ("is not set" in lower or "api key" in lower):
        return HTTPException(
            status_code=500,
            detail=_first_clean_line(raw) or "Server API keys are not configured.",
        )

    # Bifrost gateway down / connection failures
    if any(
        tok in lower
        for tok in ("bifrost gateway is down", "unreachable", "connection refused", "timed out", "connecterror")
    ):
        return HTTPException(
            status_code=503,
            detail=_first_clean_line(raw) or "LLM gateway is currently unreachable.",
        )

    clean_line = _first_clean_line(raw)
    return HTTPException(
        status_code=500,
        detail=clean_line or "Verification failed due to an internal error.",
    )


def _first_clean_line(text: str) -> str:
    """Return a single clean non-traceback error line for API error responses."""
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Traceback ") or stripped.startswith('File "'):
            continue
        return stripped[:240]
    return ""


@app.get("/")
def root_info(request: Request) -> Any:
    """
    Root endpoint.
    Serves the interactive single-page Web UI by default.
    Returns service JSON metadata when requested with Accept: application/json
    (without text/html) or with ?format=json.
    """
    accept = request.headers.get("accept", "").lower()
    format_param = request.query_params.get("format", "").lower()

    if format_param == "json" or ("application/json" in accept and "text/html" not in accept):
        return JSONResponse(
            {
                "service": "Stale-Answer Detector API",
                "version": "1.0.0",
                "docs_url": "/docs",
                "health_url": "/health",
                "verify_endpoint": "POST /verify",
                "guard_endpoint": "POST /guard",
            }
        )

    return HTMLResponse(content=get_web_ui_html())


@app.get("/api")
@app.get("/api/info")
def api_info(request: Request) -> Any:
    """Programmatic JSON service metadata endpoint or Web UI for browsers."""
    accept = request.headers.get("accept", "").lower()
    ua = request.headers.get("user-agent", "").lower()
    format_param = request.query_params.get("format", "").lower()

    if (
        ("text/html" in accept or "mozilla" in ua)
        and format_param != "json"
        and "application/json" not in accept
    ):
        return HTMLResponse(content=get_web_ui_html())

    return JSONResponse(
        {
            "service": "Stale-Answer Detector API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/health",
            "verify_endpoint": "POST /verify",
        }
    )


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Check API liveness and active LLM routing mode (Bifrost vs. Direct Gemini)."""
    gw = check_gateway_health(timeout=0.5)
    return {
        "status": "ok",
        "routing_mode": "bifrost" if is_use_bifrost() else "direct_gemini",
        "gateway": gw,
    }


@app.post("/verify", response_model=VerifyResponse)
def verify_answer(request: VerifyRequest) -> dict[str, Any]:
    """
    Extract time-sensitive claims from `request.answer`, verify each claim against
    live Tavily sources, and return the structured claims + freshness score JSON.
    """
    audit_t0 = time.perf_counter()
    try:
        # Step 1: Extract claims via configured LLM (Bifrost or Direct Gemini).
        claims, extract_model = extract_claims(request.answer)
        extract_meta = get_extract_metrics()

        # Step 2: Verify each claim using Tavily Search/Extract (only when claims exist).
        if not claims:
            results = []
        elif request.dual_judge:
            results = verify_claims(claims, use_cache=request.use_cache, dual_judge=True)
        else:
            results = verify_claims(claims, use_cache=request.use_cache)
    except Exception as exc:
        raise _map_exception_to_http(exc) from exc

    audit_duration_ms = round((time.perf_counter() - audit_t0) * 1000.0, 1)
    payload = build_report_payload(
        mode="REST API",
        original_answer=request.answer,
        verification_results=results,
        extract_model=extract_model or None,
        extract_meta=extract_meta,
        audit_duration_ms=audit_duration_ms,
    )
    payload["routing_mode"] = "bifrost" if is_use_bifrost() else "direct_gemini"
    return payload


@app.post(
    "/guard",
    response_model=GuardResponse,
    summary="Freshness Guard for AI applications",
    description=(
        "Verify a chatbot or RAG answer against live web sources, returning a "
        "verdict, verified claims, cited corrected answer, sources, and real usage. "
        "Protected by input size limits, optional GUARD_API_KEY env authentication, "
        "and an in-memory per-IP rate limit."
    ),
)
def guard_answer(payload: GuardRequest, request: Request) -> dict[str, Any]:
    """Run stale_guard.guard() over HTTP with API key and per-IP rate limiting."""
    _enforce_guard_api_key(request)
    _enforce_guard_rate_limit(request)
    try:
        return run_freshness_guard(
            question=payload.question,
            answer=payload.answer,
            context=payload.context,
            max_claims=payload.max_claims,
            use_cache=payload.use_cache,
            dual_judge=payload.dual_judge,
            auto_correct=payload.auto_correct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _map_exception_to_http(exc) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
