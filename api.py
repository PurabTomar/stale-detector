"""
api.py - FastAPI REST service for Stale-Answer Detector.

Exposes POST /verify so other AI applications, agents, or evaluation pipelines
can verify claims inside an LLM answer over HTTP and receive structured JSON.

Run locally with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from extractor import extract_claims, get_last_gateway_metrics as get_extract_metrics
from llm import check_gateway_health, is_use_bifrost
from reporting import build_report_payload
from verifier import verify_claims

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
    uncertain: int
    outdated: int
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
