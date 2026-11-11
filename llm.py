"""
llm.py - Bifrost-aware OpenAI-compatible LLM client wrapper.

Reads USE_BIFROST, LLM_BASE_URL, LLM_API_KEY (or GEMINI_API_KEY when direct),
LLM_MODEL, LLM_MODEL_FALLBACKS, and optional LLM_VIRTUAL_KEY from .env (or
Streamlit Secrets) so requests can either:
- Route through a Bifrost AI gateway (`USE_BIFROST=true`, default), or
- Call Gemini's OpenAI-compatible endpoint directly (`USE_BIFROST=false`) for
  deployments such as Streamlit Community Cloud where no local Bifrost runs.

Bifrost Gateway features (`USE_BIFROST=true`):
- Sends native per-request fallbacks via `extra_body={"fallbacks": [...]}` and
  virtual key auth via `default_headers={"x-bf-vk": ...}`.
- Keeps an app-level single retry ONLY as a safety net when the gateway is
  unreachable (connection/timeout errors), plus simulated-outage testing.
- Records per-request telemetry (model answered, latency ms, prompt/completion
  tokens, fallback status) from Bifrost's `extra_fields` / `x-bifrost-*` headers.
- Checks Bifrost gateway health via `GET /health`.
"""

import json
import os
import re
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
)

# Load environment variables from a local .env file if one exists.
load_dotenv()

# Default local Bifrost gateway/dashboard URL per docs.getbifrost.ai.
DEFAULT_BIFROST_URL = "http://localhost:8080"

# Official Google Gemini OpenAI-compatible endpoint used when USE_BIFROST=false.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Prefix injected on the primary model name when simulating a provider outage.
SIMULATED_OUTAGE_PREFIX = "simulated-outage-"

# HTTP status codes considered safe for fallback safety-net recovery.
FALLBACK_STATUS_CODES = {429, 500, 502, 503, 504, 404}
UNREACHABLE_STATUS_CODES = {502, 503, 504}

# Thread-local store for per-request gateway metrics and outage toggle state.
_thread_state = threading.local()


def _get_config_value(name: str) -> str:
    """Read a setting from environment variables first, falling back to Streamlit Secrets."""
    val = os.getenv(name, "").strip()
    if val:
        return val
    if "streamlit" in sys.modules:
        try:
            st_mod = sys.modules["streamlit"]
            secrets = getattr(st_mod, "secrets", None)
            if secrets is not None and name in secrets:
                return str(secrets[name]).strip()
        except Exception:
            pass
    return ""


def is_use_bifrost() -> bool:
    """
    Return True when routing requests through Bifrost (`USE_BIFROST=true`, default).
    Returns False when `USE_BIFROST` is explicitly disabled (`false`, `0`, `no`, `off`),
    switching to Google Gemini's OpenAI-compatible endpoint directly.
    """
    load_dotenv(override=False)
    raw = _get_config_value("USE_BIFROST").lower()
    if not raw:
        return True
    return raw not in {"0", "false", "no", "off"}


def _normalize_gemini_model(model: str) -> str:
    """Strip Bifrost provider prefixes like `gemini/` or `google/` for direct Gemini calls."""
    cleaned = model.strip()
    lower = cleaned.lower()
    if lower.startswith("gemini/"):
        return cleaned.split("/", 1)[1].strip()
    if lower.startswith("google/"):
        return cleaned.split("/", 1)[1].strip()
    return cleaned


def _is_gemini_model(model: str) -> bool:
    """Return True if a configured model name refers to a Gemini model."""
    return _normalize_gemini_model(model).lower().startswith("gemini")


def get_llm_config() -> tuple[str, str | None, list[str]]:
    """
    Read LLM settings from environment variables (.env) or Streamlit Secrets.

    When `USE_BIFROST=false`, routes directly to Gemini's OpenAI-compatible
    endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`),
    accepting `LLM_API_KEY` or `GEMINI_API_KEY` and normalizing Gemini model names.

    Returns:
        A tuple of (api_key, base_url, model_chain), where model_chain starts
        with the primary model followed by any deduplicated fallback models.

    Raises:
        ValueError: If the required API key is missing.
    """
    # Reload .env in case the user edited it while the app is running.
    load_dotenv(override=False)

    use_bifrost = is_use_bifrost()
    raw_base_url = _get_config_value("LLM_BASE_URL")
    raw_primary = _get_config_value("LLM_MODEL")
    raw_fallbacks = _get_config_value("LLM_MODEL_FALLBACKS")
    fallback_models = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]

    if not use_bifrost:
        api_key = _get_config_value("LLM_API_KEY") or _get_config_value("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_API_KEY (or GEMINI_API_KEY) is not set. Please add your Gemini API key "
                "to your .env file or Streamlit Secrets when USE_BIFROST=false."
            )

        # Point directly at Google's OpenAI-compatible endpoint (ignoring localhost/Bifrost URLs).
        if raw_base_url and "generativelanguage.googleapis.com" in raw_base_url.lower():
            base_url = raw_base_url if raw_base_url.endswith("/") else raw_base_url + "/"
        else:
            base_url = GEMINI_OPENAI_BASE_URL

        gemini_override = _get_config_value("GEMINI_MODEL")
        gemini_fallbacks = [
            _normalize_gemini_model(m) for m in fallback_models if _is_gemini_model(m)
        ]

        if raw_primary and _is_gemini_model(raw_primary):
            primary_model = _normalize_gemini_model(raw_primary)
        elif gemini_override:
            primary_model = _normalize_gemini_model(gemini_override)
        elif gemini_fallbacks:
            primary_model = gemini_fallbacks[0]
        else:
            primary_model = DEFAULT_GEMINI_MODEL

        model_chain: list[str] = []
        for candidate in [primary_model, *gemini_fallbacks]:
            if candidate and candidate not in model_chain:
                model_chain.append(candidate)

        return api_key, base_url, model_chain

    api_key = _get_config_value("LLM_API_KEY")
    base_url = raw_base_url or None
    primary_model = raw_primary or "gpt-4o-mini"

    model_chain = []
    for candidate in [primary_model, *fallback_models]:
        if candidate and candidate not in model_chain:
            model_chain.append(candidate)

    if not api_key:
        raise ValueError(
            "LLM_API_KEY is not set. Please add it to your .env file "
            "(if your Bifrost gateway does not require a key, set a placeholder like 'bifrost-local')."
        )

    return api_key, base_url, model_chain


def get_llm_client() -> tuple[OpenAI, list[str], str | None]:
    """
    Create and return an OpenAI SDK client configured from .env / Secrets.

    Returns:
        A tuple of (OpenAI client, model_chain list, base_url).
    """
    api_key, base_url, model_chain = get_llm_config()

    # Optional Bifrost virtual key; sent via x-bf-vk only when routing via Bifrost.
    virtual_key = _get_config_value("LLM_VIRTUAL_KEY") if is_use_bifrost() else ""
    default_headers = {"x-bf-vk": virtual_key} if virtual_key else None

    # max_retries=0 disables the SDK's hidden retry loop so our explicit
    # 1-retry-per-model + fallback chain behaves predictably.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers,
        max_retries=0,
    )
    return client, model_chain, base_url


def _extract_status_code(exc: Exception) -> int | None:
    """
    Extract an HTTP status code from an OpenAI SDK or gateway exception.
    """
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    resp_status = getattr(response, "status_code", None)
    if isinstance(resp_status, int):
        return resp_status

    # Fallback: check the error message text for common gateway/provider codes.
    match = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", str(exc))
    if match:
        return int(match.group(1))

    return None


def set_simulate_outage(enabled: bool) -> None:
    """Enable or disable primary-provider outage simulation for the current thread."""
    _thread_state.simulate_outage = bool(enabled)


def is_simulate_outage() -> bool:
    """Return True if primary-provider outage simulation is enabled."""
    flag = getattr(_thread_state, "simulate_outage", None)
    if flag is not None:
        return bool(flag)
    return os.getenv("BIFROST_SIMULATE_OUTAGE", "").strip().lower() in {"1", "true", "yes"}


def pop_last_gateway_metrics() -> dict | None:
    """Return and clear the most recent per-request gateway metrics dict, if any."""
    metrics = getattr(_thread_state, "last_metrics", None)
    _thread_state.last_metrics = None
    return metrics


def get_last_gateway_metrics() -> dict | None:
    """Peek at the most recent per-request gateway metrics dict without clearing it."""
    return getattr(_thread_state, "last_metrics", None)


def _record_gateway_metrics(metrics: dict) -> None:
    """Save the latest per-request gateway metrics dict for UI telemetry."""
    _thread_state.last_metrics = dict(metrics)


def is_bifrost_gateway(base_url: str | None, virtual_key: str | None = None) -> bool:
    """
    Return True when configured to route through a Bifrost (or custom) gateway
    rather than hitting OpenAI or Gemini's direct API directly.
    """
    if not is_use_bifrost():
        return False
    if virtual_key and virtual_key.strip():
        return True
    if not base_url or not base_url.strip():
        return False
    lower = base_url.lower()
    return "api.openai.com" not in lower and "generativelanguage.googleapis.com" not in lower


def get_gateway_root_url(base_url: str | None = None) -> str:
    """
    Derive the Bifrost gateway root URL (scheme + host[:port]) from LLM_BASE_URL,
    defaulting to http://localhost:8080 when unset or pointing at a direct cloud API.
    """
    load_dotenv(override=False)
    raw = (base_url if base_url is not None else _get_config_value("LLM_BASE_URL")).strip()
    lower = raw.lower()
    if raw and "api.openai.com" not in lower and "generativelanguage.googleapis.com" not in lower:
        parsed = urlparse(raw)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    return DEFAULT_BIFROST_URL


def get_gateway_dashboard_url(base_url: str | None = None) -> str:
    """Return the Bifrost web dashboard logs URL (default: http://localhost:8080)."""
    return get_gateway_root_url(base_url)


def check_gateway_health(base_url: str | None = None, timeout: float = 0.8) -> dict:
    """
    Probe Bifrost's public `GET /health` endpoint per docs.getbifrost.ai.

    When `USE_BIFROST=false` (and no explicit gateway `base_url` was passed), skips
    network polling cleanly and reports Direct Gemini mode.

    Returns:
        Dict with healthy (bool), mode (str), status (str), url (str),
        health_url (str | None), version (str | None), providers (list), and detail (str).
    """
    if base_url is None and not is_use_bifrost():
        return {
            "healthy": True,
            "mode": "direct_gemini",
            "status": "direct_gemini",
            "url": GEMINI_OPENAI_BASE_URL,
            "health_url": None,
            "version": None,
            "providers": ["gemini"],
            "detail": "Direct Gemini OpenAI-compatible API (USE_BIFROST=false)",
        }

    root_url = get_gateway_root_url(base_url)
    health_url = f"{root_url}/health"
    try:
        resp = httpx.get(health_url, timeout=timeout)
        payload = {}
        try:
            data = resp.json()
            if isinstance(data, dict):
                payload = data
        except Exception:
            payload = {}

        status_text = str(payload.get("status", "")).strip()
        providers = payload.get("providers") if isinstance(payload.get("providers"), list) else []
        version = payload.get("version")

        is_ok = resp.status_code == 200 and (
            not status_text or status_text.lower() in {"ok", "healthy"}
        )
        return {
            "healthy": is_ok,
            "mode": "bifrost",
            "status": status_text or ("ok" if is_ok else f"HTTP {resp.status_code}"),
            "url": root_url,
            "health_url": health_url,
            "version": version,
            "providers": providers,
            "detail": f"HTTP {resp.status_code}" + (f" ({version})" if version else ""),
        }
    except Exception as exc:
        return {
            "healthy": False,
            "mode": "bifrost",
            "status": "unreachable",
            "url": root_url,
            "health_url": health_url,
            "version": None,
            "providers": [],
            "detail": str(exc),
        }


def _is_gateway_unreachable(exc: Exception) -> bool:
    """Return True if an exception indicates the endpoint could not be reached."""
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.ConnectError,
            httpx.TimeoutException,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return True
    status_code = _extract_status_code(exc)
    if status_code in UNREACHABLE_STATUS_CODES:
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection refused",
            "connecterror",
            "unreachable",
            "timed out",
            "connection error",
            "winerror 10061",
            "failed to establish a new connection",
        )
    )


def format_llm_error(
    exc: Exception,
    *,
    use_bifrost: bool | None = None,
    base_url: str | None = None,
    primary_model: str | None = None,
) -> str:
    """
    Convert an LLM SDK or Bifrost gateway failure into a concise, beginner-friendly
    message without exposing raw Python stack traces.
    """
    bifrost_enabled = is_use_bifrost() if use_bifrost is None else bool(use_bifrost)
    endpoint = base_url or (GEMINI_OPENAI_BASE_URL if not bifrost_enabled else get_gateway_root_url())
    status_code = _extract_status_code(exc)
    raw_msg = str(exc or "").strip()
    lower = raw_msg.lower()

    if bifrost_enabled and _is_gateway_unreachable(exc):
        return (
            f"Bifrost gateway is down or unreachable at {endpoint}. "
            "Start Bifrost locally on port 8080, or set USE_BIFROST=false in .env "
            "to call Gemini directly."
        )

    if (not bifrost_enabled) and _is_gateway_unreachable(exc):
        return (
            f"Could not connect to the Gemini API endpoint ({endpoint}). "
            "Please check your network connection and try again."
        )

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)) or status_code in {401, 403} or any(
        tok in lower
        for tok in (
            "invalid api key",
            "incorrect api key",
            "api_key_invalid",
            "api key not valid",
            "authentication",
            "unauthorized",
            "invalid x-bf-vk",
            "virtual key",
        )
    ):
        if "virtual key" in lower or "x-bf-vk" in lower:
            return (
                "Invalid Bifrost virtual key (LLM_VIRTUAL_KEY / x-bf-vk). "
                "Please double-check the value in your .env file."
            )
        if not bifrost_enabled:
            return (
                "Invalid Gemini API key (401/403). Please verify LLM_API_KEY "
                "(or GEMINI_API_KEY) in your .env file or Streamlit Secrets."
            )
        return (
            "Invalid LLM API key or gateway credentials (401/403). Please verify "
            "LLM_API_KEY and your Bifrost provider keys."
        )

    if status_code == 429 or "rate limit" in lower or "quota" in lower or "resource_exhausted" in lower:
        return (
            "The LLM rate or quota limit was reached (429). Please wait a moment "
            "or add fallback models in LLM_MODEL_FALLBACKS."
        )

    if status_code == 404 or "not found" in lower:
        model_hint = f" '{primary_model}'" if primary_model else ""
        return (
            f"The configured model{model_hint} was not found (404). Check LLM_MODEL "
            "in your .env or Streamlit Secrets (for Direct Gemini, use e.g. 'gemini-2.5-flash')."
        )

    if "empty message" in lower or "no choices" in lower:
        return "The LLM returned an empty response. Please try again."

    # Strip multi-line stack traces and keep only a concise first line.
    first_line = next((ln.strip() for ln in raw_msg.splitlines() if ln.strip() and not ln.strip().startswith("File \"")), "")
    if len(first_line) > 180:
        first_line = first_line[:177] + "..."
    status_note = f" (HTTP {status_code})" if status_code else ""
    if first_line and "traceback" not in first_line.lower():
        return f"LLM request failed{status_note}: {first_line}"
    return f"LLM request failed{status_note}. Please verify your API configuration and try again."


def _models_refer_to_same(configured: str, returned: str) -> bool:
    """
    Compare a configured model string (e.g. 'openai/gpt-4o-mini') against a
    model string returned by an upstream response (e.g. 'gpt-4o-mini-2024-07-18').
    """
    if not configured or not returned:
        return False
    c_norm = configured.strip().lower()
    r_norm = returned.strip().lower()
    if c_norm == r_norm:
        return True
    c_base = c_norm.split("/", 1)[-1]
    r_base = r_norm.split("/", 1)[-1]
    if c_base == r_base:
        return True
    # Only match dated/versioned snapshots like 'gpt-4o-mini-2024-07-18', not distinct
    # product variants like 'gemini-2.5-flash' vs 'gemini-2.5-flash-lite'.
    return bool(re.match(rf"^{re.escape(c_base)}-\d", r_base))


def _build_outage_model_name(primary_model: str) -> str:
    """Build a synthetic model identifier that triggers an error on the primary model."""
    if "/" in primary_model:
        provider, model_part = primary_model.split("/", 1)
        return f"{provider}/{SIMULATED_OUTAGE_PREFIX}{model_part}"
    return f"{SIMULATED_OUTAGE_PREFIX}{primary_model}"


def _extract_response_metadata(
    response: Any,
    raw_headers: dict[str, str] | None,
    original_primary: str,
    requested_model: str,
    fallback_models: list[str],
    elapsed_ms: float,
    simulated_outage: bool = False,
    forced_fallback_used: bool = False,
    fallback_target: str | None = None,
    routing_mode: str | None = None,
) -> dict:
    """
    Build per-request gateway telemetry from OpenAI response fields plus Bifrost's
    `extra_fields` (`routing_info`, `latency`) and `x-bifrost-*` headers.
    """
    headers = {str(k).lower(): str(v) for k, v in (raw_headers or {}).items()}
    model_extra = getattr(response, "model_extra", None)
    if not isinstance(model_extra, dict):
        model_extra = {}
    extra_fields = model_extra.get("extra_fields") or getattr(response, "extra_fields", None)
    if not isinstance(extra_fields, dict):
        extra_fields = {}

    routing_info = extra_fields.get("routing_info")
    if not isinstance(routing_info, dict):
        routing_info = {}

    # 1. Token usage (OpenAI SDK CompletionUsage or dict in mocks)
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    else:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

    # 2. Answering provider & model
    ef_provider = str(routing_info.get("provider") or extra_fields.get("provider") or "").strip()
    ef_model = str(routing_info.get("model") or "").strip()
    hdr_provider = headers.get("x-bifrost-routing-info-provider", "").strip()
    hdr_model = headers.get("x-bifrost-routing-info-model", "").strip()
    resp_model = getattr(response, "model", None)
    resp_model_str = resp_model.strip() if isinstance(resp_model, str) else ""

    answering_provider = ef_provider or hdr_provider
    candidate_model = ef_model or hdr_model or resp_model_str or fallback_target or requested_model

    if answering_provider and candidate_model and "/" not in candidate_model:
        qualified_candidate = f"{answering_provider}/{candidate_model}"
    else:
        qualified_candidate = candidate_model

    # Prefer exact configured name from model_chain when it matches the returned model.
    model_used = qualified_candidate
    for configured in [original_primary, *fallback_models]:
        if _models_refer_to_same(configured, qualified_candidate) or _models_refer_to_same(
            configured, candidate_model
        ):
            model_used = configured
            break

    # If the response came from a fallback model explicitly in fallback_target and
    # response didn't identify a different model, honor fallback_target.
    if fallback_target and not resp_model_str and not ef_model and not hdr_model:
        model_used = fallback_target

    # 3. Fallback status (Bifrost metadata OR outage/model comparison)
    ef_is_fallback = routing_info.get("is_fallback")
    hdr_is_fallback = headers.get("x-bifrost-routing-info-is-fallback") or headers.get(
        "x-bifrost-is-fallback", ""
    )

    matched_fallback = any(
        _models_refer_to_same(fb, model_used) or _models_refer_to_same(fb, candidate_model)
        for fb in fallback_models
    ) and not _models_refer_to_same(original_primary, model_used)

    fallback_used = bool(
        forced_fallback_used
        or ef_is_fallback is True
        or hdr_is_fallback.strip().lower() == "true"
        or (simulated_outage and not _models_refer_to_same(original_primary, model_used))
        or matched_fallback
    )

    # 4. Latency in milliseconds (prefer Bifrost extra_fields.latency / header when single call)
    gw_latency_raw = extra_fields.get("latency") or headers.get("x-bifrost-latency")
    latency_ms = round(float(elapsed_ms), 1)
    if not forced_fallback_used and gw_latency_raw is not None:
        try:
            cleaned_lat = str(gw_latency_raw).strip().removesuffix("ms").strip()
            gw_val = float(cleaned_lat)
            if gw_val > 0:
                latency_ms = round(gw_val, 1)
        except (TypeError, ValueError):
            pass

    return {
        "primary_model": original_primary,
        "requested_model": requested_model,
        "model_used": model_used,
        "provider_used": answering_provider or (model_used.split("/", 1)[0] if "/" in model_used else ("gemini" if routing_mode == "direct_gemini" else None)),
        "fallback_models": list(fallback_models),
        "fallback_used": fallback_used,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "simulated_outage": simulated_outage,
        "routing_mode": routing_mode or ("bifrost" if is_use_bifrost() else "direct_gemini"),
        "timestamp": time.strftime("%H:%M:%S"),
    }


def _invoke_chat_completion(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    fallbacks: list[str] | None = None,
    use_native_fallbacks: bool = True,
) -> tuple[Any, dict[str, str]]:
    """
    Invoke `chat.completions.create` with Bifrost's native `extra_body={"fallbacks": ...}`
    when configured, reading raw HTTP response headers when using a real OpenAI client.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_native_fallbacks and fallbacks:
        kwargs["extra_body"] = {"fallbacks": list(fallbacks)}

    # Use with_raw_response on real OpenAI clients to capture x-bifrost-* headers;
    # fall back to .create() so standard MagicMock clients in tests work cleanly.
    if isinstance(client, OpenAI):
        raw_resp = client.chat.completions.with_raw_response.create(**kwargs)
        parsed = raw_resp.parse()
        headers = dict(raw_resp.headers) if getattr(raw_resp, "headers", None) else {}
        return parsed, headers

    response = client.chat.completions.create(**kwargs)
    return response, {}


def _extract_content(response: Any) -> str:
    """Extract and validate non-empty assistant message text from ChatCompletion."""
    if not getattr(response, "choices", None):
        raise ValueError("LLM returned no choices in the response.")
    content = response.choices[0].message.content
    if not content or not str(content).strip():
        raise ValueError("LLM returned an empty message.")
    return str(content).strip()


def call_llm_with_meta(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    simulate_outage: bool | None = None,
) -> tuple[str, dict]:
    """
    Send a chat completion request (via Bifrost native fallbacks or Direct Gemini
    OpenAI-compatible endpoint when USE_BIFROST=false) and return
    `(response_text, telemetry_meta)`.

    Behavior:
    1. When USE_BIFROST=true, sends `model=primary_model` with
       `extra_body={"fallbacks": fallback_models}` so Bifrost fails over natively.
    2. When USE_BIFROST=false, calls Gemini's OpenAI-compatible endpoint directly
       and walks `fallback_models` explicitly on recoverable status codes.
    3. Keeps a single safety-net retry ONLY when the endpoint is unreachable
       (connection failure, timeout, or 502/503/504).
    4. If `simulate_outage` is True, forces the primary model to fail on purpose so
       Bifrost (or the fallback path) switches cleanly to the next fallback model.
    """
    client, model_chain, base_url = get_llm_client()
    original_primary = model_chain[0]
    fallback_models = model_chain[1:]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    outage = is_simulate_outage() if simulate_outage is None else bool(simulate_outage)
    request_model = _build_outage_model_name(original_primary) if outage else original_primary

    # Direct OpenAI and Direct Gemini endpoints reject Bifrost's root `fallbacks` field.
    is_direct_openai = bool(base_url and "api.openai.com" in base_url.lower() and not _get_config_value("LLM_VIRTUAL_KEY"))
    is_direct_gemini = bool(base_url and "generativelanguage.googleapis.com" in base_url.lower())
    is_direct_endpoint = is_direct_openai or is_direct_gemini
    use_native_fallbacks = bool(fallback_models) and not is_direct_endpoint
    routing_mode = "direct_gemini" if is_direct_gemini else "bifrost"

    last_error: Exception | None = None
    t0 = time.perf_counter()

    # Step 1: Primary request (with 1 safety-net retry ONLY if gateway/endpoint unreachable).
    for attempt in range(2):
        try:
            response, headers = _invoke_chat_completion(
                client=client,
                model=request_model,
                messages=messages,
                temperature=temperature,
                fallbacks=fallback_models,
                use_native_fallbacks=use_native_fallbacks,
            )
            content = _extract_content(response)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            meta = _extract_response_metadata(
                response=response,
                raw_headers=headers,
                original_primary=original_primary,
                requested_model=request_model,
                fallback_models=fallback_models,
                elapsed_ms=elapsed_ms,
                simulated_outage=outage,
                routing_mode=routing_mode,
            )
            _record_gateway_metrics(meta)
            return content, meta

        except BadRequestError as exc:
            last_error = exc
            # If a non-Bifrost OpenAI proxy rejects "fallbacks", retry once without extra_body.
            if use_native_fallbacks and "fallbacks" in str(exc).lower():
                use_native_fallbacks = False
                continue
            break

        except Exception as exc:
            last_error = exc
            # Keep app-level retry ONLY when the endpoint itself is unreachable.
            if attempt == 0 and _is_gateway_unreachable(exc):
                time.sleep(1)
                continue
            break

    # Step 2: Explicit app-level failover when simulating an outage or calling Direct Gemini/OpenAI.
    should_try_explicit_fallback = False
    if last_error is not None and fallback_models and not _is_gateway_unreachable(last_error):
        status_code = _extract_status_code(last_error)
        if outage:
            should_try_explicit_fallback = True
        elif not use_native_fallbacks and status_code in FALLBACK_STATUS_CODES:
            should_try_explicit_fallback = True

    if should_try_explicit_fallback:
        for fb_idx, fb_model in enumerate(fallback_models):
            remaining_fallbacks = fallback_models[fb_idx + 1 :]
            try:
                response, headers = _invoke_chat_completion(
                    client=client,
                    model=fb_model,
                    messages=messages,
                    temperature=temperature,
                    fallbacks=remaining_fallbacks,
                    use_native_fallbacks=bool(remaining_fallbacks) and not is_direct_endpoint,
                )
                content = _extract_content(response)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                meta = _extract_response_metadata(
                    response=response,
                    raw_headers=headers,
                    original_primary=original_primary,
                    requested_model=request_model,
                    fallback_models=fallback_models,
                    elapsed_ms=elapsed_ms,
                    simulated_outage=outage,
                    forced_fallback_used=True,
                    fallback_target=fb_model,
                    routing_mode=routing_mode,
                )
                _record_gateway_metrics(meta)
                return content, meta
            except Exception as exc:
                last_error = exc
                status_code = _extract_status_code(exc)
                if fb_idx < len(fallback_models) - 1 and (outage or status_code in FALLBACK_STATUS_CODES):
                    continue
                break

    endpoint_label = base_url or "https://api.openai.com/v1"
    friendly_error = format_llm_error(
        last_error or RuntimeError("Unknown LLM error"),
        use_bifrost=(not is_direct_endpoint),
        base_url=endpoint_label,
        primary_model=original_primary,
    )
    raise RuntimeError(friendly_error) from last_error


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
    simulate_outage: bool | None = None,
) -> tuple[str, str]:
    """
    Send a chat completion request using Bifrost native fallbacks and return
    `(response_text, model_that_answered)`.

    Per-request telemetry is also stored via `pop_last_gateway_metrics()` so callers
    can read latency, token usage, and fallback status without breaking 2-tuple unpacking.
    """
    content, meta = call_llm_with_meta(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        simulate_outage=simulate_outage,
    )
    return content, meta["model_used"]


QA_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer the user's question directly and clearly "
    "with specific factual details (such as prices, versions, dates, company leadership, "
    "laws, or statistics) in 1 to 3 concise plain-text paragraphs without markdown headings or bullet formatting."
)


def ask_question(question: str, temperature: float = 0.3) -> tuple[str, dict]:
    """
    Ask the LLM a question via Bifrost (supporting native fallbacks and outage simulation)
    and return `(answer_text, telemetry_meta)`.

    Raises:
        ValueError: If question is blank.
        RuntimeError: If the gateway/LLM call fails.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Please enter a question before clicking Ask & Audit.")

    pop_last_gateway_metrics()
    t0 = time.perf_counter()
    answer_text, model_used = call_llm(
        system_prompt=QA_SYSTEM_PROMPT,
        user_prompt=cleaned,
        temperature=temperature,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    meta = pop_last_gateway_metrics()
    if not isinstance(meta, dict):
        meta = {
            "primary_model": model_used,
            "requested_model": model_used,
            "model_used": model_used,
            "provider_used": model_used.split("/", 1)[0] if "/" in str(model_used) else None,
            "fallback_models": [],
            "fallback_used": False,
            "latency_ms": elapsed_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "simulated_outage": is_simulate_outage(),
            "routing_mode": "bifrost" if is_use_bifrost() else "direct_gemini",
            "timestamp": time.strftime("%H:%M:%S"),
        }
    meta["operation"] = "Ask & Audit Q&A"
    return answer_text, meta


def parse_json_response(raw_text: str) -> Any:
    """
    Parse JSON from an LLM response, even if wrapped in markdown fences or extra text.

    Args:
        raw_text: Raw string returned by the LLM.

    Returns:
        The parsed Python object (dict or list).

    Raises:
        ValueError: If valid JSON cannot be extracted from raw_text.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot parse JSON from an empty LLM response.")

    cleaned = raw_text.strip()

    # Remove markdown code fences like ```json ... ``` or ``` ... ```.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Fast path: the entire cleaned string is valid JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: locate the first '[' or '{' and decode one JSON value from there.
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch in ("[", "{"):
            try:
                parsed, _ = decoder.raw_decode(cleaned, idx)
                return parsed
            except json.JSONDecodeError:
                continue

    preview = raw_text[:200].replace("\n", " ")
    raise ValueError(f"LLM did not return valid JSON. Response started with: {preview!r}")
