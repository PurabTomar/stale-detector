"""
llm.py - OpenAI-compatible LLM client wrapper with model fallback chain.

Reads LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, and LLM_MODEL_FALLBACKS from .env
so you can switch between OpenAI and a Bifrost gateway (or any OpenAI-compatible
proxy) without changing any Python code.

Fallback behavior:
- Each model gets 1 retry if its first call fails.
- If a model still fails with HTTP 429, 500, 503, or 404 after that retry,
  call_llm automatically moves on to the next model in LLM_MODEL_FALLBACKS.
"""

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a local .env file if one exists.
load_dotenv()

# HTTP status codes that trigger moving to the next fallback model.
FALLBACK_STATUS_CODES = {429, 500, 503, 404}


def get_llm_config() -> tuple[str, str | None, list[str]]:
    """
    Read LLM settings from environment variables (.env).

    Returns:
        A tuple of (api_key, base_url, model_chain), where model_chain starts
        with LLM_MODEL followed by any models in LLM_MODEL_FALLBACKS.

    Raises:
        ValueError: If LLM_API_KEY is missing.
    """
    # Reload .env in case the user edited it while the app is running.
    load_dotenv(override=False)

    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    primary_model = os.getenv("LLM_MODEL", "").strip() or "gpt-4o-mini"

    # Parse comma-separated fallback models, skipping blanks and duplicates.
    raw_fallbacks = os.getenv("LLM_MODEL_FALLBACKS", "")
    fallback_models = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]

    model_chain: list[str] = []
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
    Create and return an OpenAI SDK client configured from .env.

    Returns:
        A tuple of (OpenAI client, model_chain list, base_url).
    """
    api_key, base_url, model_chain = get_llm_config()

    # Optional Bifrost virtual key; sent via the x-bf-vk header only if set.
    virtual_key = os.getenv("LLM_VIRTUAL_KEY", "").strip()
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

    # Fallback: check the error message text for 404, 429, 500, or 503.
    match = re.search(r"\b(404|429|500|503)\b", str(exc))
    if match:
        return int(match.group(1))

    return None


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """
    Send a chat completion request using the configured model fallback chain.

    For each model in [LLM_MODEL, *LLM_MODEL_FALLBACKS]:
    1. Try once; if it fails, wait briefly and retry once.
    2. If it still fails with 429, 500, 503, or 404 and another fallback model
       is configured, automatically try the next model in the chain.

    Args:
        system_prompt: Instructions defining the LLM's role and output format.
        user_prompt: The task input for the LLM.
        temperature: Sampling temperature (lower values give more consistent JSON).

    Returns:
        A tuple of (response_text, model_that_answered).

    Raises:
        ValueError: If required credentials (like LLM_API_KEY) are missing.
        RuntimeError: If the model(s) fail and cannot fall back further.
    """
    client, model_chain, base_url = get_llm_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    errors_by_model: list[str] = []
    max_attempts_per_model = 2  # 1 initial attempt + 1 retry

    for model_idx, model in enumerate(model_chain):
        for attempt in range(1, max_attempts_per_model + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

                # Check that the API returned a non-empty message.
                if not response.choices:
                    raise ValueError("LLM returned no choices in the response.")

                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("LLM returned an empty message.")

                return content.strip(), model

            except Exception as exc:
                last_error = exc
                if attempt < max_attempts_per_model:
                    # Brief pause before retrying the same model once.
                    time.sleep(1)

        # Both attempts on this model failed; decide whether to try the next fallback.
        errors_by_model.append(f"{model}: {last_error}")
        status_code = _extract_status_code(last_error) if last_error else None
        has_next_model = model_idx < len(model_chain) - 1

        if has_next_model and status_code in FALLBACK_STATUS_CODES:
            continue

        # Either the error isn't one of 429/500/503/404, or we ran out of models.
        break

    endpoint_label = base_url or "https://api.openai.com/v1"
    error_summary = " | ".join(errors_by_model)
    raise RuntimeError(
        f"LLM call failed (base_url='{endpoint_label}', models_tried={model_chain}): {error_summary}"
    ) from last_error


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
