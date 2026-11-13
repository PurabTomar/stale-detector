"""
extractor.py - Extracts potentially outdated factual claims from an LLM answer or article.

Uses the configured LLM to scan text and return a structured list of claims, each
tagged with an exact verbatim source span, a category, and a targeted web search
query for live verification.
"""

import inspect
import re
import threading
import time

from llm import call_llm, parse_json_response, pop_last_gateway_metrics

_extractor_state = threading.local()


def get_last_gateway_metrics() -> dict | None:
    """Return telemetry recorded during the most recent `extract_claims` call."""
    return getattr(_extractor_state, "last_metrics", None)


# Allowed categories for extracted claims.
VALID_CATEGORIES = (
    "price",
    "version",
    "regulation",
    "company",
    "statistic",
    "other",
)

EXTRACTION_SYSTEM_PROMPT = """You are a careful fact-checking assistant.
Your job is to inspect an LLM-generated answer or article and extract specific,
independently verifiable factual claims that could become outdated over time
OR claims about CURRENT FACTS that can simply be wrong (for example: who holds an
office such as Prime Minister, President, CEO, Cabinet Minister; capital cities;
records; "latest" statements; product or API/software versions; prices; laws or
regulations; company info; and statistics), even if they are not strictly time-sensitive.

Skip conversational filler, greetings, disclaimers, purely subjective opinions,
vague recommendations without concrete facts, and duplicate paraphrases of a
claim already extracted. Do not split one compound figure unnecessarily, and do
not invent details absent from the input text.

Return ONLY a valid JSON array (no markdown explanation, no extra text).
Each element in the JSON array must be an object with these exact keys:
- "claim": Keep the claim text EXACTLY as in the input text so the annotation
  span matches (copy the exact "span" whenever it already stands on its own; only
  replace an ambiguous pronoun like "it" or "their" with the named entity so
  the claim makes sense alone). NEVER rewrite, sanitize, or paraphrase a specific
  fact, office holder, product, version, price, or entity claim into a vaguer statement.
- "span": The exact verbatim substring copied character-for-character from the input
  text that states this claim (preserve exact casing, numbers, punctuation, version
  dots, and currency symbols so it can be located with a string match).
- "category": Must be one of: "price", "version", "regulation", "company", "statistic", "other".
  Product-name, hardware/software model, release, and availability claims (such as
  "iPhone 18 Pro is available", "Python 3.15", or "GPT-5") MUST use "version"
  (or "price" if stating a price, or "company" if about leadership/ownership/corporate
  lineup) — NEVER label named product claims as "other". Office holder or company leadership
  claims should use "company" or "other".
- "search_query": A concise, targeted web search query to check the current truth of this claim.

If the text contains no time-sensitive or verifiable factual claims, return [].
"""


_PRODUCT_VERSION_TERMS = (
    "iphone",
    "ipad",
    "macbook",
    "imac",
    "airpods",
    "apple watch",
    "vision pro",
    "ios",
    "ipados",
    "macos",
    "watchos",
    "tvos",
    "visionos",
    "pixel",
    "galaxy",
    "android",
    "windows",
    "surface",
    "playstation",
    "ps5",
    "xbox",
    "nintendo switch",
    "switch 2",
    "steam deck",
    "gpt",
    "chatgpt",
    "claude",
    "gemini",
    "llama",
    "mistral",
    "copilot",
    "python",
    "react",
    "next.js",
    "nextjs",
    "node.js",
    "nodejs",
    "typescript",
    "golang",
    "rust",
    "kubernetes",
    "docker",
    "pytorch",
    "tensorflow",
    "streamlit",
    "fastapi",
    "cuda",
)

_KNOWN_COMPANY_TERMS = (
    "apple",
    "openai",
    "anthropic",
    "google",
    "alphabet",
    "microsoft",
    "nvidia",
    "tesla",
    "meta",
    "amazon",
    "aws",
    "stripe",
    "netflix",
    "disney",
    "starbucks",
    "github",
    "twitter",
    "xai",
)

_CORPORATE_ACTION_RE = re.compile(
    r"\b(?:ceo|chief\s+executive|cfo|cto|coo|chairman|president|prime\s+minister|minister|"
    r"chancellor|governor|mayor|founder|co-founder|"
    r"acquired|acquisition|merged|merger|subsidiary|headquarter(?:ed|s)?|ipo|valuation|"
    r"board\s+of\s+directors|stepped\s+down|appointed)\b",
    re.IGNORECASE,
)

_MODEL_VERSION_RE = re.compile(
    r"(?:\b(?:v|r)\d+(?:\.\d+)+\b|"
    r"\b[A-Za-z][A-Za-z0-9.+-]{1,20}[\s\-]+(?:(?:Pro|Max|Ultra|Plus|Mini|Air|SE|Studio|Turbo)\s+)*\d+(?:\.\d+)?(?:\s+(?:Pro|Max|Ultra|Plus|Mini|Air|SE|Studio|Turbo))?\b)",
    re.IGNORECASE,
)

_RELEASE_CONTEXT_RE = re.compile(
    r"\b(?:available|released|release|launched|launch|announced|ships|shipping|"
    r"version|update|latest|flagship|lineup|series|edition|on\s+sale)\b",
    re.IGNORECASE,
)

_LEADING_PRONOUN_RE = re.compile(
    r"^(?:it|its|they|their|this|that|these|those|he|she|his|her)\b",
    re.IGNORECASE,
)


def _infer_product_or_company_category(
    claim_text: str,
    span_text: str | None = None,
) -> str:
    """
    Reclassify named product/model/company claims away from 'other' when the
    extractor LLM mislabels them as 'other'.
    """
    combined = " ".join(f"{claim_text} {span_text or ''}".split()).strip()
    if not combined:
        return "other"
    lower = combined.lower()

    has_product_term = any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower)
        for term in _PRODUCT_VERSION_TERMS
    )
    has_company_term = any(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lower)
        for term in _KNOWN_COMPANY_TERMS
    )
    has_corp_action = bool(_CORPORATE_ACTION_RE.search(combined))
    has_model_ver = bool(_MODEL_VERSION_RE.search(combined))
    has_release_ctx = bool(_RELEASE_CONTEXT_RE.search(combined))

    if has_corp_action:
        return "company"
    if has_product_term:
        return "version"
    if has_model_ver and (has_release_ctx or has_company_term):
        return "version"
    if has_company_term:
        return "company"
    return "other"


def _extract_word_tokens_for_fidelity(text: str) -> set[str]:
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return {tok for tok in re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", cleaned) if len(tok) >= 2 or tok.isdigit()}


def _preserve_exact_claim_wording(claim_text: str, span_str: str | None) -> str:
    """
    Keep normalized `claim` anchored closely to the user's exact span wording so
    an extractor cannot rewrite specific facts (e.g. 'iPhone 18 Pro is available')
    into vaguer paraphrases.
    """
    if not span_str:
        return claim_text
    clean_span = " ".join(span_str.split()).strip()
    if len(clean_span) < 6:
        return claim_text

    # Do not replace with a lowercase clause fragment (e.g. "starts at ..."), while still
    # preserving camelCase product names like iPhone, iPad, iOS, macOS, watchOS.
    first_word_match = re.match(r"^[A-Za-z]+", clean_span)
    first_word = first_word_match.group(0) if first_word_match else ""
    is_camel_brand = bool(re.match(r"^(?:i[A-Z]|e[A-Z]|mac[A-Z]|watch[A-Z]|tv[A-Z]|vision[A-Z])", first_word))
    if (first_word and first_word[0].islower() and not is_camel_brand) or _LEADING_PRONOUN_RE.match(clean_span):
        return claim_text

    # If claim and span only differ by trailing punctuation/whitespace, preserve claim as-is.
    if clean_span.rstrip(".!?").lower() == " ".join(claim_text.split()).rstrip(".!?").lower():
        return claim_text

    span_tokens = _extract_word_tokens_for_fidelity(clean_span)
    claim_tokens = _extract_word_tokens_for_fidelity(claim_text)
    if not span_tokens:
        return claim_text

    span_num_tokens = {t for t in span_tokens if any(c.isdigit() for c in t)}
    dropped_numbers = span_num_tokens - claim_tokens
    overlap_ratio = len(span_tokens & claim_tokens) / max(1, len(span_tokens))

    if dropped_numbers or overlap_ratio < 0.70:
        if claim_text.endswith(".") and not clean_span.endswith((".", "!", "?")):
            return f"{clean_span}."
        return clean_span

    return claim_text


def _range_overlaps(
    start: int,
    end: int,
    occupied_ranges: list[tuple[int, int]] | None,
) -> bool:
    """Return True if `[start, end)` overlaps any span in `occupied_ranges`."""
    if not occupied_ranges:
        return False
    return any(start < prev_end and end > prev_start for prev_start, prev_end in occupied_ranges)


def _find_literal_variants(
    source_text: str,
    candidate: str,
    occupied_ranges: list[tuple[int, int]] | None = None,
) -> tuple[int, int, str] | None:
    """
    Locate `candidate` inside `source_text` via string matching:
    1. Exact substring (`str.find`)
    2. Case-insensitive substring (`str.lower().find`)
    3. Whitespace-collapsed regex match (exact case, then case-insensitive)
    Skipping any occurrences that overlap `occupied_ranges`.
    """
    if not source_text or not candidate:
        return None

    clean_cand = candidate.strip()
    if len(clean_cand) < 2:
        return None

    variants = [clean_cand]
    unquoted = clean_cand.strip("\"'` ")
    if unquoted and unquoted not in variants:
        variants.append(unquoted)
    no_trail = unquoted.rstrip(".!?;,") if unquoted else ""
    if len(no_trail) >= 2 and no_trail not in variants:
        variants.append(no_trail)

    lower_source = source_text.lower()

    # 1. Exact character-for-character substring search (`str.find`).
    for var in variants:
        search_from = 0
        while search_from < len(source_text):
            pos = source_text.find(var, search_from)
            if pos == -1:
                break
            end = pos + len(var)
            if not _range_overlaps(pos, end, occupied_ranges):
                return pos, end, source_text[pos:end]
            search_from = pos + 1

    # 2. Case-insensitive literal substring search.
    for var in variants:
        lower_var = var.lower()
        search_from = 0
        while search_from < len(lower_source):
            pos = lower_source.find(lower_var, search_from)
            if pos == -1:
                break
            end = pos + len(var)
            if not _range_overlaps(pos, end, occupied_ranges):
                return pos, end, source_text[pos:end]
            search_from = pos + 1

    # 3. Whitespace-normalized string search (exact case first, then case-insensitive).
    for var in variants:
        words = var.split()
        if not words:
            continue
        pattern = r"\s+".join(re.escape(w) for w in words)
        for flags in (0, re.IGNORECASE):
            for match in re.finditer(pattern, source_text, flags=flags):
                s, e = match.start(), match.end()
                if e > s and not _range_overlaps(s, e, occupied_ranges):
                    return s, e, source_text[s:e]

    return None


def locate_span_in_text(
    source_text: str,
    span_text: str | None = None,
    claim_text: str | None = None,
    occupied_ranges: list[tuple[int, int]] | None = None,
) -> tuple[int, int, str] | None:
    """
    Locate a claim's exact span in `source_text` via string matching.

    Checks `span_text` (exact verbatim substring returned by the extractor) first,
    then falls back to direct string matching of `claim_text`.
    Returns `(start, end, matched_substring)` or `None`.
    """
    if not source_text:
        return None

    if span_text and span_text.strip():
        found = _find_literal_variants(
            source_text,
            span_text,
            occupied_ranges=occupied_ranges,
        )
        if found is not None:
            return found

    if claim_text and claim_text.strip() and len(claim_text.strip()) >= 4:
        found = _find_literal_variants(
            source_text,
            claim_text,
            occupied_ranges=occupied_ranges,
        )
        if found is not None:
            return found

    return None


def _normalize_claims(
    parsed_data: object,
    source_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Validate and normalize the JSON returned by the LLM into a clean list of claim dicts.
    """
    # Some models wrap a JSON list inside an object like {"claims": [...]}.
    if isinstance(parsed_data, dict):
        for value in parsed_data.values():
            if isinstance(value, list):
                parsed_data = value
                break

    if not isinstance(parsed_data, list):
        raise ValueError(
            "Expected the LLM to return a JSON list of claims, "
            f"but got {type(parsed_data).__name__}."
        )

    cleaned_claims: list[dict[str, str]] = []
    occupied_ranges: list[tuple[int, int]] = []
    seen_claim_keys: set[tuple[str, str]] = set()

    for item in parsed_data:
        if not isinstance(item, dict):
            continue

        claim_text = str(item.get("claim", "")).strip()
        if not claim_text:
            continue

        raw_category = str(item.get("category", "other")).strip().lower()
        category = raw_category if raw_category in VALID_CATEGORIES else "other"
        raw_span = str(item.get("span") or "").strip()

        matched_span: str | None = None
        if source_text:
            located = locate_span_in_text(
                source_text,
                span_text=raw_span or None,
                claim_text=claim_text,
                occupied_ranges=occupied_ranges,
            )
            if located is not None:
                occupied_ranges.append((located[0], located[1]))
                matched_span = located[2]
            elif raw_span:
                matched_span = raw_span
        elif raw_span:
            matched_span = raw_span

        claim_text = _preserve_exact_claim_wording(claim_text, matched_span)
        if category == "other":
            category = _infer_product_or_company_category(claim_text, matched_span)

        norm_claim_key = (" ".join(claim_text.lower().split()).rstrip(".!?"), category)
        if norm_claim_key in seen_claim_keys:
            continue
        seen_claim_keys.add(norm_claim_key)

        search_query = str(item.get("search_query", "")).strip() or claim_text

        entry: dict[str, str] = {
            "claim": claim_text,
            "category": category,
            "search_query": search_query,
        }
        if matched_span:
            entry["span"] = matched_span

        cleaned_claims.append(entry)

    return cleaned_claims


def _invoke_extract_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """Call `llm.call_llm(..., task="extract")` while staying safe with 3-arg test mocks."""
    side_effect = getattr(call_llm, "side_effect", None)
    sig_target = side_effect if callable(side_effect) else call_llm
    kwargs: dict = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
    }
    try:
        params = inspect.signature(sig_target).parameters
        accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params = {}
        accepts_varkw = True

    if accepts_varkw or "task" in params:
        kwargs["task"] = "extract"

    try:
        return call_llm(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )


def extract_claims(answer_text: str) -> tuple[list[dict[str, str]], str]:
    """
    Extract potentially stale claims from a pasted LLM answer or article text.

    Uses `LLM_MODEL_EXTRACT` (fast/cheap model) via task-based routing when
    configured, falling back to `LLM_MODEL`.

    Args:
        answer_text: The full LLM-generated text or article body to inspect.

    Returns:
        A tuple of (claims_list, model_used), where claims_list contains dicts with:
        - claim (str)
        - category (str): one of price, version, regulation, company, statistic, other
        - search_query (str)
        - span (str, optional): exact substring in `answer_text` when matched/returned

    Raises:
        ValueError: If the LLM returns invalid/malformed JSON.
        RuntimeError: If the LLM API call fails even after retries/fallbacks.
    """
    if not answer_text or not answer_text.strip():
        _extractor_state.last_metrics = None
        return [], ""

    user_prompt = (
        "Extract all potentially outdated or time-sensitive factual claims from "
        "the following text (include the exact verbatim \"span\" copied from the text):\n\n"
        f"{answer_text.strip()}"
    )

    pop_last_gateway_metrics()
    t0 = time.perf_counter()
    raw_response, model_used = _invoke_extract_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    metrics = pop_last_gateway_metrics()
    if metrics is None:
        metrics = {
            "task": "extract",
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
            "simulated_outage": False,
            "timestamp": time.strftime("%H:%M:%S"),
        }
    metrics["task"] = metrics.get("task") or "extract"
    metrics["operation"] = "Extract claims"
    _extractor_state.last_metrics = metrics

    # Parse and validate the JSON list returned by the LLM.
    parsed = parse_json_response(raw_response)
    return _normalize_claims(parsed, source_text=answer_text), model_used
