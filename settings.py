"""
settings.py - Centralized configuration, input limits, and security validation.

Configurable limits:
- MAX_TEXT_CHARS: Maximum characters allowed for text input (default: 6000).
- MAX_CLAIMS_PER_AUDIT: Maximum claims extracted/verified per audit (default: 8).
- DEFAULT_MAX_CLAIMS_PER_AUDIT: Default claims cap when unconfigured (default: 6).
- DEFAULT_MAX_SESSION_AUDITS: Maximum audits allowed per session (default: 5).
- SSRF / Local IP filtering for URLs.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import urlparse

# Limits configurable via environment variables or sensible defaults
MAX_TEXT_CHARS: int = int(os.environ.get("MAX_TEXT_CHARS", "6000"))
MAX_CLAIMS_PER_AUDIT: int = int(os.environ.get("MAX_CLAIMS_PER_AUDIT", "8"))
DEFAULT_MAX_CLAIMS_PER_AUDIT: int = 6
DEFAULT_MAX_SESSION_AUDITS: int = int(os.environ.get("MAX_SESSION_AUDITS", "5"))
MAX_QUESTION_CHARS: int = int(os.environ.get("MAX_QUESTION_CHARS", "500"))
MAX_URL_CHARS: int = 2048


def validate_text_input(text: str | None) -> tuple[bool, str | None]:
    """Validate user-pasted text input.

    Returns:
        `(is_valid, friendly_error_message)`
    """
    if not text or not str(text).strip():
        return False, "Please paste an LLM answer or enter text to audit."

    clean = str(text).strip()
    if len(clean) > MAX_TEXT_CHARS:
        return (
            False,
            f"Input text exceeds the maximum limit of {MAX_TEXT_CHARS:,} characters "
            f"(received {len(clean):,} characters). Please shorten the text to proceed.",
        )
    return True, None


def validate_question_input(question: str | None) -> tuple[bool, str | None]:
    """Validate a question query.

    Returns:
        `(is_valid, friendly_error_message)`
    """
    if not question or not str(question).strip():
        return False, "Please enter a question to ask and audit."

    clean = str(question).strip()
    if len(clean) > MAX_QUESTION_CHARS:
        return (
            False,
            f"Question exceeds the maximum limit of {MAX_QUESTION_CHARS} characters "
            f"(received {len(clean)} characters). Please enter a concise question.",
        )
    return True, None


def is_private_or_local_host(hostname: str) -> bool:
    """Return True if hostname resolves to private, loopback, link-local, or reserved IP address."""
    clean_host = hostname.strip().lower().strip("[]")
    if not clean_host:
        return True

    # Common local/private host aliases
    if clean_host in {"localhost", "localhost.localdomain", "broadcasthost", "0.0.0.0"}:
        return True

    # Try parsing direct IP
    try:
        ip = ipaddress.ip_address(clean_host)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass

    # Resolve DNS to check resulting IP addresses
    try:
        addr_infos = socket.getaddrinfo(clean_host, None)
        for _, _, _, _, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
                if (
                    ip.is_private
                    or ip.is_loopback
                    or ip.is_link_local
                    or ip.is_multicast
                    or ip.is_reserved
                    or ip.is_unspecified
                ):
                    return True
            except ValueError:
                continue
    except socket.gaierror:
        # DNS resolution failure will be caught as connection error downstream
        return False
    except Exception:
        return True

    return False


def validate_url_input(url: str | None) -> tuple[bool, str | None]:
    """Validate a URL input for auditing.

    Enforces:
    1. Not empty and length <= MAX_URL_CHARS
    2. Must use http or https scheme
    3. Hostname cannot resolve to private, local, or loopback address (SSRF guard)

    Returns:
        `(is_valid, friendly_error_message)`
    """
    if not url or not str(url).strip():
        return False, "Please enter a valid webpage URL to audit."

    clean_url = str(url).strip()
    if len(clean_url) > MAX_URL_CHARS:
        return False, f"URL exceeds maximum length of {MAX_URL_CHARS} characters."

    try:
        parsed = urlparse(clean_url)
    except Exception:
        return False, "Invalid URL format. Please provide a full URL starting with http:// or https://."

    if parsed.scheme.lower() not in {"http", "https"}:
        return False, "Only HTTP and HTTPS URLs are supported (e.g. https://example.com/article)."

    if not parsed.netloc:
        return False, "Invalid URL: missing domain name or host."

    hostname = parsed.hostname or ""
    if not hostname:
        return False, "Invalid URL: unable to extract domain name."

    if is_private_or_local_host(hostname):
        return (
            False,
            f"Private or local network addresses ('{hostname}') cannot be audited for security reasons. "
            "Please provide a public webpage URL.",
        )

    return True, None
