"""One bounded, non-streaming model call for the query-rewriting modes (decomposition, HyDE).

Uses the existing GENERATION_* settings (OpenAI-compatible proxy). One attempt, no retry. Missing
configuration or any provider failure is `RetrievalUnavailable`; the key, headers and provider
body are never returned. Empty or malformed *text* is the calling mode's decision.
"""

from __future__ import annotations

import httpx

from ..config import Settings
from .semantic import RetrievalUnavailable

REQUEST_TIMEOUT_SECONDS = 30

# Test seam: a transport used when none is passed (None means the real network).
DEFAULT_TRANSPORT: httpx.BaseTransport | None = None


def check_generation_config(settings: Settings) -> None:
    """Fail clearly before any retrieval work. Messages name variables, never values."""
    for name, value in (("GENERATION_API_BASE_URL", settings.generation_api_base_url),
                        ("GENERATION_API_KEY", settings.generation_api_key),
                        ("GENERATION_MODEL_NAME", settings.generation_model_name)):
        if not value.strip():
            raise RetrievalUnavailable(f"{name} is not set.")


def _fail(reason: str) -> RetrievalUnavailable:
    return RetrievalUnavailable(f"Query rewriting model step failed ({reason}).")


def complete_text(system: str, question: str, settings: Settings,
                  transport: httpx.BaseTransport | None = None) -> str:
    """One chat call. The question is the only variable input, passed as delimited data."""
    check_generation_config(settings)
    url = settings.generation_api_base_url.rstrip("/") + "/chat/completions"
    body = {"model": settings.generation_model_name, "temperature": 0, "stream": False,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": f"<question>\n{question}\n</question>"}]}
    headers = {"Authorization": f"Bearer {settings.generation_api_key}"}
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS,
                          transport=transport or DEFAULT_TRANSPORT) as client:
            res = client.post(url, json=body, headers=headers)
    except httpx.TimeoutException:
        raise _fail("timeout") from None
    except (httpx.HTTPError, httpx.InvalidURL):
        raise _fail("connection error") from None
    if not res.is_success:
        raise _fail(f"HTTP {res.status_code}")
    try:
        content = res.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise _fail("unusable reply") from None
    if not isinstance(content, str):
        raise _fail("unusable reply")
    return content
