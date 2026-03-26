"""
Shared Groq LLM factory with retry and rate-limit handling.

Free tier limits for openai/gpt-oss-20b:
  - 30 RPM  (1 request every 2 seconds - conservative)
  - 6,000 TPM
  - 14,400 RPD

Strategy:
  - ChatGroq built-in max_retries handles transient network errors
  - tenacity exponential backoff handles 429 rate-limit responses
  - max_tokens capped per call to stay within TPM budget
"""

import os
import time
import logging
from typing import Any

from langchain_groq import ChatGroq
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger("code_synthesis.groq")

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0

# Minimum seconds between calls to stay under 30 RPM (conservative: 2.1s)
_MIN_CALL_INTERVAL = 2.1
_last_call_time: float = 0.0


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True for Groq 429 / rate-limit errors."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "too many requests" in msg
    )


def build_groq_model(
    *,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
) -> ChatGroq:
    """
    Build a ChatGroq instance.

    The GROQ_API_KEY environment variable must be set before calling this.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set. "
            "Get a free key at https://console.groq.com/"
        )

    kwargs: dict[str, Any] = dict(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,  # built-in retry for transient errors
        timeout=60,
        groq_api_key=api_key,
    )

    if json_mode:
        # Groq supports OpenAI-compatible JSON mode
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatGroq(**kwargs)


def _throttle() -> None:
    """Sleep if needed to respect the 30 RPM limit."""
    global _last_call_time
    elapsed = time.monotonic() - _last_call_time
    if elapsed < _MIN_CALL_INTERVAL:
        time.sleep(_MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.monotonic()


@retry(
    retry=retry_if_exception(_is_rate_limit_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def invoke_with_retry(model: ChatGroq, messages: list) -> Any:
    """
    Invoke a ChatGroq model with exponential backoff on rate-limit errors.

    Usage:
        model = build_groq_model()
        response = invoke_with_retry(model, [SystemMessage(...), HumanMessage(...)])
        text = response.content
    """
    _throttle()
    return model.invoke(messages)
