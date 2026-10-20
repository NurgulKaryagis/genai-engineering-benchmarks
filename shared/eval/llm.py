"""Thin OpenAI chat-completions wrapper that records tokens, latency and cost."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# USD per 1M tokens: (input, cached input, output).
# Source: https://developers.openai.com/api/docs/pricing, read 2026-09-30.
PRICES = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
}

MAX_ATTEMPTS = 3
TIMEOUT_S = 60.0
RETRYABLE = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str  # the alias that was requested, e.g. "gpt-4o-mini"
    model_id: str  # the dated model ID the API reports, for reproducibility
    input_tokens: int  # includes cached tokens
    cached_tokens: int
    output_tokens: int
    latency_s: float  # of the successful attempt only
    cost_usd: float
    attempts: int
    finish_reason: str


@cache
def get_client() -> OpenAI:
    # Retries are handled in call_llm so that failed attempts can be logged.
    return OpenAI(max_retries=0, timeout=TIMEOUT_S)


def compute_cost(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float:
    price_in, price_cached, price_out = PRICES[model]
    uncached = input_tokens - cached_tokens
    return (
        uncached * price_in + cached_tokens * price_cached + output_tokens * price_out
    ) / 1_000_000


def _append_log(log_path: Path | None, record: dict) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    log_path: Path | None = None,
    tag: dict | None = None,
) -> LLMResult:
    """Call the model, retrying transient errors, and append one JSONL log record.

    Failed attempts are logged too. If every attempt fails, the last error is raised.
    """
    if model not in PRICES:
        raise ValueError(f"no price known for model {model!r}")
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if seed is not None:
        kwargs["seed"] = seed
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            response = get_client().chat.completions.create(**kwargs)
        except RETRYABLE as error:
            _append_log(
                log_path,
                {"model": model, "attempt": attempt, "error": repr(error), "tag": tag},
            )
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2**attempt)
            continue
        latency = time.perf_counter() - start
        break

    usage = response.usage
    details = usage.prompt_tokens_details
    cached = details.cached_tokens if details and details.cached_tokens else 0
    choice = response.choices[0]
    result = LLMResult(
        text=choice.message.content or "",
        model=model,
        model_id=response.model,
        input_tokens=usage.prompt_tokens,
        cached_tokens=cached,
        output_tokens=usage.completion_tokens,
        latency_s=latency,
        cost_usd=compute_cost(
            model, usage.prompt_tokens, cached, usage.completion_tokens
        ),
        attempts=attempt,
        finish_reason=choice.finish_reason,
    )
    _append_log(
        log_path,
        {
            **asdict(result),
            "temperature": temperature,
            "seed": seed,
            "messages": messages,
            "tag": tag,
        },
    )
    return result
