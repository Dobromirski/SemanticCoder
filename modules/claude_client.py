"""Thin wrapper over the Anthropic API with retry logic and model switching."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import anthropic


MODEL_OPUS = "claude-opus-4-20250514"
MODEL_SONNET = "claude-sonnet-4-20250514"


class ClaudeClient:
    def __init__(self, api_key: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        system: str,
        user: str,
        model: str = MODEL_SONNET,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> str:
        """Single API call with exponential backoff."""
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return resp.content[0].text
            except anthropic.RateLimitError as exc:
                wait = getattr(exc, "retry_after", None) or (2 ** attempt * 2)
                time.sleep(wait)
                last_err = exc
            except anthropic.APIError as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
                last_err = exc
        raise last_err  # type: ignore[misc]

    def call_json(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> dict | list:
        """Call the API and parse JSON from the response.

        Handles the common case where Claude wraps JSON in ```json``` blocks.
        """
        raw = self.call(system, user, **kwargs)
        return parse_json(raw)


def parse_json(raw: str) -> dict | list:
    """Extract JSON from a string, tolerating markdown fences."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", raw)
    if match:
        return json.loads(match.group(1).strip())
    # Try finding a top-level [ ] or { }
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"Could not extract JSON from response:\n{raw[:200]}...")
