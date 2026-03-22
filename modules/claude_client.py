"""Claude Code CLI wrapper for Max subscription — no API key needed.

Calls the `claude` CLI as a subprocess, passing prompts via stdin.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any


MODEL_OPUS = "opus"
MODEL_SONNET = "sonnet"


def _find_claude_cli() -> str:
    """Find the claude CLI executable, checking common locations."""
    # Try PATH first
    found = shutil.which("claude")
    if found:
        return found

    # Check common npm global install locations (Windows)
    candidates = [
        os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
        os.path.expandvars(r"%APPDATA%\npm\claude"),
        os.path.expanduser("~/.npm-global/bin/claude"),
        "/usr/local/bin/claude",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    raise RuntimeError(
        "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
    )


class ClaudeClient:
    """Wrapper that calls Claude Code CLI in non-interactive mode."""

    def __init__(self) -> None:
        self._cli_path = _find_claude_cli()

    def call(
        self,
        system: str,
        user: str,
        model: str = MODEL_SONNET,
        retries: int = 3,
        timeout: int = 600,
        **_kwargs: Any,
    ) -> str:
        """Single CLI call with retry logic."""
        retry_delay = 10
        last_err: Exception | None = None

        for attempt in range(retries):
            try:
                cmd = [
                    self._cli_path,
                    "-p",
                    "--output-format", "text",
                    "--no-session-persistence",
                    "--model", model,
                    "--disallowed-tools",
                    "Edit,Write,Bash,NotebookEdit,Glob,Grep,WebSearch,WebFetch",
                    "--system-prompt", system,
                ]

                # Remove CLAUDECODE env var to allow nested calls
                env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

                result = subprocess.run(
                    cmd,
                    input=user,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=timeout,
                    env=env,
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else f"Exit code {result.returncode}"
                    raise RuntimeError(f"Claude CLI error: {error_msg}")

                response = result.stdout.strip()
                if not response:
                    raise RuntimeError("Claude CLI returned empty response")

                return response

            except subprocess.TimeoutExpired:
                last_err = RuntimeError(f"Claude CLI timeout (attempt {attempt + 1})")
                if attempt < retries - 1:
                    time.sleep(retry_delay)
                    continue

            except RuntimeError as exc:
                last_err = exc
                if attempt < retries - 1:
                    print(
                        f"  Attempt {attempt + 1} failed, retrying in {retry_delay}s...",
                        file=sys.stderr,
                    )
                    time.sleep(retry_delay)
                    continue

        raise last_err  # type: ignore[misc]

    def call_json(
        self,
        system: str,
        user: str,
        **kwargs: Any,
    ) -> dict | list:
        """Call CLI and parse JSON from the response."""
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
    raise ValueError(f"Could not extract JSON from response:\n{raw[:300]}...")
