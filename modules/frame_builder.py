"""Iterative 3-pass coding frame generation using Claude."""

from __future__ import annotations

import json
from typing import Callable

from .claude_client import ClaudeClient, MODEL_OPUS
from .prompts import (
    SYSTEM_FRAME_BUILDER,
    SYSTEM_FRAME_REFINER,
    BASE_FRAME_INSTRUCTION,
    BASE_FRAME_INSTRUCTION_NONE,
)


def _normalize_frame(raw: dict | list) -> dict:
    """Normalize Claude's JSON response into our standard frame format.

    Claude may return codes under different keys, nested dicts, or as a bare list.
    We normalize to: {"codes": [...], "decision_rules": [...]}
    """
    # If Claude returned a list directly, treat as codes
    if isinstance(raw, list):
        return {"codes": _normalize_codes(raw), "decision_rules": []}

    # Look for codes under various keys (including nested)
    codes = _find_codes_list(raw)

    # Look for decision rules
    rules = _find_list_by_keys(
        raw, ("decision_rules", "rules", "disambiguation_rules", "disambiguation")
    )

    return {
        "codes": _normalize_codes(codes),
        "decision_rules": rules or [],
    }


def _find_codes_list(d: dict) -> list:
    """Recursively find the list of code objects in a dict."""
    # Direct keys
    for key in ("codes", "coding_frame", "frame", "categories", "codebook"):
        val = d.get(key)
        if isinstance(val, list) and val:
            return val
        # Handle nested: {"coding_frame": {"codes": [...]}}
        if isinstance(val, dict):
            nested = _find_codes_list(val)
            if nested:
                return nested

    # Fallback: find ANY list of dicts that have a "code" or "label" field
    for key, val in d.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            if any(k in val[0] for k in ("code", "label", "id", "name")):
                return val

    return []


def _find_list_by_keys(d: dict, keys: tuple[str, ...]) -> list | None:
    """Find a list value under any of the given keys, including nested."""
    for key in keys:
        val = d.get(key)
        if isinstance(val, list):
            return val

    # Check nested dicts
    for val in d.values():
        if isinstance(val, dict):
            found = _find_list_by_keys(val, keys)
            if found is not None:
                return found

    return None


def _normalize_codes(codes: list) -> list:
    """Normalize each code entry to our standard format."""
    normalized = []
    for c in codes:
        if not isinstance(c, dict):
            continue
        normalized.append({
            "code": c.get("code", c.get("id", c.get("number", 0))),
            "label": c.get("label", c.get("name", c.get("title", ""))),
            "description": c.get("description", c.get("desc", c.get("definition", ""))),
            "includes": c.get("includes", c.get("examples", c.get("keywords", []))),
            "excludes": c.get("excludes", []),
        })
    return normalized


def build_frame_initial(
    client: ClaudeClient,
    responses: list[str],
    max_codes: int = 25,
    language: str = "Bulgarian",
    base_frame_text: str | None = None,
) -> dict:
    """Pass 1: Build initial frame from first batch of responses."""
    if base_frame_text:
        base_instruction = BASE_FRAME_INSTRUCTION.format(
            base_frame_text=base_frame_text
        )
    else:
        base_instruction = BASE_FRAME_INSTRUCTION_NONE

    system = SYSTEM_FRAME_BUILDER.format(
        max_codes=max_codes,
        language=language,
        base_frame_instruction=base_instruction,
    )

    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(responses))
    user_msg = (
        f"Here are {len(responses)} open-ended survey responses. "
        "Analyze them and build the coding frame.\n\n"
        f"{numbered}"
    )

    raw = client.call_json(system=system, user=user_msg, model=MODEL_OPUS)
    return _normalize_frame(raw)


def refine_frame(
    client: ClaudeClient,
    existing_frame: dict,
    new_responses: list[str],
    max_codes: int = 25,
    language: str = "Bulgarian",
) -> dict:
    """Pass 2/3: Refine existing frame with a new batch of responses."""
    system = SYSTEM_FRAME_REFINER.format(
        existing_frame_json=json.dumps(existing_frame, ensure_ascii=False),
        max_codes=max_codes,
        language=language,
    )

    numbered = "\n".join(f"{i+1}. {r}" for i, r in enumerate(new_responses))
    user_msg = (
        f"Here are {len(new_responses)} NEW responses (a different batch). "
        "Review them and refine the coding frame.\n\n"
        f"{numbered}"
    )

    raw = client.call_json(system=system, user=user_msg, model=MODEL_OPUS)
    return _normalize_frame(raw)


def build_frame_iteratively(
    client: ClaudeClient,
    all_responses: list[str],
    batch_size: int = 100,
    max_codes: int = 25,
    language: str = "Bulgarian",
    base_frame_text: str | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    """Run 3-pass iterative frame building.

    Returns a list of frame versions (one per iteration) so the UI can
    show the evolution. The last element is the final frame.
    """
    frames: list[dict] = []

    # Pass 1 — draft
    batch1 = all_responses[:batch_size]
    frame = build_frame_initial(
        client, batch1, max_codes, language, base_frame_text
    )
    frames.append(frame)
    if progress_callback:
        progress_callback(1, 3, frame)

    # Pass 2 — refine
    batch2 = all_responses[batch_size : batch_size * 2]
    if batch2:
        frame = refine_frame(client, frame, batch2, max_codes, language)
        frames.append(frame)
        if progress_callback:
            progress_callback(2, 3, frame)

    # Pass 3 — finalize
    batch3 = all_responses[batch_size * 2 : batch_size * 3]
    if batch3:
        frame = refine_frame(client, frame, batch3, max_codes, language)
        frames.append(frame)
        if progress_callback:
            progress_callback(3, 3, frame)

    return frames
