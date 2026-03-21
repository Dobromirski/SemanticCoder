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

    return client.call_json(system=system, user=user_msg, model=MODEL_OPUS)


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

    return client.call_json(system=system, user=user_msg, model=MODEL_OPUS)


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
