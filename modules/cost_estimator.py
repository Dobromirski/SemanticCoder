"""Estimate API costs before running coding or validation."""

from __future__ import annotations

import pandas as pd

# Pricing per 1M tokens (as of March 2026)
PRICING = {
    "opus": {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0, "output": 15.0},
}

# Rough token-per-character ratio for Cyrillic text
CHARS_PER_TOKEN = 2.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def estimate_frame_building_cost(responses: list[str], passes: int = 3) -> dict:
    """Estimate cost for iterative frame building (Opus)."""
    system_tokens = 800  # approximate system prompt
    batch_tokens = sum(_estimate_tokens(r) for r in responses[:100])
    output_tokens = 2000  # frame JSON

    total_input = (system_tokens + batch_tokens) * passes
    total_output = output_tokens * passes

    cost = (
        total_input / 1_000_000 * PRICING["opus"]["input"]
        + total_output / 1_000_000 * PRICING["opus"]["output"]
    )
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": cost,
        "model": "Opus",
        "calls": passes,
    }


def estimate_coding_cost(
    df: pd.DataFrame, text_column: str, batch_size: int = 25
) -> dict:
    """Estimate cost for mass coding (Sonnet)."""
    non_empty = df[df[text_column].notna() & (df[text_column].astype(str).str.strip() != "")]
    n = len(non_empty)
    num_batches = (n + batch_size - 1) // batch_size

    avg_response_tokens = non_empty[text_column].astype(str).apply(_estimate_tokens).mean()
    system_tokens = 1500  # frame + rules

    input_per_batch = system_tokens + int(avg_response_tokens * batch_size)
    output_per_batch = batch_size * 12  # ~12 tokens per {id, code}

    total_input = input_per_batch * num_batches
    total_output = output_per_batch * num_batches

    cost = (
        total_input / 1_000_000 * PRICING["sonnet"]["input"]
        + total_output / 1_000_000 * PRICING["sonnet"]["output"]
    )
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": cost,
        "model": "Sonnet",
        "calls": num_batches,
        "responses": n,
    }


def estimate_validation_cost(sample_size: int = 100, batch_size: int = 25) -> dict:
    """Estimate cost for validation pass (Opus)."""
    num_batches = (sample_size + batch_size - 1) // batch_size
    system_tokens = 1500
    avg_response_tokens = 20
    input_per_batch = system_tokens + avg_response_tokens * batch_size
    output_per_batch = batch_size * 30  # ~30 tokens per {id, code, reasoning}

    total_input = input_per_batch * num_batches
    total_output = output_per_batch * num_batches

    cost = (
        total_input / 1_000_000 * PRICING["opus"]["input"]
        + total_output / 1_000_000 * PRICING["opus"]["output"]
    )
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": cost,
        "model": "Opus",
        "calls": num_batches,
    }


def format_cost_summary(
    frame_cost: dict | None = None,
    coding_cost: dict | None = None,
    validation_cost: dict | None = None,
) -> str:
    """Human-readable cost summary."""
    lines = []
    total = 0.0
    if frame_cost:
        lines.append(f"Frame building ({frame_cost['model']}, {frame_cost['calls']} calls): ${frame_cost['cost_usd']:.2f}")
        total += frame_cost["cost_usd"]
    if coding_cost:
        lines.append(f"Mass coding ({coding_cost['model']}, {coding_cost['calls']} batches, {coding_cost['responses']} responses): ${coding_cost['cost_usd']:.2f}")
        total += coding_cost["cost_usd"]
    if validation_cost:
        lines.append(f"Validation ({validation_cost['model']}, {validation_cost['calls']} batches): ${validation_cost['cost_usd']:.2f}")
        total += validation_cost["cost_usd"]
    lines.append(f"**Total estimated: ${total:.2f}**")
    return "\n".join(lines)
