"""Post-coding audit — reviews code 98/99 assignments and recodes if needed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .claude_client import ClaudeClient, MODEL_OPUS
from .prompts import format_frame_for_prompt, format_decision_rules


SYSTEM_AUDIT_98 = """\
You are an expert survey coder performing a QUALITY AUDIT on coded responses.

A previous coder assigned code 98 (Other) or 99 (DK/NA) to the responses below. \
Your job is to review each one and decide:
- Is 98/99 correct? Some responses genuinely don't fit any code.
- Or was it a MISS — the response actually fits one of the substantive codes?

CODING FRAME:
{frame_text}

DECISION RULES:
{decision_rules_text}

For each response, return your verdict:
- If the existing code is correct, keep it.
- If it should be recoded, provide the correct code.

Return ONLY valid JSON (no markdown):
[{{"id": <row_id>, "original_code": <98_or_99>, "correct_code": <number>, "reasoning": "..."}}]
"""

SYSTEM_RECONCILE = """\
You are an expert survey coder resolving DISAGREEMENTS between two independent coders.

Two coders assigned different codes to the same responses. Review each case and \
decide which code is correct (or assign a third code if both are wrong).

CODING FRAME:
{frame_text}

DECISION RULES:
{decision_rules_text}

For each disagreement, provide the final code with reasoning.

Return ONLY valid JSON (no markdown):
[{{"id": <row_id>, "code_a": <number>, "code_b": <number>, "final_code": <number>, "reasoning": "..."}}]
"""


@dataclass
class AuditResult:
    """Result of auditing code 98/99 assignments."""
    total_reviewed: int = 0
    recoded: int = 0
    kept: int = 0
    changes: list[dict] = field(default_factory=list)


def audit_other_codes(
    client: ClaudeClient,
    df: pd.DataFrame,
    text_column: str,
    code_column: str,
    frame: dict,
    batch_size: int = 25,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, AuditResult]:
    """Review all code 98 and 99 assignments — recode misses.

    Returns updated DataFrame and audit result.
    """
    df = df.copy()
    result = AuditResult()

    # Collect 98/99 rows (skip truly empty/garbage for 99)
    mask_98 = df[code_column] == 98
    mask_99_with_text = (df[code_column] == 99) & (
        df[text_column].astype(str).str.strip().str.len() > 3
    )
    review_mask = mask_98 | mask_99_with_text
    to_review = df[review_mask]

    if to_review.empty:
        return df, result

    result.total_reviewed = len(to_review)

    system = SYSTEM_AUDIT_98.format(
        frame_text=format_frame_for_prompt(frame),
        decision_rules_text=format_decision_rules(frame),
    )

    indices = to_review.index.tolist()
    total_batches = (len(indices) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        batch_indices = indices[start : start + batch_size]

        items = []
        for idx in batch_indices:
            items.append({
                "id": int(idx),
                "text": str(df.at[idx, text_column]).strip(),
                "current_code": int(df.at[idx, code_column]),
            })

        user_msg = json.dumps(items, ensure_ascii=False)

        try:
            verdicts = client.call_json(
                system=system, user=user_msg, model=MODEL_OPUS
            )
            for v in verdicts:
                row_id = v.get("id")
                correct = v.get("correct_code")
                original = v.get("original_code")
                if row_id is not None and correct is not None:
                    if correct != original:
                        df.at[row_id, code_column] = int(correct)
                        result.recoded += 1
                        result.changes.append(v)
                    else:
                        result.kept += 1
        except Exception:
            pass  # Skip failed batches

        if progress_callback:
            progress_callback(batch_num + 1, total_batches, "Auditing 98/99 codes")

    return df, result


def reconcile_disagreements(
    client: ClaudeClient,
    df: pd.DataFrame,
    text_column: str,
    disagreements: list[dict],
    frame: dict,
    code_column: str | None = None,
    batch_size: int = 25,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Resolve disagreements between original coding and audit using a third opinion.

    Returns updated DataFrame and list of resolutions.
    """
    df = df.copy()
    if not disagreements:
        return df, []

    # Determine code column
    if code_column is None:
        code_column = text_column.replace("t", "c") if text_column.endswith("t") else f"{text_column}_code"

    system = SYSTEM_RECONCILE.format(
        frame_text=format_frame_for_prompt(frame),
        decision_rules_text=format_decision_rules(frame),
    )

    total_batches = (len(disagreements) + batch_size - 1) // batch_size
    resolutions = []

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        batch = disagreements[start : start + batch_size]

        items = []
        for d in batch:
            items.append({
                "id": d["row"],
                "text": d["text"],
                "code_a": d["original_code"],
                "code_b": d["audit_code"],
            })

        user_msg = json.dumps(items, ensure_ascii=False)

        try:
            verdicts = client.call_json(
                system=system, user=user_msg, model=MODEL_OPUS
            )
            for v in verdicts:
                row_id = v.get("id")
                final = v.get("final_code")
                if row_id is not None and final is not None and row_id in df.index:
                    df.at[row_id, code_column] = int(final)
                    resolutions.append(v)
        except Exception:
            pass

        if progress_callback:
            progress_callback(batch_num + 1, total_batches, "Resolving disagreements")

    return df, resolutions
