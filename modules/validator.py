"""Dual-pass validation — independent audit of coded responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from .claude_client import ClaudeClient, MODEL_OPUS
from .prompts import SYSTEM_VALIDATOR, format_frame_for_prompt, format_decision_rules


@dataclass
class ValidationResult:
    agreement_rate: float
    total_compared: int
    agreements: int
    disagreements: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        pct = f"{self.agreement_rate:.1%}"
        return (
            f"Agreement rate: {pct} "
            f"({self.agreements}/{self.total_compared}). "
            f"Disagreements: {len(self.disagreements)}."
        )


def run_validation(
    client: ClaudeClient,
    df: pd.DataFrame,
    text_column: str,
    code_column: str,
    frame: dict,
    sample_size: int = 100,
    batch_size: int = 25,
    seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
) -> ValidationResult:
    """Re-code a random sample and compare with existing codes."""
    # Sample non-empty, coded rows
    valid = df[df[code_column].notna() & df[text_column].notna()].copy()
    if len(valid) == 0:
        return ValidationResult(0.0, 0, 0)

    sample = valid.sample(n=min(sample_size, len(valid)), random_state=seed)

    system = SYSTEM_VALIDATOR.format(
        frame_json=format_frame_for_prompt(frame),
        decision_rules_text=format_decision_rules(frame),
    )

    # Batch and send
    indices = sample.index.tolist()
    total_batches = (len(indices) + batch_size - 1) // batch_size
    audit_results: dict[int, dict] = {}

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        batch_indices = indices[start : start + batch_size]

        items = []
        for idx in batch_indices:
            text = str(df.at[idx, text_column]).strip()
            items.append({"id": int(idx), "text": text})

        user_msg = json.dumps(items, ensure_ascii=False)

        try:
            result = client.call_json(system=system, user=user_msg, model=MODEL_OPUS)
            for entry in result:
                row_id = entry.get("id")
                if row_id is not None:
                    audit_results[row_id] = entry
        except Exception:
            pass  # Skip failed batches — they won't count in the comparison

        if progress_callback:
            progress_callback(batch_num + 1, total_batches)

    # Compare
    agreements = 0
    disagreements = []
    compared = 0

    for idx in indices:
        if idx not in audit_results:
            continue
        compared += 1
        original_code = int(df.at[idx, code_column])
        audit_entry = audit_results[idx]
        audit_code = int(audit_entry.get("code", -1))

        if original_code == audit_code:
            agreements += 1
        else:
            disagreements.append({
                "row": int(idx),
                "text": str(df.at[idx, text_column]),
                "original_code": original_code,
                "audit_code": audit_code,
                "reasoning": audit_entry.get("reasoning", ""),
            })

    rate = agreements / compared if compared > 0 else 0.0
    return ValidationResult(
        agreement_rate=rate,
        total_compared=compared,
        agreements=agreements,
        disagreements=disagreements,
    )
