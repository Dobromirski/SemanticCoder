"""Mass coding engine — batches responses and sends them to Claude."""

from __future__ import annotations

import json
from typing import Callable

import pandas as pd

from .claude_client import ClaudeClient, MODEL_SONNET
from .prompts import SYSTEM_CODER, format_frame_for_prompt, format_decision_rules


def code_column(
    client: ClaudeClient,
    df: pd.DataFrame,
    text_column: str,
    code_column_name: str,
    frame: dict,
    batch_size: int = 25,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Code all responses in *text_column* and write results to *code_column_name*.

    Empty / NaN responses are auto-coded 99.
    """
    df = df.copy()
    df[code_column_name] = pd.NA

    # Separate empty from non-empty
    mask_empty = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
    df.loc[mask_empty, code_column_name] = 99

    non_empty = df[~mask_empty]
    if non_empty.empty:
        return df

    # Build system prompt
    system = SYSTEM_CODER.format(
        frame_json=format_frame_for_prompt(frame),
        decision_rules_text=format_decision_rules(frame),
    )

    # Batch and process
    indices = non_empty.index.tolist()
    total_batches = (len(indices) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        batch_indices = indices[start : start + batch_size]

        items = []
        for idx in batch_indices:
            text = str(df.at[idx, text_column]).strip()
            items.append({"id": int(idx), "text": text})

        user_msg = json.dumps(items, ensure_ascii=False)

        try:
            result = client.call_json(system=system, user=user_msg, model=MODEL_SONNET)
            for entry in result:
                row_id = entry.get("id")
                code = entry.get("code")
                if row_id is not None and code is not None:
                    df.at[row_id, code_column_name] = int(code)
        except Exception:
            # Mark failed batch as 98 and continue
            for idx in batch_indices:
                if pd.isna(df.at[idx, code_column_name]):
                    df.at[idx, code_column_name] = 98

        if progress_callback:
            progress_callback(batch_num + 1, total_batches)

    # Ensure integer codes
    df[code_column_name] = pd.to_numeric(df[code_column_name], errors="coerce").astype("Int64")
    return df
