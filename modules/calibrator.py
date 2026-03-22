"""Human-in-the-loop calibration — extract decision rules from human corrections."""

from __future__ import annotations

import json

from .claude_client import ClaudeClient, MODEL_OPUS
from .prompts import format_frame_for_prompt, format_decision_rules


SYSTEM_EXTRACT_RULES = """\
You are a survey methodology expert analyzing a researcher's corrections to coded data.

The researcher reviewed {n_total} randomly selected coded responses and corrected {n_corrected} of them.
Your task is to identify PATTERNS in these corrections and formulate them as explicit decision rules.

CODING FRAME:
{frame_text}

EXISTING DECISION RULES:
{decision_rules_text}

CORRECTIONS MADE BY THE RESEARCHER:
{corrections_json}

Analyze the corrections and:
1. Look for systematic patterns (not one-off judgment calls)
2. Formulate each pattern as a clear decision rule
3. Explain your reasoning — what do the corrections have in common?
4. Mark your confidence: HIGH (clear pattern from 2+ corrections), MEDIUM (single correction but generalizable), LOW (might be a one-off)

Return ONLY valid JSON (no markdown):
{{
  "extracted_rules": [
    {{
      "rule": "When the response mentions X in context Y, assign code Z instead of W",
      "confidence": "HIGH|MEDIUM|LOW",
      "based_on": [list of row IDs that support this rule],
      "reasoning": "Why this pattern exists"
    }}
  ],
  "one_offs": [
    {{
      "row_id": <id>,
      "note": "This correction appears to be a specific judgment call, not a general rule"
    }}
  ]
}}
"""


def extract_rules_from_corrections(
    client: ClaudeClient,
    corrections: list[dict],
    total_reviewed: int,
    frame: dict,
) -> dict:
    """Analyze human corrections and extract generalizable decision rules.

    Args:
        corrections: list of dicts with keys: row_id, text, original_code, corrected_code
        total_reviewed: total number of items shown to the human
        frame: the current coding frame

    Returns:
        dict with "extracted_rules" and "one_offs" lists
    """
    if not corrections:
        return {"extracted_rules": [], "one_offs": []}

    system = SYSTEM_EXTRACT_RULES.format(
        n_total=total_reviewed,
        n_corrected=len(corrections),
        frame_text=format_frame_for_prompt(frame),
        decision_rules_text=format_decision_rules(frame),
        corrections_json=json.dumps(corrections, ensure_ascii=False, indent=2),
    )

    user_msg = (
        "Please analyze these corrections and extract decision rules. "
        "Focus on patterns that would improve coding consistency."
    )

    try:
        result = client.call_json(system=system, user=user_msg, model=MODEL_OPUS)
        return result
    except Exception:
        return {"extracted_rules": [], "one_offs": []}


def format_rules_for_approval(extracted: dict) -> str:
    """Format extracted rules into human-readable text for review."""
    lines = []

    rules = extracted.get("extracted_rules", [])
    if rules:
        lines.append("## Proposed Decision Rules\n")
        for i, r in enumerate(rules, 1):
            conf = r.get("confidence", "?")
            conf_icon = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "LOW": "[LOW]"}.get(conf, f"[{conf}]")
            lines.append(f"**{i}. {conf_icon}** {r.get('rule', '')}")
            lines.append(f"   *Reasoning:* {r.get('reasoning', '')}")
            based = r.get("based_on", [])
            if based:
                lines.append(f"   *Based on rows:* {', '.join(str(b) for b in based)}")
            lines.append("")

    one_offs = extracted.get("one_offs", [])
    if one_offs:
        lines.append("## One-off Corrections (no general rule extracted)\n")
        for o in one_offs:
            lines.append(f"- Row {o.get('row_id', '?')}: {o.get('note', '')}")

    return "\n".join(lines)


def merge_approved_rules(frame: dict, approved_rules: list[dict]) -> dict:
    """Add approved rules to the coding frame's decision_rules.

    Args:
        frame: current coding frame dict
        approved_rules: list of rule dicts with at least a "rule" key

    Returns:
        updated frame with new rules appended
    """
    frame = frame.copy()
    existing = frame.get("decision_rules", [])

    for r in approved_rules:
        rule_text = r.get("rule", "")
        if rule_text:
            existing.append({
                "between": "calibration",
                "rule": rule_text,
            })

    frame["decision_rules"] = existing
    return frame
