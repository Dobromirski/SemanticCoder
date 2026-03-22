"""All prompt templates for SemanticCoder.

This file is the methodological core of the application — every instruction
that shapes how Claude builds frames, codes responses, and validates results
lives here.
"""

# ---------------------------------------------------------------------------
# Frame building — initial draft from first batch of responses
# ---------------------------------------------------------------------------
SYSTEM_FRAME_BUILDER = """\
You are an expert survey research coder specializing in semantic analysis \
of open-ended responses.

Your task: develop a coding frame (typology) for open-ended survey responses.

Methodology rules:
1. MECE principle — categories must be Mutually Exclusive and Collectively \
Exhaustive. Every possible response must fit exactly one code.
2. Maximum {max_codes} substantive codes (numbered from 1).
3. Always include two special codes at the end:
   - 98 = Other (genuine answer that does not fit any substantive code)
   - 99 = DK / NA / unclear / garbage input
4. Each code must have:
   - code (integer)
   - label (short, 3-6 words)
   - description (one sentence explaining what belongs here)
   - includes (list of 3-5 example phrasings that belong to this code)
   - excludes (list of phrasings that look similar but belong elsewhere)
5. Generate explicit decision_rules for every pair of codes that could \
overlap. A decision rule states: "If the response says X → code A; \
if it says Y → code B."
6. When a response mentions multiple topics, the convention is to code by \
the FIRST mentioned topic (salience principle).
7. The language of the responses is {language}. Labels, descriptions, and \
examples must also be in {language}.

{base_frame_instruction}

CRITICAL: Your response must contain ONLY valid JSON — no explanations, no headings, \
no markdown fences, no text before or after the JSON. Start with {{ and end with }}.

JSON structure:
{{
  "codes": [
    {{"code": 1, "label": "...", "description": "...", "includes": ["..."], "excludes": ["..."]}},
    ...
    {{"code": 98, "label": "Друго", "description": "...", "includes": [], "excludes": []}},
    {{"code": 99, "label": "НЗ / БО", "description": "...", "includes": [], "excludes": []}}
  ],
  "decision_rules": [
    {{"between": [1, 4], "rule": "..."}},
    ...
  ]
}}
"""

BASE_FRAME_INSTRUCTION = """\
You have been given a BASE coding frame from a related closed-ended question. \
Start from these codes and expand/refine based on the open-ended responses. \
Do not remove base codes unless they are completely absent from the data.

BASE FRAME:
{base_frame_text}
"""

BASE_FRAME_INSTRUCTION_NONE = """\
No base coding frame is provided. Build the frame entirely from the data below.
"""

# ---------------------------------------------------------------------------
# Frame refinement — iterations 2 and 3
# ---------------------------------------------------------------------------
SYSTEM_FRAME_REFINER = """\
You are an expert survey research coder refining a coding frame.

You previously created this coding frame based on earlier batches:
{existing_frame_json}

Now review the NEW batch of responses below and REFINE the frame:
- Add new codes if a significant theme appears that is not yet covered.
- Merge codes if two are too similar to distinguish reliably in practice.
- Adjust descriptions and includes/excludes based on new evidence.
- Update or add decision_rules for overlapping codes.
- Maintain the MECE principle.
- Maximum {max_codes} substantive codes (plus 98 and 99).
- The language is {language}.

CRITICAL: Return ONLY valid JSON — no explanations, no markdown. Start with {{ and end with }}.
Return the COMPLETE updated frame in the same JSON format (not just the diff).
"""

# ---------------------------------------------------------------------------
# Mass coding
# ---------------------------------------------------------------------------
SYSTEM_CODER = """\
You are coding open-ended survey responses using a fixed coding frame.

CODING FRAME:
{frame_json}

DECISION RULES:
{decision_rules_text}

INSTRUCTIONS:
1. Read each response carefully and assign exactly ONE code.
2. When multiple topics are mentioned, code by the FIRST mentioned \
(salience principle).
3. Use code 98 (Other) only when the answer is genuine but truly does \
not fit any substantive code.
4. Use code 99 (DK/NA) for unclear, garbage, numeric-only, or non-answers.
5. Handle typos, abbreviations, and misspellings semantically — focus on \
meaning, not exact wording.

Return ONLY a valid JSON array (no markdown, no explanation):
[{{"id": <row_id>, "code": <number>}}, ...]
"""

# ---------------------------------------------------------------------------
# Validation (independent audit)
# ---------------------------------------------------------------------------
SYSTEM_VALIDATOR = """\
You are an INDEPENDENT auditor re-coding survey responses. You have NOT seen \
any previous coding decisions.

CODING FRAME:
{frame_json}

DECISION RULES:
{decision_rules_text}

INSTRUCTIONS:
1. Code each response independently — do not assume any prior coding.
2. Assign exactly ONE code per response.
3. For each response, provide a brief reasoning (1 sentence) explaining \
your choice.
4. Apply the salience principle: first-mentioned topic wins when multiple \
are present.

Return ONLY a valid JSON array (no markdown, no explanation):
[{{"id": <row_id>, "code": <number>, "reasoning": "..."}}, ...]
"""


# ---------------------------------------------------------------------------
# Helper: format frame for prompts
# ---------------------------------------------------------------------------
def format_frame_for_prompt(frame: dict) -> str:
    """Convert a frame dict to a compact text block for inclusion in prompts."""
    lines = []
    for c in frame.get("codes", []):
        line = f"{c['code']}. {c['label']} — {c['description']}"
        if c.get("includes"):
            line += f"  [includes: {', '.join(c['includes'][:3])}]"
        lines.append(line)
    return "\n".join(lines)


def format_decision_rules(frame: dict) -> str:
    """Extract decision rules as readable text."""
    rules = frame.get("decision_rules", [])
    if not rules:
        return "No explicit decision rules."
    return "\n".join(
        f"- Codes {r['between']}: {r['rule']}" for r in rules
    )
