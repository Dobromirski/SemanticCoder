"""File upload/download and Excel/CSV handling."""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd


def load_file(uploaded_file: Any) -> pd.DataFrame:
    """Read a Streamlit UploadedFile into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except (UnicodeDecodeError, UnicodeError):
                continue
        raise ValueError("Could not decode CSV with any known encoding.")
    if name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {name}")


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic: columns where >50% of values are strings longer than 5 chars."""
    text_cols = []
    for col in df.columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        str_vals = series.astype(str)
        long_strings = (str_vals.str.len() > 5).sum()
        if long_strings / len(series) > 0.5:
            # Exclude columns that look numeric
            try:
                pd.to_numeric(series)
                continue
            except (ValueError, TypeError):
                pass
            text_cols.append(col)
    return text_cols


def export_coded_excel(df: pd.DataFrame) -> io.BytesIO:
    """Generate downloadable Excel bytes from a DataFrame."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf


def export_frame_excel(frame: dict) -> io.BytesIO:
    """Export the coding frame as a structured Excel file."""
    rows = []
    for c in frame.get("codes", []):
        rows.append({
            "Code": c["code"],
            "Label": c["label"],
            "Description": c.get("description", ""),
            "Includes": "; ".join(c.get("includes", [])),
            "Excludes": "; ".join(c.get("excludes", [])),
        })
    df = pd.DataFrame(rows)

    # Add decision rules as a second sheet
    rules = frame.get("decision_rules", [])
    rules_rows = [{"Between": str(r["between"]), "Rule": r["rule"]} for r in rules]
    df_rules = pd.DataFrame(rules_rows) if rules_rows else pd.DataFrame(columns=["Between", "Rule"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Coding Frame", index=False)
        df_rules.to_excel(writer, sheet_name="Decision Rules", index=False)
    buf.seek(0)
    return buf


def export_frame_json(frame: dict) -> str:
    """Export the coding frame as formatted JSON."""
    return json.dumps(frame, ensure_ascii=False, indent=2)


def export_validation_report(result: Any) -> io.BytesIO:
    """Export validation results as Excel."""
    summary = pd.DataFrame([{
        "Agreement Rate": f"{result.agreement_rate:.1%}",
        "Total Compared": result.total_compared,
        "Agreements": result.agreements,
        "Disagreements": len(result.disagreements),
    }])

    disagreements = pd.DataFrame(result.disagreements) if result.disagreements else pd.DataFrame()

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if not disagreements.empty:
            disagreements.to_excel(writer, sheet_name="Disagreements", index=False)
    buf.seek(0)
    return buf


def generate_spss_syntax(frame: dict, variable_name: str = "q1c") -> str:
    """Generate SPSS VALUE LABELS syntax for the coding frame."""
    lines = [f"VALUE LABELS {variable_name}"]
    for c in frame.get("codes", []):
        label = c["label"].replace("'", "''")
        lines.append(f"  {c['code']} '{label}'")
    lines[-1] += "."
    return "\n".join(lines)
