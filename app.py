"""SemanticCoder — Streamlit app for semantic coding of open-ended survey responses."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from modules.claude_client import ClaudeClient
from modules.auditor import audit_other_codes, reconcile_disagreements
from modules.coder import code_column
from modules.data_io import (
    detect_text_columns,
    export_coded_excel,
    export_frame_excel,
    export_frame_json,
    export_validation_report,
    generate_spss_syntax,
    load_file,
)
from modules.frame_builder import build_frame_initial, refine_frame
from modules.validator import run_validation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SemanticCoder",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "step": 1,
    "df": None,
    "text_columns": [],
    "config": {"max_codes": 25, "language": "Bulgarian", "batch_size": 25},
    "frame": None,
    "frame_history": [],
    "frame_iteration": 0,
    "frame_approved": False,
    "coded_df": None,
    "current_coding_col_idx": 0,
    "coding_phase": "code",  # code → audit_98 → validate → reconcile → done
    "audit_98_result": None,
    "validation": None,
    "reconciliation": None,
}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def get_client() -> ClaudeClient:
    return ClaudeClient()


# ---------------------------------------------------------------------------
# Sidebar — step indicator
# ---------------------------------------------------------------------------
STEPS = {
    1: "Upload & Configure",
    2: "Build Coding Frame",
    3: "Code & Validate (3 agents)",
    5: "Export",
}

with st.sidebar:
    st.title("SemanticCoder")
    st.caption("AI-powered semantic coding for survey research")
    st.divider()
    for num, label in STEPS.items():
        if num < st.session_state.step:
            st.write(f"~~{num}. {label}~~")
        elif num == st.session_state.step:
            st.write(f"**{num}. {label}** ←")
        else:
            st.write(f"{num}. {label}")


# ===================================================================
# STEP 1 — Upload & Configure
# ===================================================================
if st.session_state.step == 1:
    st.header("1. Upload & Configure")

    uploaded = st.file_uploader(
        "Upload your data file", type=["xlsx", "xls", "csv"]
    )

    if uploaded:
        try:
            df = load_file(uploaded)
            st.session_state.df = df
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as exc:
            st.error(f"Error loading file: {exc}")
            st.stop()

    if st.session_state.df is not None:
        df = st.session_state.df
        detected = detect_text_columns(df)

        st.subheader("Configuration")

        col1, col2 = st.columns(2)
        with col1:
            text_cols = st.multiselect(
                "Select text columns to code",
                options=list(df.columns),
                default=detected[:3] if detected else [],
            )
            language = st.selectbox(
                "Response language",
                ["Bulgarian", "English", "Russian", "Other"],
                index=0,
            )
        with col2:
            max_codes = st.number_input(
                "Max substantive codes", min_value=5, max_value=50, value=25
            )
            batch_size = st.number_input(
                "Batch size for coding", min_value=10, max_value=50, value=25
            )

        st.subheader("Base coding frame (optional)")
        base_frame = st.text_area(
            "Paste a base coding frame from a related closed-ended question "
            "(numbered list, e.g. '1. Economy\\n2. Healthcare\\n...')",
            height=150,
            placeholder="1. Обедняване\n2. Безработица\n3. Политическа нестабилност\n...",
        )

        if text_cols and st.button("Proceed to Frame Building", type="primary"):
            st.session_state.text_columns = text_cols
            st.session_state.config = {
                "max_codes": max_codes,
                "language": language,
                "batch_size": batch_size,
                "base_frame": base_frame.strip() if base_frame else None,
            }
            st.session_state.step = 2
            st.session_state.frame_iteration = 0
            st.session_state.frame_history = []
            st.session_state.frame_approved = False
            st.rerun()


# ===================================================================
# STEP 2 — Build Coding Frame
# ===================================================================
elif st.session_state.step == 2:
    st.header("2. Build Coding Frame")

    cfg = st.session_state.config
    df = st.session_state.df
    # Use first selected text column for frame building
    text_col = st.session_state.text_columns[0]
    all_responses = df[text_col].dropna().astype(str).tolist()

    iteration = st.session_state.frame_iteration

    if not st.session_state.frame_approved:
        if iteration < 3:
            batch_start = iteration * 100
            batch_end = batch_start + 100
            batch = all_responses[batch_start:batch_end]

            if not batch:
                st.warning("Not enough responses for another iteration.")
                st.session_state.frame_approved = True
                st.rerun()

            st.info(
                f"Iteration {iteration + 1}/3: Analyzing responses "
                f"{batch_start + 1}–{min(batch_end, len(all_responses))}..."
            )

            if st.button(f"Run iteration {iteration + 1}", type="primary"):
                client = get_client()
                with st.spinner(f"Claude is analyzing batch {iteration + 1}..."):
                    try:
                        if iteration == 0:
                            frame = build_frame_initial(
                                client,
                                batch,
                                cfg["max_codes"],
                                cfg["language"],
                                cfg.get("base_frame"),
                            )
                        else:
                            frame = refine_frame(
                                client,
                                st.session_state.frame,
                                batch,
                                cfg["max_codes"],
                                cfg["language"],
                            )
                        st.session_state.frame = frame
                        st.session_state.frame_history.append(frame)
                        st.session_state.frame_iteration = iteration + 1
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        # Show current frame if we have one
        if st.session_state.frame:
            frame = st.session_state.frame
            st.subheader(f"Current frame (after iteration {iteration})")

            # Editable table
            codes_df = pd.DataFrame(frame.get("codes", []))
            edited = st.data_editor(
                codes_df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"frame_editor_{iteration}",
            )

            # Decision rules
            rules_text = "\n".join(
                f"Codes {r['between']}: {r['rule']}"
                for r in frame.get("decision_rules", [])
            )
            edited_rules = st.text_area(
                "Decision rules (edit if needed)",
                value=rules_text,
                height=150,
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                if iteration < 3 and st.button("Continue to next iteration"):
                    # Save edits back to frame
                    frame["codes"] = edited.to_dict("records")
                    st.session_state.frame = frame
                    st.rerun()
            with col2:
                if st.button("Approve this frame", type="primary"):
                    frame["codes"] = edited.to_dict("records")
                    st.session_state.frame = frame
                    st.session_state.frame_approved = True
                    st.rerun()
            with col3:
                if st.button("Reset and start over"):
                    st.session_state.frame = None
                    st.session_state.frame_history = []
                    st.session_state.frame_iteration = 0
                    st.rerun()

    else:
        # Frame is approved — show cost estimate and proceed button
        st.success("Coding frame approved!")

        frame = st.session_state.frame
        codes_df = pd.DataFrame(frame.get("codes", []))
        st.dataframe(codes_df, use_container_width=True)

        total_cols = len(st.session_state.text_columns)
        st.info(f"Ready to code {total_cols} column(s) via Max subscription (no API cost).")

        if st.button("Proceed to Coding", type="primary"):
            st.session_state.step = 3
            st.session_state.current_coding_col_idx = 0
            st.rerun()

        if st.button("Back to edit frame"):
            st.session_state.frame_approved = False
            st.rerun()


# ===================================================================
# STEP 3 — Code, Audit, Validate (3-agent pipeline)
# ===================================================================
elif st.session_state.step == 3:
    st.header("3. Code & Validate")

    df = st.session_state.df
    frame = st.session_state.frame
    cfg = st.session_state.config
    text_cols = st.session_state.text_columns
    col_idx = st.session_state.current_coding_col_idx
    phase = st.session_state.coding_phase

    def _code_col_name(tc: str) -> str:
        return tc.replace("t", "c") if tc.endswith("t") else f"{tc}_code"

    # --- PHASE 1: Coding (Sonnet) ---
    if phase == "code":
        if col_idx < len(text_cols):
            text_col = text_cols[col_idx]
            code_col = _code_col_name(text_col)
            st.subheader(f"Agent 1 — Coding: {text_col} ({col_idx + 1}/{len(text_cols)})")

            if st.button(f"Start coding '{text_col}'", type="primary"):
                client = get_client()
                progress = st.progress(0, text="Starting...")

                def on_progress(current: int, total: int) -> None:
                    progress.progress(current / total, text=f"Batch {current}/{total}")

                with st.spinner("Agent 1 (Sonnet) coding in progress..."):
                    coded_df = code_column(
                        client, df, text_col, code_col, frame,
                        batch_size=cfg.get("batch_size", 25),
                        progress_callback=on_progress,
                    )
                    st.session_state.df = coded_df
                    st.session_state.current_coding_col_idx = col_idx + 1
                    st.rerun()
        else:
            # All columns coded — show distribution and move to audit
            st.success(f"Agent 1 complete — all {len(text_cols)} columns coded.")
            for tc in text_cols:
                cc = _code_col_name(tc)
                if cc in df.columns:
                    st.subheader(f"Distribution: {cc}")
                    dist = df[cc].value_counts().sort_index()
                    n98 = int(dist.get(98, 0))
                    n99 = int(dist.get(99, 0))
                    total = len(df)
                    st.bar_chart(dist)
                    if n98 + n99 > 0:
                        pct = (n98 + n99) / total * 100
                        st.warning(f"Code 98/99: {n98 + n99} responses ({pct:.1f}%) — Agent 2 will review these.")

            if st.button("Run Agent 2 — Audit 98/99 codes", type="primary"):
                st.session_state.coding_phase = "audit_98"
                st.rerun()

    # --- PHASE 2: Audit 98/99 (Opus) ---
    elif phase == "audit_98":
        st.subheader("Agent 2 — Auditing 98/99 codes (Opus)")
        st.caption("An independent agent reviews all code 98 (Other) and 99 (DK/NA) assignments, "
                   "checking if they were genuine or misses from Agent 1.")

        if st.session_state.audit_98_result is None:
            client = get_client()
            progress = st.progress(0, text="Starting audit...")

            def on_audit_progress(current: int, total: int, msg: str) -> None:
                progress.progress(current / total, text=f"{msg} — batch {current}/{total}")

            all_audit_results = []
            for tc in text_cols:
                cc = _code_col_name(tc)
                if cc not in df.columns:
                    continue
                with st.spinner(f"Agent 2 auditing {cc}..."):
                    updated_df, audit_res = audit_other_codes(
                        client, df, tc, cc, frame,
                        progress_callback=on_audit_progress,
                    )
                    st.session_state.df = updated_df
                    df = updated_df
                    all_audit_results.append((cc, audit_res))

            st.session_state.audit_98_result = all_audit_results
            st.rerun()
        else:
            for cc, audit_res in st.session_state.audit_98_result:
                st.write(f"**{cc}**: reviewed {audit_res.total_reviewed}, "
                         f"recoded {audit_res.recoded}, kept {audit_res.kept}")
                if audit_res.changes:
                    with st.expander(f"Changes in {cc}"):
                        st.dataframe(pd.DataFrame(audit_res.changes), use_container_width=True)

            # Show updated distributions
            for tc in text_cols:
                cc = _code_col_name(tc)
                if cc in df.columns:
                    dist = df[cc].value_counts().sort_index()
                    n98 = int(dist.get(98, 0))
                    total = len(df)
                    st.write(f"**{cc}** after audit: {n98} remaining code 98 ({n98/total*100:.1f}%)")

            if st.button("Run Agent 3 — Independent Validation", type="primary"):
                st.session_state.coding_phase = "validate"
                st.rerun()

    # --- PHASE 3: Independent validation (Opus) ---
    elif phase == "validate":
        st.subheader("Agent 3 — Independent Validation (Opus)")
        st.caption("A third independent agent re-codes a random sample of 100 responses "
                   "without seeing the previous coding. Measures agreement rate.")

        if st.session_state.validation is None:
            text_col = text_cols[0]
            code_col = _code_col_name(text_col)

            client = get_client()
            progress = st.progress(0, text="Starting validation...")

            def on_val_progress(current: int, total: int) -> None:
                progress.progress(current / total, text=f"Batch {current}/{total}")

            with st.spinner("Agent 3 validating independently..."):
                result = run_validation(
                    client, df, text_col, code_col, frame,
                    sample_size=100,
                    progress_callback=on_val_progress,
                )
                st.session_state.validation = result
                st.rerun()
        else:
            result = st.session_state.validation

            col1, col2, col3 = st.columns(3)
            col1.metric("Agreement Rate", f"{result.agreement_rate:.1%}")
            col2.metric("Compared", result.total_compared)
            col3.metric("Disagreements", len(result.disagreements))

            if result.agreement_rate >= 0.90:
                st.success("Agreement >= 90% — coding is reliable.")
            elif result.agreement_rate >= 0.80:
                st.warning("Agreement 80-90% — review disagreements.")
            else:
                st.error("Agreement < 80% — consider revising the coding frame.")

            if result.disagreements:
                st.subheader("Disagreements")
                st.dataframe(pd.DataFrame(result.disagreements), use_container_width=True)

                if len(result.disagreements) > 0:
                    if st.button("Resolve disagreements (third opinion)", type="primary"):
                        st.session_state.coding_phase = "reconcile"
                        st.rerun()

            if st.button("Proceed to Export", type="primary"):
                st.session_state.step = 5
                st.rerun()

    # --- PHASE 4: Reconciliation (Opus) ---
    elif phase == "reconcile":
        st.subheader("Resolving Disagreements")
        st.caption("A fourth agent reviews each disagreement and picks the correct code "
                   "(or assigns a new one if both are wrong).")

        if st.session_state.reconciliation is None:
            text_col = text_cols[0]
            code_col = _code_col_name(text_col)
            result = st.session_state.validation

            client = get_client()
            progress = st.progress(0, text="Resolving...")

            def on_rec_progress(current: int, total: int, msg: str) -> None:
                progress.progress(current / total, text=f"{msg} — batch {current}/{total}")

            with st.spinner("Resolving disagreements..."):
                updated_df, resolutions = reconcile_disagreements(
                    client, df, text_col,
                    result.disagreements, frame,
                    progress_callback=on_rec_progress,
                )
                st.session_state.df = updated_df
                st.session_state.reconciliation = resolutions
                st.rerun()
        else:
            resolutions = st.session_state.reconciliation
            st.success(f"Resolved {len(resolutions)} disagreements.")
            if resolutions:
                st.dataframe(pd.DataFrame(resolutions), use_container_width=True)

            if st.button("Re-run validation to confirm", type="secondary"):
                st.session_state.validation = None
                st.session_state.reconciliation = None
                st.session_state.coding_phase = "validate"
                st.rerun()

            if st.button("Proceed to Export", type="primary"):
                st.session_state.step = 5
                st.rerun()


# ===================================================================
# STEP 4 — (reserved, pipeline is in step 3)
# ===================================================================


# ===================================================================
# STEP 5 — Export
# ===================================================================
elif st.session_state.step == 5:
    st.header("5. Export")

    df = st.session_state.df
    frame = st.session_state.frame

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Coded Data")
        excel_buf = export_coded_excel(df)
        st.download_button(
            "Download Coded Data (Excel)",
            data=excel_buf,
            file_name="coded_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("Coding Frame")
        frame_buf = export_frame_excel(frame)
        st.download_button(
            "Download Coding Frame (Excel)",
            data=frame_buf,
            file_name="coding_frame.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        frame_json = export_frame_json(frame)
        st.download_button(
            "Download Coding Frame (JSON)",
            data=frame_json,
            file_name="coding_frame.json",
            mime="application/json",
        )

    with col2:
        if st.session_state.validation:
            st.subheader("Validation Report")
            val_buf = export_validation_report(st.session_state.validation)
            st.download_button(
                "Download Validation Report (Excel)",
                data=val_buf,
                file_name="validation_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.subheader("SPSS Syntax")
        text_cols = st.session_state.text_columns
        spss_blocks = []
        for text_col in text_cols:
            code_col = text_col.replace("t", "c") if text_col.endswith("t") else f"{text_col}_code"
            spss_blocks.append(generate_spss_syntax(frame, code_col))
        spss_text = "\n\n".join(spss_blocks)
        st.code(spss_text, language="text")
        st.download_button(
            "Download SPSS Syntax",
            data=spss_text,
            file_name="value_labels.sps",
            mime="text/plain",
        )

    st.divider()
    if st.button("Start new project"):
        for key in DEFAULTS:
            st.session_state[key] = DEFAULTS[key]
        st.rerun()
