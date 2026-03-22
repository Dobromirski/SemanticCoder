"""SemanticCoder — Streamlit app for semantic coding of open-ended survey responses."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from modules.claude_client import ClaudeClient
from modules.auditor import audit_other_codes, reconcile_disagreements
from modules.calibrator import extract_rules_from_corrections, format_rules_for_approval, merge_approved_rules
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
    "coding_phase": "code",  # code → review_code → audit_98 → review_audit → validate → reconcile → done
    "audit_98_result": None,
    "validation": None,
    "reconciliation": None,
    "calibration_sample": None,
    "calibration_extracted": None,
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
                        # Validate frame has codes
                        if not frame.get("codes"):
                            st.error("Frame returned with no codes. Raw response saved in debug.")
                            st.session_state["_debug_frame"] = frame
                        else:
                            st.session_state.frame = frame
                            st.session_state.frame_history.append(frame)
                            st.session_state.frame_iteration = iteration + 1
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")

        # Debug: show raw response if frame was empty
        if st.session_state.get("_debug_frame") is not None:
            with st.expander("Debug: raw Claude response"):
                st.json(st.session_state["_debug_frame"])

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

    # ----- Helper: render calibration review panel -----
    def _render_calibration_panel(
        phase_name: str,
        text_col: str,
        code_col: str,
        next_phase: str,
        next_label: str,
    ):
        """Show 30 random coded items for human review, extract rules from corrections."""
        st.subheader(f"Calibration — Review 30 random codings ({phase_name})")
        st.caption(
            "Review these randomly selected codings. Correct any you disagree with. "
            "The system will analyze your corrections and propose new decision rules."
        )

        # Generate sample once
        if st.session_state.calibration_sample is None:
            valid = df[df[code_col].notna() & ~df[code_col].isin([99])].copy()
            n = min(30, len(valid))
            sample = valid.sample(n=n, random_state=42)

            # Build frame lookup for labels
            code_labels = {c["code"]: c["label"] for c in frame.get("codes", [])}
            code_labels[98] = "Other"
            code_labels[99] = "DK/NA"

            review_data = []
            for idx, row in sample.iterrows():
                code_val = int(row[code_col])
                review_data.append({
                    "row_id": int(idx),
                    "text": str(row[text_col]).strip(),
                    "assigned_code": code_val,
                    "label": code_labels.get(code_val, f"Code {code_val}"),
                    "your_code": code_val,
                })
            st.session_state.calibration_sample = review_data

        review_data = st.session_state.calibration_sample

        # Build code options for selectbox
        code_options = {c["code"]: f"{c['code']} — {c['label']}" for c in frame.get("codes", [])}
        code_options[98] = "98 — Other"
        code_options[99] = "99 — DK/NA"
        code_list = sorted(code_options.keys())

        # Editable review table
        st.write(f"**{len(review_data)} items to review:**")

        corrections = []
        for i, item in enumerate(review_data):
            with st.container():
                cols = st.columns([4, 2, 2])
                with cols[0]:
                    st.write(f"**[{item['row_id']}]** {item['text']}")
                with cols[1]:
                    st.caption(f"Assigned: {item['assigned_code']} — {item['label']}")
                with cols[2]:
                    new_code = st.selectbox(
                        "Your code",
                        options=code_list,
                        format_func=lambda x: code_options.get(x, str(x)),
                        index=code_list.index(item["assigned_code"]) if item["assigned_code"] in code_list else 0,
                        key=f"cal_{phase_name}_{i}",
                        label_visibility="collapsed",
                    )
                    if new_code != item["assigned_code"]:
                        corrections.append({
                            "row_id": item["row_id"],
                            "text": item["text"],
                            "original_code": item["assigned_code"],
                            "corrected_code": new_code,
                        })

        st.divider()

        if corrections:
            st.info(f"You corrected **{len(corrections)}** out of {len(review_data)} items.")

            # Extract rules button
            if st.session_state.calibration_extracted is None:
                if st.button("Analyze my corrections", type="primary"):
                    client = get_client()
                    with st.spinner("Extracting patterns from your corrections..."):
                        extracted = extract_rules_from_corrections(
                            client, corrections, len(review_data), frame,
                        )
                        st.session_state.calibration_extracted = extracted
                        st.rerun()
            else:
                extracted = st.session_state.calibration_extracted
                rules = extracted.get("extracted_rules", [])

                if rules:
                    st.markdown(format_rules_for_approval(extracted))

                    # Checkboxes to approve/reject each rule
                    approved = []
                    for j, rule in enumerate(rules):
                        checked = st.checkbox(
                            f"Approve: {rule.get('rule', '')}",
                            value=rule.get("confidence") in ("HIGH", "MEDIUM"),
                            key=f"approve_rule_{phase_name}_{j}",
                        )
                        if checked:
                            approved.append(rule)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Apply approved rules & continue", type="primary"):
                            # Apply corrections to df
                            for c in corrections:
                                df.at[c["row_id"], code_col] = int(c["corrected_code"])
                            st.session_state.df = df

                            # Merge approved rules into frame
                            if approved:
                                updated_frame = merge_approved_rules(frame, approved)
                                st.session_state.frame = updated_frame

                            st.session_state.calibration_sample = None
                            st.session_state.calibration_extracted = None
                            st.session_state.coding_phase = next_phase
                            st.rerun()
                    with col_b:
                        if st.button("Skip rules, just apply corrections"):
                            for c in corrections:
                                df.at[c["row_id"], code_col] = int(c["corrected_code"])
                            st.session_state.df = df
                            st.session_state.calibration_sample = None
                            st.session_state.calibration_extracted = None
                            st.session_state.coding_phase = next_phase
                            st.rerun()
                else:
                    st.success("No systematic patterns found — your corrections appear to be case-specific.")
                    if st.button("Apply corrections & continue", type="primary"):
                        for c in corrections:
                            df.at[c["row_id"], code_col] = int(c["corrected_code"])
                        st.session_state.df = df
                        st.session_state.calibration_sample = None
                        st.session_state.calibration_extracted = None
                        st.session_state.coding_phase = next_phase
                        st.rerun()
        else:
            st.success("No corrections — all 30 items look good!")
            if st.button(next_label, type="primary"):
                st.session_state.calibration_sample = None
                st.session_state.calibration_extracted = None
                st.session_state.coding_phase = next_phase
                st.rerun()

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
            # All columns coded — show distribution and move to review
            st.success(f"Agent 1 complete — all {len(text_cols)} columns coded.")
            for tc in text_cols:
                cc = _code_col_name(tc)
                if cc in df.columns:
                    st.subheader(f"Distribution: {cc}")
                    dist = df[cc].value_counts().sort_index()
                    n98 = int(dist.get(98, 0))
                    n99 = int(dist.get(99, 0))
                    total_rows = len(df)
                    st.bar_chart(dist)
                    if n98 + n99 > 0:
                        pct = (n98 + n99) / total_rows * 100
                        st.warning(f"Code 98/99: {n98 + n99} responses ({pct:.1f}%)")

            if st.button("Review 30 random codings before audit", type="primary"):
                st.session_state.coding_phase = "review_code"
                st.rerun()

    # --- PHASE 1b: Human review after initial coding ---
    elif phase == "review_code":
        text_col = text_cols[0]
        code_col = _code_col_name(text_col)
        _render_calibration_panel(
            phase_name="after_coding",
            text_col=text_col,
            code_col=code_col,
            next_phase="audit_98",
            next_label="Proceed to Agent 2 — Audit 98/99 codes",
        )

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
                    total_rows = len(df)
                    st.write(f"**{cc}** after audit: {n98} remaining code 98 ({n98/total_rows*100:.1f}%)")

            if st.button("Review 30 random codings before validation", type="primary"):
                st.session_state.coding_phase = "review_audit"
                st.rerun()

    # --- PHASE 2b: Human review after audit ---
    elif phase == "review_audit":
        text_col = text_cols[0]
        code_col = _code_col_name(text_col)
        _render_calibration_panel(
            phase_name="after_audit",
            text_col=text_col,
            code_col=code_col,
            next_phase="validate",
            next_label="Proceed to Agent 3 — Independent Validation",
        )

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
