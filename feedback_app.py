"""
Streamlit website for the employee feedback Human-in-the-Loop workflow.

Local:
    streamlit run langgraph/feedback_app.py

Streamlit Community Cloud:
    Main file: langgraph/feedback_app.py  (or feedback_app.py if this folder is the repo)
    Secrets:   GEMINI_API_KEY = "your-key"
    Python:    3.12 in Advanced settings
"""

from __future__ import annotations

import html
import io
import os
import re
import uuid

import streamlit as st

st.set_page_config(
    page_title="Employee Feedback Generator",
    page_icon="📋",
    layout="wide",
)

# Streamlit Cloud injects secrets here. Copy into the environment before
# importing the graph so ChatGoogleGenerativeAI can see the key.
if not os.getenv("GEMINI_API_KEY"):
    try:
        _secret_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        _secret_key = None
    if _secret_key:
        os.environ["GEMINI_API_KEY"] = str(_secret_key)

try:
    from feedback_graph import (
        build_final_sheet,
        get_interrupt_questions,
        resume_review,
        start_review,
    )
except ValueError as exc:
    st.error(str(exc))
    st.info(
        "On Streamlit Cloud: App settings → Secrets, then add:\n\n"
        'GEMINI_API_KEY = "your-gemini-api-key"\n\n'
        "Locally: put the same key in `langgraph/.env`."
    )
    st.stop()

STEP_ORDER = ("form", "questions", "results")
STEP_LABELS = {
    "form": "1. Details",
    "questions": "2. Questions",
    "results": "3. Results",
}

GUIDED_SECTIONS = (
    ("form_obs_learning", "Learning", "Technologies learned, adaptability, learning effort…"),
    ("form_obs_organization", "Organization", "Transparency, openness, focus, org processes…"),
    (
        "form_obs_project",
        "Project Execution",
        "Delivery time, quality, budget, ownership…",
    ),
    ("form_obs_customer", "Customer", "Responsiveness, relations, communication…"),
    (
        "form_obs_leadership",
        "Leadership",
        "Proactiveness, mentoring, social skills, motivation, empathy…",
    ),
)

# Durable defaults for form widgets. Streamlit drops widget keys when they unmount;
# we mirror values into persisted_form / persisted_answers so Back restores them.
FORM_DEFAULTS = {
    "form_employee_name": "",
    "form_role": "Software Engineer",
    "form_experience": "1+ years",
    "form_obs_general": "",
    "form_obs_learning": "",
    "form_obs_organization": "",
    "form_obs_project": "",
    "form_obs_customer": "",
    "form_obs_leadership": "",
    "form_previous_year": "",
}


def _persist_inputs() -> None:
    """Copy currently mounted widget values into durable session stores."""
    persisted_form = st.session_state.setdefault("persisted_form", {})
    for key in FORM_DEFAULTS:
        if key in st.session_state:
            persisted_form[key] = st.session_state[key]

    persisted_answers = st.session_state.setdefault("persisted_answers", {})
    for key in list(st.session_state.keys()):
        if str(key).startswith("answer_q_"):
            persisted_answers[key] = st.session_state[key]


def _restore_inputs() -> None:
    """Restore widget keys cleared by Streamlit when steps unmount."""
    persisted_form = st.session_state.get("persisted_form", {})
    for key, default in FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = persisted_form.get(key, default)

    persisted_answers = st.session_state.get("persisted_answers", {})
    for key, value in persisted_answers.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_persisted_answers() -> None:
    st.session_state.persisted_answers = {}
    for key in list(st.session_state.keys()):
        if str(key).startswith("answer_q_"):
            del st.session_state[key]


def _init_session() -> None:
    defaults = {
        "step": "form",
        "max_step_reached": "form",
        "thread_id": None,
        "questions": [],
        "questions_needed": False,
        "employee_name": "",
        "previous_year_feedback": "",
        "reviewer_answers": "",
        "final_feedback": None,
        "sheet_df": None,
        "evidence_analysis": None,
        "reviewer_voice": None,
        "validation_result": None,
        "error": None,
        "persisted_form": {},
        "persisted_answers": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _restore_inputs()


def _step_index(step: str) -> int:
    try:
        return STEP_ORDER.index(step)
    except ValueError:
        return 0


def _unlock_step(step: str) -> None:
    if _step_index(step) > _step_index(st.session_state.max_step_reached):
        st.session_state.max_step_reached = step


def _go_to(step: str) -> None:
    if _step_index(step) <= _step_index(st.session_state.max_step_reached):
        _persist_inputs()
        st.session_state.step = step
        st.rerun()


def _set_step(step: str) -> None:
    """Stepper callback — navigation only, never re-runs the LLM."""
    if _step_index(step) <= _step_index(st.session_state.max_step_reached):
        _persist_inputs()
        st.session_state.step = step


def _reset() -> None:
    st.session_state.step = "form"
    st.session_state.max_step_reached = "form"
    st.session_state.thread_id = None
    st.session_state.questions = []
    st.session_state.questions_needed = False
    st.session_state.employee_name = ""
    st.session_state.previous_year_feedback = ""
    st.session_state.reviewer_answers = ""
    st.session_state.final_feedback = None
    st.session_state.sheet_df = None
    st.session_state.evidence_analysis = None
    st.session_state.reviewer_voice = None
    st.session_state.validation_result = None
    st.session_state.error = None
    st.session_state.persisted_form = {}
    for key, default in FORM_DEFAULTS.items():
        st.session_state[key] = default
    _clear_persisted_answers()


def _make_thread_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "employee"
    return f"employee-feedback-{slug}-{uuid.uuid4().hex[:8]}"


def _build_observations(
    general: str,
    learning: str,
    organization: str,
    project: str,
    customer: str,
    leadership: str,
) -> str:
    parts: list[str] = []
    if general.strip():
        parts.append(general.strip())

    for title, text in (
        ("Learning", learning),
        ("Organization", organization),
        ("Project Execution", project),
        ("Customer", customer),
        ("Leadership", leadership),
    ):
        if text.strip():
            parts.append(f"{title.upper()}:\n{text.strip()}")

    return "\n\n".join(parts).strip()


def _store_transparency(result: dict) -> None:
    if result.get("evidence_analysis") is not None:
        st.session_state.evidence_analysis = result["evidence_analysis"]
    if result.get("reviewer_voice") is not None:
        st.session_state.reviewer_voice = result["reviewer_voice"]
    if result.get("validation_result") is not None:
        st.session_state.validation_result = result["validation_result"]


def _finish_with_result(result: dict) -> None:
    _store_transparency(result)
    final_feedback = result["final_feedback"]
    previous_year = st.session_state.previous_year_feedback
    sheet_df = build_final_sheet(final_feedback, previous_year)
    st.session_state.final_feedback = final_feedback
    st.session_state.sheet_df = sheet_df
    st.session_state.step = "results"
    _unlock_step("results")
    st.session_state.error = None


def _run_with_status(label: str, runner):
    """Run a graph stream inside st.status, updating on each node."""
    with st.status(label, expanded=True) as status:
        st.write("Starting…")

        def on_node(_node_name: str, node_label: str) -> None:
            st.write(node_label)
            status.update(label=node_label)

        result = runner(on_node)
        status.update(label="Done", state="complete")
        return result


def _status_caption() -> str:
    step = st.session_state.step
    if step == "form":
        if st.session_state.final_feedback is not None:
            return "Status: feedback already generated — submitting again will regenerate."
        return "Status: enter employee details and observations."
    if step == "questions":
        if not st.session_state.questions_needed:
            return "Status: no follow-up questions were needed for this review."
        if st.session_state.final_feedback is not None:
            return "Status: viewing saved questions (answers already submitted)."
        return "Status: paused for questions — answer to continue."
    return "Status: feedback ready — edit the sheet, then copy or download."


def _render_stepper() -> None:
    max_idx = _step_index(st.session_state.max_step_reached)
    cols = st.columns(3)
    for i, step_id in enumerate(STEP_ORDER):
        label = STEP_LABELS[step_id]
        unlocked = i <= max_idx
        is_current = st.session_state.step == step_id

        questions_skipped = (
            step_id == "questions"
            and not st.session_state.questions_needed
            and st.session_state.final_feedback is not None
        )

        with cols[i]:
            if questions_skipped:
                st.button(
                    f"{label} (not needed)",
                    key=f"stepper_{step_id}",
                    disabled=True,
                    use_container_width=True,
                )
            elif unlocked:
                st.button(
                    f"{'● ' if is_current else ''}{label}",
                    key=f"stepper_{step_id}",
                    type="primary" if is_current else "secondary",
                    use_container_width=True,
                    on_click=_set_step,
                    args=(step_id,),
                )
            else:
                st.button(
                    label,
                    key=f"stepper_{step_id}",
                    disabled=True,
                    use_container_width=True,
                )

    st.caption(_status_caption())


def _render_transparency_panel() -> None:
    evidence = st.session_state.evidence_analysis
    voice = st.session_state.reviewer_voice
    validation = st.session_state.validation_result

    if not evidence and not voice and not validation:
        return

    with st.expander("Evidence, voice & validation", expanded=False):
        if evidence:
            st.markdown("**Confirmed strengths**")
            strengths = evidence.get("confirmed_strengths") or []
            if strengths:
                for item in strengths:
                    st.markdown(f"- {item}")
            else:
                st.caption("None listed.")

            st.markdown("**Missing information**")
            missing = evidence.get("missing_information") or []
            if missing:
                for item in missing:
                    st.markdown(f"- {item}")
            else:
                st.caption("None listed.")

        if voice:
            st.markdown("**Reviewer voice profile**")
            st.markdown(
                f"- Formality: `{voice.get('formality', '')}`  \n"
                f"- Person: `{voice.get('person', '')}`  \n"
                f"- Tone: `{voice.get('tone', '')}`  \n"
                f"- Sentence style: {voice.get('sentence_style', '')}"
            )
            phrases = voice.get("typical_phrases") or []
            if phrases:
                st.markdown("Typical phrases: " + ", ".join(f"`{p}`" for p in phrases))
            if voice.get("style_notes"):
                st.caption(voice["style_notes"])

        if validation:
            issues = validation.get("issues") or []
            st.markdown("**Validation issues**")
            if issues:
                for issue in issues:
                    st.markdown(f"- {issue}")
            else:
                st.caption("No issues reported.")
            approved = validation.get("approved")
            if approved is not None:
                st.caption(f"Approved flag: `{approved}`")


def _sheet_html(sheet_df) -> str:
    columns = list(sheet_df.columns)
    header_cells = "".join(
        f"<th>{html.escape(str(col))}</th>" for col in columns
    )
    body_rows = []
    for _, row in sheet_df.iterrows():
        cells = "".join(
            f"<td>{html.escape(str(row[col]) if row[col] is not None else '')}</td>"
            for col in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return f"""
<style>
.feedback-sheet-wrap {{
  max-height: 70vh;
  overflow: auto;
  border: 1px solid rgba(128,128,128,0.35);
  border-radius: 8px;
}}
.feedback-sheet {{
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 0.92rem;
}}
.feedback-sheet th, .feedback-sheet td {{
  border: 1px solid rgba(128,128,128,0.35);
  padding: 10px 12px;
  vertical-align: top;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.45;
}}
.feedback-sheet th {{
  position: sticky;
  top: 0;
  background: var(--background-color, #1e1e1e);
  text-align: left;
  font-weight: 600;
  z-index: 1;
}}
.feedback-sheet td:nth-child(1),
.feedback-sheet th:nth-child(1) {{ width: 12%; }}
.feedback-sheet td:nth-child(2),
.feedback-sheet th:nth-child(2) {{ width: 14%; }}
.feedback-sheet td:nth-child(3),
.feedback-sheet th:nth-child(3) {{ width: 18%; }}
.feedback-sheet td:nth-child(4),
.feedback-sheet th:nth-child(4) {{ width: 28%; }}
.feedback-sheet td:nth-child(5),
.feedback-sheet th:nth-child(5) {{ width: 28%; }}
</style>
<div class="feedback-sheet-wrap">
  <table class="feedback-sheet">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {"".join(body_rows)}
    </tbody>
  </table>
</div>
"""


def _render_form() -> None:
    st.header("Step 1 — Employee details & observations")
    st.caption(
        "Use general notes plus optional area sections. "
        "The app will ask follow-up questions only if needed, then generate sheet-ready feedback."
    )

    if st.session_state.final_feedback is not None:
        st.warning(
            "Feedback already exists for this session. "
            "Submitting again will regenerate feedback and replace the current results."
        )

    with st.form("employee_form"):
        col1, col2 = st.columns(2)
        with col1:
            employee_name = st.text_input(
                "Employee name *",
                key="form_employee_name",
                placeholder="e.g. Mohit",
            )
            employee_role = st.text_input(
                "Role *",
                key="form_role",
                placeholder="e.g. Software Engineer",
            )
        with col2:
            experience = st.text_input(
                "Experience *",
                key="form_experience",
                placeholder="e.g. 1+ years",
            )

        general = st.text_area(
            "General notes *",
            height=160,
            key="form_obs_general",
            placeholder=(
                "Write overall observations: strengths, gaps, whether you worked "
                "directly with them, anything that does not fit a single area."
            ),
        )

        st.markdown("**Optional area notes** (helps fill the sheet)")
        for key, title, placeholder in GUIDED_SECTIONS:
            with st.expander(title, expanded=False):
                st.text_area(
                    title,
                    height=100,
                    key=key,
                    placeholder=placeholder,
                    label_visibility="collapsed",
                )

        previous_year_feedback = st.text_area(
            "Previous-year feedback (optional)",
            height=120,
            key="form_previous_year",
            placeholder=(
                "Paste previous-year review notes if available.\n"
                "Leave empty for a first-year review."
            ),
        )

        submitted = st.form_submit_button("Analyze & continue", type="primary")

    nav_cols = st.columns([1, 3])
    with nav_cols[0]:
        if st.session_state.final_feedback is not None:
            if st.button("View results →", use_container_width=True):
                _go_to("results")

    if not submitted:
        return

    observations = _build_observations(
        general,
        st.session_state.get("form_obs_learning", ""),
        st.session_state.get("form_obs_organization", ""),
        st.session_state.get("form_obs_project", ""),
        st.session_state.get("form_obs_customer", ""),
        st.session_state.get("form_obs_leadership", ""),
    )

    if not employee_name.strip() or not general.strip():
        st.error("Employee name and general notes are required.")
        return

    _persist_inputs()
    _clear_persisted_answers()

    thread_id = _make_thread_id(employee_name)
    state = {
        "employee_name": employee_name.strip(),
        "employee_role": (employee_role or "").strip() or "Software Engineer",
        "experience": (experience or "").strip() or "1+ years",
        "reviewer_observations": observations,
        "previous_year_feedback": (previous_year_feedback or "").strip(),
    }

    st.session_state.thread_id = thread_id
    st.session_state.employee_name = employee_name.strip()
    st.session_state.previous_year_feedback = (previous_year_feedback or "").strip()
    st.session_state.questions = []
    st.session_state.questions_needed = False
    st.session_state.reviewer_answers = ""
    st.session_state.final_feedback = None
    st.session_state.sheet_df = None
    st.session_state.evidence_analysis = None
    st.session_state.reviewer_voice = None
    st.session_state.validation_result = None
    st.session_state.max_step_reached = "form"

    try:
        result = _run_with_status(
            "Analyzing evidence…",
            lambda on_node: start_review(state, thread_id, on_node=on_node),
        )
    except Exception as exc:  # noqa: BLE001
        st.session_state.error = str(exc)
        st.error(f"Something went wrong: {exc}")
        return

    _store_transparency(result)

    questions = get_interrupt_questions(result)
    if questions:
        st.session_state.questions = questions
        st.session_state.questions_needed = True
        st.session_state.step = "questions"
        _unlock_step("questions")
        st.rerun()

    if "final_feedback" in result:
        st.session_state.questions_needed = False
        st.session_state.reviewer_answers = "No additional information was required."
        try:
            _finish_with_result(result)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to build sheet: {exc}")
            return
        st.rerun()

    st.error("Unexpected graph result — no questions and no final feedback.")


def _collect_per_question_answers() -> str:
    parts = []
    for i, question in enumerate(st.session_state.questions, start=1):
        answer = (st.session_state.get(f"answer_q_{i}", "") or "").strip()
        if answer:
            parts.append(f"{i}. {answer}")
        else:
            parts.append(f"{i}. (no answer) — {question}")
    return "\n".join(parts)


def _render_questions() -> None:
    st.header(f"Step 2 — Follow-up questions for {st.session_state.employee_name}")

    back_cols = st.columns([1, 1, 2])
    with back_cols[0]:
        if st.button("← Back to details", use_container_width=True):
            _go_to("form")
    with back_cols[1]:
        if st.session_state.final_feedback is not None:
            if st.button("View results →", use_container_width=True):
                _go_to("results")

    _render_transparency_panel()

    if not st.session_state.questions_needed:
        st.info("No follow-up questions were needed for this review.")
        return

    already_done = st.session_state.final_feedback is not None
    if already_done:
        st.info(
            "Answers were already submitted and feedback was generated. "
            "You can view them below; use Step 1 to regenerate."
        )
        if st.session_state.reviewer_answers:
            st.text_area(
                "Submitted answers",
                value=st.session_state.reviewer_answers,
                height=220,
                disabled=True,
            )
        return

    st.caption(
        "At most a few grouped follow-up questions (not one per Excel row). "
        "Your original notes fill matching sheet cells. Skip anything you do not know — "
        "those cells stay blank."
    )

    for i, question in enumerate(st.session_state.questions, start=1):
        st.markdown(f"**{i}. {question}**")
        st.text_area(
            f"Answer {i}",
            height=90,
            key=f"answer_q_{i}",
            label_visibility="collapsed",
            placeholder="Your answer (optional)",
        )

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        submit = st.button("Submit answers", type="primary", use_container_width=True)
    with col2:
        skip = st.button("Skip — no more info", use_container_width=True)

    if not submit and not skip:
        return

    if skip:
        resume_text = "No additional information provided."
    else:
        resume_text = _collect_per_question_answers()

    _persist_inputs()
    st.session_state.reviewer_answers = resume_text

    try:
        result = _run_with_status(
            "Writing review…",
            lambda on_node: resume_review(
                resume_text,
                st.session_state.thread_id,
                on_node=on_node,
            ),
        )
        _finish_with_result(result)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong: {exc}")
        return

    st.rerun()


def _render_results() -> None:
    st.header(f"Step 3 — Feedback for {st.session_state.employee_name}")

    nav_cols = st.columns([1, 1, 2])
    with nav_cols[0]:
        if st.button("← Back to details", use_container_width=True):
            _go_to("form")
    with nav_cols[1]:
        if st.session_state.questions_needed:
            if st.button("← View questions", use_container_width=True):
                _go_to("questions")

    final_feedback = st.session_state.final_feedback
    sheet_df = st.session_state.sheet_df

    if final_feedback:
        with st.expander("Employee summary", expanded=True):
            st.write(final_feedback.get("employee_summary", ""))
        with st.expander("Overall feedback", expanded=True):
            st.write(final_feedback.get("overall_feedback", ""))

    _render_transparency_panel()

    if st.session_state.reviewer_answers:
        with st.expander("Your answers (audit trail)", expanded=False):
            st.text(st.session_state.reviewer_answers)

    st.subheader("Sheet-ready table")
    st.caption(
        "Edit Previous Year / Good / Can Improve below, then copy or download. "
        "Area and Sub Areas stay fixed."
    )

    if sheet_df is not None:
        edited_df = st.data_editor(
            sheet_df,
            disabled=["Area", "Sub Areas"],
            use_container_width=True,
            hide_index=True,
            key="sheet_editor",
            height=520,
            column_config={
                "Area": st.column_config.TextColumn(width="small"),
                "Sub Areas": st.column_config.TextColumn(width="medium"),
                "Previous Year": st.column_config.TextColumn(width="large"),
                "Good": st.column_config.TextColumn(width="large"),
                "Can Improve": st.column_config.TextColumn(width="large"),
            },
        )
        st.session_state.sheet_df = edited_df

        with st.expander("Readable preview", expanded=False):
            st.markdown(_sheet_html(edited_df), unsafe_allow_html=True)

        tsv = edited_df.to_csv(sep="\t", index=False)
        st.text_area(
            "Copy for Google Sheets (select all, then paste)",
            value=tsv,
            height=160,
            key="sheets_tsv_copy",
        )

        excel_buffer = io.BytesIO()
        edited_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
        safe_name = st.session_state.employee_name.replace(" ", "_") or "employee"

        st.markdown("---")
        action_cols = st.columns([1, 1, 1, 2])
        with action_cols[0]:
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name=f"{safe_name}_feedback.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with action_cols[1]:
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"{safe_name}_feedback.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with action_cols[2]:
            confirm_reset = st.checkbox("Confirm reset", key="confirm_reset")
            if st.button(
                "Start another review",
                disabled=not confirm_reset,
                use_container_width=True,
            ):
                _reset()
                st.rerun()


def main() -> None:
    _init_session()

    st.title("Employee Feedback Generator")
    st.markdown(
        "Human-in-the-loop LangGraph + Gemini — enter observations, "
        "answer questions if asked, get sheet-format feedback."
    )

    _render_stepper()
    st.divider()

    if st.session_state.step == "form":
        _render_form()
    elif st.session_state.step == "questions":
        _render_questions()
    elif st.session_state.step == "results":
        _render_results()
    else:
        st.session_state.step = "form"
        st.rerun()


if __name__ == "__main__":
    main()
