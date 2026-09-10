"""
Employee feedback Human-in-the-Loop LangGraph (sheet-format output).

Extracted from employee_feedback_human_in_loop_sheet_format.ipynb for use
by the Streamlit app. The graph is compiled once at import so MemorySaver
checkpoints survive Streamlit reruns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# This project folder is named "langgraph", which shadows the installed
# langgraph package when Streamlit adds the script directory to sys.path.
# Import the real package with the local folder temporarily removed.
_HERE = Path(__file__).resolve().parent
_path_backup = list(sys.path)
try:
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != _HERE]
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
finally:
    sys.path[:] = _path_backup

def _get_gemini_api_key() -> str:
    """Read GEMINI_API_KEY from .env, the environment, or Streamlit secrets."""
    load_dotenv(_HERE / ".env")
    load_dotenv()  # also allow repo-root .env

    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if key:
        return key

    try:
        import streamlit as st

        secret = st.secrets.get("GEMINI_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass

    return ""


_api_key = _get_gemini_api_key()
if not _api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Add it to Streamlit Secrets, langgraph/.env, or the environment."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=_api_key,
    temperature=0.2,
)

# Same Flash model for voice analysis and sheet review writing (free-tier friendly).
review_llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=_api_key,
    temperature=0.35,
)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    fact: str = Field(
        description="A factual observation explicitly provided by the reviewer."
    )
    category: str = Field(
        description="The performance area related to the observation."
    )


class EvidenceAnalysis(BaseModel):
    confirmed_strengths: list[str] = Field(
        description="Strengths directly supported by reviewer observations."
    )
    confirmed_improvements: list[str] = Field(
        description="Improvement opportunities directly supported by observations."
    )
    missing_information: list[str] = Field(
        description="Important information missing before evaluating performance fairly."
    )
    evidence: list[Evidence] = Field(
        description="Individual facts extracted from reviewer observations."
    )


class MissingQuestions(BaseModel):
    questions: list[str] = Field(
        description=(
            "Between 0 and 6 grouped questions covering high-value sheet gaps. "
            "One question may cover several sub-areas in a category. "
            "Example: 'Customer (responsiveness, relations, communication): "
            "any observation this year?' Prefer fewer when notes already cover a lot."
        )
    )


class FeedbackItem(BaseModel):
    category: str = Field(
        description=(
            "Main category: Learning, Organization, Project Execution, "
            "Customer, or Leadership."
        )
    )
    sub_area: str
    good: str
    can_improve: str
    confidence: Literal["high", "medium", "limited_observation"]
    evidence_used: list[str]


class FeedbackResult(BaseModel):
    employee_summary: str
    feedback: list[FeedbackItem]
    overall_feedback: str


class ValidationResult(BaseModel):
    approved: bool
    issues: list[str]
    corrected_feedback: FeedbackResult


class PreviousYearMapping(BaseModel):
    rows: list[dict] = Field(
        description=(
            "List of mappings with exactly these keys: "
            "sub_area and previous_year. "
            "Use an empty string when no matching previous-year information exists."
        )
    )


class ReviewerVoice(BaseModel):
    formality: Literal["casual", "mixed", "formal"] = Field(
        description="How formal the reviewer writes."
    )
    person: Literal["first_name", "he_she", "mixed"] = Field(
        description=(
            "How the reviewer refers to the employee: by first name, "
            "he/she pronouns, or a mix."
        )
    )
    sentence_style: str = Field(
        description="Short note on sentence length and structure (how they write)."
    )
    typical_phrases: list[str] = Field(
        description=(
            "A few phrases that illustrate tone only — NOT slogans to paste "
            "into every feedback row."
        )
    )
    tone: Literal["direct", "warm", "mixed"] = Field(
        description="Overall tone of the reviewer's writing."
    )
    style_notes: str = Field(
        description=(
            "One-paragraph instruction: match how they write (directness, "
            "pronouns), polish grammar, and do not reuse the same phrase "
            "across many rows."
        )
    )


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class FeedbackState(TypedDict, total=False):
    employee_name: str
    employee_role: str
    experience: str
    reviewer_observations: str
    previous_year_feedback: str
    evidence_analysis: dict
    missing_questions: list[str]
    reviewer_answers: str
    combined_observations: str
    reviewer_voice: dict
    feedback_result: dict
    validation_result: dict
    final_feedback: dict


# ---------------------------------------------------------------------------
# Sheet structure
# ---------------------------------------------------------------------------

REVIEW_STRUCTURE = [
    ("Overall", ["Overall"]),
    (
        "Learning",
        [
            "Learnings",
            "Ability to learn",
            "Efforts put into learning",
            "Effects on quality or time",
        ],
    ),
    (
        "Organization",
        [
            "Transparency / Honesty",
            "Openness",
            "Flow & Focus",
            "Developing Competency",
            "Radical Candor",
            "Non-core responsibilities",
            "Org processes",
        ],
    ),
    (
        "Project Execution",
        [
            "Work output - time taken",
            "Work output - quality",
            "Work output - Budget",
        ],
    ),
    (
        "Customer",
        [
            "Responsiveness",
            "Relations & Appreciation",
            "Communication",
        ],
    ),
    (
        "Leadership",
        [
            "Pro activeness",
            "Team Management",
            "Social Skills",
            "Motivation",
            "Empathy",
        ],
    ),
]

SHEET_SUBAREAS = [
    sub_area for category, subareas in REVIEW_STRUCTURE for sub_area in subareas
]

# Official rubric: guiding question + bullets for each sheet sub-area.
REVIEW_DEFINITIONS: dict[str, dict[str, dict[str, object]]] = {
    "Learning": {
        "Learnings": {
            "question": "What new technologies or other learnings have happened?",
            "bullets": [
                "New technologies, tools, frameworks, languages",
                "New domain knowledge or processes",
                "New concepts or skills learned",
            ],
        },
        "Ability to learn": {
            "question": (
                "What instances show the ability to learn, unlearn, or adapt?"
            ),
            "bullets": [
                "Learned something unfamiliar",
                "Adapted to a new technology/project",
                "Changed approach based on feedback",
                "Learned from mistakes",
            ],
        },
        "Efforts put into learning": {
            "question": "What efforts were made to learn or improve?",
            "bullets": [
                "Courses, tutorials, documentation",
                "POCs and experiments",
                "Technical research",
                "Practicing or learning outside assigned work",
                "Seeking guidance or knowledge sharing",
            ],
        },
        "Effects on quality or time": {
            "question": (
                "What visible impact did the learning have on quality or "
                "execution time?"
            ),
            "bullets": [
                "Faster development/debugging",
                "Fewer bugs or less rework",
                "Better code/design",
                "More independent execution",
                "Improved performance or customer outcome",
            ],
        },
    },
    "Organization": {
        "Transparency / Honesty": {
            "question": "How transparent and honest was the employee?",
            "bullets": [
                "Communicating problems, mistakes, risks, and delays",
                "Providing accurate status",
                "Being honest in discussions and inquiries",
            ],
        },
        "Openness": {
            "question": "How open was the employee to feedback?",
            "bullets": [
                "Accepting feedback",
                "Acting on feedback",
                "Changing/improving based on suggestions",
            ],
        },
        "Flow & Focus": {
            "question": "How effectively does the employee focus on important work?",
            "bullets": [
                "Deep focus on complex problems",
                "Productive use of time",
                "Ability to work independently",
                "Strong output from focused work",
            ],
        },
        "Developing Competency": {
            "question": (
                "Is the employee at the expected level for their seniority "
                "and improving?"
            ),
            "bullets": [
                "Technical/domain knowledge",
                "Problem-solving",
                "Independence",
                "Handling increasingly complex work",
                "Improving engineering skills",
            ],
        },
        "Radical Candor": {
            "question": (
                "Does the employee give honest feedback while helping "
                "others improve?"
            ),
            "bullets": [
                "Constructive feedback",
                "Raising concerns directly",
                "Challenging ideas respectfully",
                "Helping others grow",
            ],
        },
        "Non-core responsibilities": {
            "question": (
                "What responsibilities were taken beyond the normal role?"
            ),
            "bullets": [
                "Mentoring",
                "Interviews/hiring",
                "Documentation",
                "Team activities",
                "Helping other teams",
                "Process improvements",
                "Other organizational contributions",
            ],
        },
        "Org processes": {
            "question": (
                "How well does the employee follow organizational processes?"
            ),
            "bullets": [
                "Planned leave and advance communication",
                "Daily status",
                "Weekly time tracking",
                "Reporting to project leads",
                "Following project/release processes",
            ],
        },
    },
    "Project Execution": {
        "Work output - time taken": {
            "question": "Are tasks completed efficiently?",
            "bullets": [
                "Meeting deadlines",
                "Fast execution",
                "Quick issue resolution",
                "Good estimation",
                "Avoiding unnecessary delays/rework",
            ],
        },
        "Work output - quality": {
            "question": "How good is the quality of work?",
            "bullets": [
                "Clean/maintainable code",
                "Low bugs and regressions",
                "Testing",
                "Performance",
                "Reliability",
                "Good technical decisions",
            ],
        },
        "Work output - Budget": {
            "question": (
                "How well was work delivered within expected resources/budget?"
            ),
            "bullets": [
                "Staying within estimated effort",
                "Avoiding unnecessary work/cost",
                "Efficient use of existing resources",
                "Reducing infrastructure/tool costs where applicable",
            ],
        },
    },
    "Customer": {
        "Responsiveness": {
            "question": (
                "How quickly and deeply does the employee respond to "
                "customer needs?"
            ),
            "bullets": [
                "Quickly: response time; handling urgent requests",
                "Deeply: understanding requirements; asking the right questions; "
                "identifying risks; advising on better solutions",
            ],
        },
        "Relations & Appreciation": {
            "question": "How good is the relationship with the customer?",
            "bullets": [
                "Customer trust",
                "Positive feedback",
                "Appreciation",
                "Handling difficult situations professionally",
            ],
        },
        "Communication": {
            "question": (
                "How proactively and clearly does the employee communicate?"
            ),
            "bullets": [
                "Problems",
                "Changes",
                "Delays",
                "Risks",
                "Requirements",
                "Progress",
                "Availability",
            ],
        },
    },
    "Leadership": {
        "Pro activeness": {
            "question": (
                "Does the employee identify what needs to be done and take "
                "action without waiting for instructions?"
            ),
            "bullets": [
                "Identifying problems",
                "Suggesting improvements",
                "Taking ownership",
                "Fixing issues/technical debt",
                "Automating work",
                "Taking initiative",
            ],
        },
        "Team Management": {
            "question": (
                "Where applicable, how well does the employee manage/support "
                "others?"
            ),
            "bullets": [
                "Coordination",
                "Delegation",
                "Mentoring",
                "Unblocking teammates",
                "Planning",
                "Conflict handling",
            ],
        },
        "Social Skills": {
            "question": (
                "How well does the employee build positive working relationships?"
            ),
            "bullets": [
                "Collaboration",
                "Approachability",
                "Team relationships",
                "Cross-team communication",
                "Handling disagreements professionally",
            ],
        },
        "Motivation": {
            "question": (
                "Does the employee strive for excellent work beyond minimum "
                "requirements?"
            ),
            "bullets": [
                "Going beyond assigned work",
                "Voluntary learning",
                "Improving quality",
                "Taking pride in work",
                "Solving difficult problems without being pushed",
            ],
        },
        "Empathy": {
            "question": (
                "How well does the employee understand and consider other "
                "people's perspectives?"
            ),
            "bullets": [
                "Understanding customer/team pressure",
                "Considering others' constraints",
                "Adjusting communication",
                "Helping others",
                "Handling disagreements thoughtfully",
            ],
        },
    },
}


def _format_review_definitions() -> str:
    lines: list[str] = []
    for category, subareas in REVIEW_DEFINITIONS.items():
        lines.append(f"## {category}")
        for sub_area, meta in subareas.items():
            lines.append(f"### {sub_area}")
            lines.append(f"Guiding question: {meta['question']}")
            lines.append("Evaluate using these bullets:")
            for bullet in meta["bullets"]:
                lines.append(f"- {bullet}")
            lines.append("")
    return "\n".join(lines).strip()


REVIEW_DEFINITIONS_TEXT = _format_review_definitions()


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def analyze_evidence(state: FeedbackState):
    structured_llm = llm.with_structured_output(EvidenceAnalysis)

    prompt = f"""
You are an objective employee performance review assistant.

Analyze the reviewer's observations against the OFFICIAL REVIEW DEFINITIONS below.
Map confirmed strengths and missing information to those definition bullets
(not just row titles).

OFFICIAL REVIEW DEFINITIONS:
{REVIEW_DEFINITIONS_TEXT}

STRICT RULES:

1. Only treat explicitly provided information as confirmed evidence.
2. Do not invent achievements.
3. Do not assume code quality, delivery speed, customer satisfaction, or
   leadership ability without evidence that matches the definition bullets.
4. If the reviewer has not worked directly with the employee,
   mark relevant definition areas as requiring more information.
5. Match expectations to the employee's experience level.
6. In missing_information, list specific definition gaps
   (e.g. "Org processes: weekly time tracking", "Customer: Relations & Appreciation").

Employee:
Name: {state["employee_name"]}
Role: {state["employee_role"]}
Experience: {state["experience"]}

Reviewer observations:
{state["reviewer_observations"]}
"""

    result = structured_llm.invoke(prompt)
    return {"evidence_analysis": result.model_dump()}


def generate_missing_questions(state: FeedbackState):
    structured_llm = llm.with_structured_output(MissingQuestions)

    prompt = f"""
You are helping a reviewer finish an employee performance review with as few
follow-up questions as possible.

Use the OFFICIAL REVIEW DEFINITIONS as the checklist. Ask only about
definition bullets that notes do NOT already cover. The writer will map
existing notes onto matching rows — you do NOT need one question per Excel row.

OFFICIAL REVIEW DEFINITIONS:
{REVIEW_DEFINITIONS_TEXT}

Ask ONLY for remaining high-value holes, GROUPED by sheet category
(Learning / Organization / Project Execution / Customer / Leadership).

RULES:
1. Prefer 0–3 questions if the notes already cover a lot of definition bullets.
2. Never ask more than 6 questions.
3. One question may cover several related sub-areas / bullets, e.g.
   "Customer (Responsiveness quick+deep, Relations, Communication):
   any observation this year?"
4. Target missing DEFINITION bullets, not empty cell titles alone.
5. Do NOT ask a separate question for every empty sub-area.
6. Do NOT ask Previous Year questions.
7. Skip anything already answered in the observations.
8. Keep questions short and concrete.
9. If evidence is already strong enough for a useful sheet, return an empty list.

Employee:
Name: {state["employee_name"]}
Role: {state["employee_role"]}
Experience: {state["experience"]}

Reviewer observations:
{state["reviewer_observations"]}

Evidence analysis:
{state["evidence_analysis"]}
"""

    result = structured_llm.invoke(prompt)
    return {"missing_questions": result.questions}


def ask_reviewer(state: FeedbackState):
    questions = state.get("missing_questions", [])

    if not questions:
        return {"reviewer_answers": "No additional information was required."}

    payload = {
        "message": (
            "Please answer the following questions. "
            "You can answer all of them together in one response."
        ),
        "questions": questions,
        "employee_name": state["employee_name"],
    }

    reviewer_answers = interrupt(payload)
    return {"reviewer_answers": reviewer_answers}


def merge_reviewer_information(state: FeedbackState):
    combined = f"""
ORIGINAL REVIEWER OBSERVATIONS:
{state["reviewer_observations"]}

ADDITIONAL REVIEWER ANSWERS:
{state.get("reviewer_answers", "No additional answers provided.")}
"""
    return {"combined_observations": combined}


def analyze_reviewer_voice(state: FeedbackState):
    structured_llm = review_llm.with_structured_output(ReviewerVoice)
    previous_year = state.get("previous_year_feedback", "") or ""

    prompt = f"""
Analyze HOW this reviewer writes. Do not invent facts about the employee.
Only describe the reviewer's wording and style.

PRIMARY SOURCE (prefer this if styles conflict):
Reviewer observations:
{state["reviewer_observations"]}

Follow-up answers:
{state.get("reviewer_answers", "None")}

SECONDARY STYLE HINT (use only if helpful; prefer current observations if different):
Previous-year feedback:
{previous_year}

Describe:
- formality (casual / mixed / formal)
- person (first_name / he_she / mixed)
- sentence_style (short note)
- typical_phrases (a few words/phrases that show their tone — examples only, NOT slogans to paste into every row)
- tone (direct / warm / mixed)
- style_notes (one paragraph instructing a writer how to sound like this reviewer:
  capture HOW they write — directness, short sentences, he/she vs name —
  explicitly say NOT to reuse the same distinctive phrase on many rows,
  and to polish grammar while staying in their voice)

Do not invent phrases they did not use.
"""

    result = structured_llm.invoke(prompt)
    return {"reviewer_voice": result.model_dump()}


def generate_sheet_feedback(state: FeedbackState):
    structured_llm = review_llm.with_structured_output(FeedbackResult)
    previous_year = state.get("previous_year_feedback", "") or ""
    reviewer_voice = state.get("reviewer_voice", {}) or {}

    prompt = f"""
Write this performance review AS THIS REVIEWER — matching their wording and style.
Do not write as a generic senior engineering manager or HR template.

You are writing the performance review using the OFFICIAL REVIEW DEFINITIONS
as the scoring rubric. Each Excel row must answer that row's guiding question
and be judged only against that row's bullets — row titles alone are not enough.

OFFICIAL REVIEW DEFINITIONS (SOURCE OF TRUTH):
{REVIEW_DEFINITIONS_TEXT}

EMPLOYEE
Name: {state["employee_name"]}
Role: {state["employee_role"]}
Experience: {state["experience"]}

REVIEWER VOICE PROFILE:
{reviewer_voice}

ORIGINAL REVIEWER OBSERVATIONS (primary style source):
{state["reviewer_observations"]}

CURRENT REVIEWER OBSERVATIONS (merged with follow-up answers):
{state["combined_observations"]}

PREVIOUS YEAR FEEDBACK:
{previous_year}

EVIDENCE ANALYSIS:
{state["evidence_analysis"]}

VOICE AND STYLE:
- Sound like this reviewer: same person/pronouns, similar directness and simplicity.
- ALSO use clear grammar, spelling, and complete sentences
  (e.g. "doesnot" → "does not", "isnt" → "isn't", fix missing verbs).
- typical_phrases from the voice profile are tone examples only —
  do NOT paste them into many rows.
- Do NOT reuse the same distinctive phrase (e.g. "which is new to him")
  as identical wording in more than ONE row — rephrase the angle instead.
- Prefer the reviewer's meaning and facts, not raw copy-paste of their notes.
- Sound like a human manager writing a review, not a template and not a dump
  of unedited notes.
- Do not invent a heavy corporate/HR tone if the reviewer writes simply.
- If previous-year style differs from current observations, prefer current observations.

WRITE EACH ROW FROM ITS DEFINITION:
- For each of the 22 sub-areas, answer that row's GUIDING QUESTION.
- Good = evidence that matches THAT row's bullets
  (new tech, POCs, deadlines, customer trust, weekly time tracking, etc.).
- Can Improve = a development point that also matches THOSE SAME bullets —
  not a random weakness copied from another row.
- Do not put a Learnings fact into Org Processes, or a timesheet fact into
  Empathy, unless the definition bullets honestly overlap.

SPREAD NOTES + FOLLOW-UP ANSWERS (MAXIMIZE COVERAGE WITHIN THE RUBRIC):
- Use ALL notes and answers. Do not drop facts that match a definition bullet
  (e.g. English clarity → Communication; mentoring → Non-core and/or Team Management).
- Map the SAME fact onto multiple rows ONLY when each row's definition fits,
  with a DIFFERENT angle (not the same sentence):
  - "Golang is new" → Learnings (new language), Ability to learn (adapted to
    unfamiliar stack). Efforts only if courses/POCs/docs/practice were mentioned.
    Effects only if speed/quality/independence impact was mentioned.
  - Java/backend knowledge → Developing Competency; Work output - quality only
    if code/quality was mentioned.
  - Mentor → Non-core responsibilities and/or Team Management — not automatically Empathy.
  - Timesheet → Org processes (weekly time tracking).
  - WFH / non-core → Non-core responsibilities.
  - Availability → Flow & Focus (and Communication availability only if that fits).
- Do not invent skills, customers, budget, appreciation, testing, empathy, etc.
- ZERO boilerplate. Never write placeholders like
  "No observation this year.", "N/A — limited visibility", or
  "Limited direct observation; this can be evaluated further…".

AFTER MAX SPREADING — REMAINING GAPS:
- If a row's bullets have NO related fact: Good = "" and Can Improve = "".
- Blank is better than boilerplate or invented content.

TOUGH / SENSITIVE FEEDBACK:
- Keep the underlying fact; drop hearsay framing like "teammate told…".
- Rewrite as a professional development point that still matches the row's bullets.
- Do not invent extra negativity. Do not soften away the real issue either.

Generate exactly one feedback item for EVERY sub-area below.

REQUIRED STRUCTURE:

LEARNING
1. Learnings
2. Ability to learn
3. Efforts put into learning
4. Effects on quality or time

ORGANIZATION
5. Transparency / Honesty
6. Openness
7. Flow & Focus
8. Developing Competency
9. Radical Candor
10. Non-core responsibilities
11. Org processes

PROJECT EXECUTION
12. Work output - time taken
13. Work output - quality
14. Work output - Budget

CUSTOMER
15. Responsiveness
16. Relations & Appreciation
17. Communication

LEADERSHIP
18. Pro activeness
19. Team Management
20. Social Skills
21. Motivation
22. Empathy

RULES FOR "PREVIOUS YEAR":
- Use ONLY the previous-year information provided above.
- Do not invent previous-year feedback.
- If no previous-year information exists for a sub-area, use an empty string.

RULES FOR "GOOD":
- Must match that row's definition bullets.
- Keep each Good cell to 1–3 short sentences.
- Do not exaggerate or invent.
- Maximize filled Good cells from available facts with row-specific wording.
- If no related fact for that definition: "".

RULES FOR "CAN IMPROVE":
- Must match that row's definition bullets.
- Keep each Can Improve cell to 1–3 short sentences.
- Do not turn lack of reviewer visibility into a weakness of the employee.
- If no related fact for that definition: "".
- Never invent facts or placeholders to fill empty cells.

Employee summary and overall feedback must be built from evidenced definition
bullets, not generic praise. Write them in the same reviewer voice.

For each item, provide:
- category
- sub_area
- good
- can_improve
- confidence
- evidence_used
"""

    result = structured_llm.invoke(prompt)
    return {"feedback_result": result.model_dump()}


def validate_feedback(state: FeedbackState):
    structured_llm = llm.with_structured_output(ValidationResult)
    reviewer_voice = state.get("reviewer_voice", {}) or {}

    prompt = f"""
You are a strict performance review quality auditor.

Your task is to identify and remove unsupported claims while PRESERVING the
reviewer's voice, and ensure every cell matches the OFFICIAL REVIEW DEFINITIONS.

OFFICIAL REVIEW DEFINITIONS:
{REVIEW_DEFINITIONS_TEXT}

AVAILABLE REVIEWER INFORMATION:
{state["combined_observations"]}

EVIDENCE ANALYSIS:
{state["evidence_analysis"]}

REVIEWER VOICE PROFILE:
{reviewer_voice}

GENERATED FEEDBACK:
{state["feedback_result"]}

Check for:

1. Invented achievements
2. Unsupported claims that do not match definition bullets for that row
3. Facts placed on the WRONG row (move to the correct definition row or remove)
4. Can Improve that does not match that row's bullets
5. Unfair negative assumptions
6. Contradictions
7. Repetition — same distinctive phrase across multiple rows
8. Generic meaningless feedback / off-rubric praise
9. ANY placeholder / boilerplate
10. Harsh / hearsay / slang dumps
11. Related definition rows left empty when an existing fact fits those bullets
12. Summary/overall that is generic instead of based on evidenced definition bullets

VOICE RULES WHILE CORRECTING:
- Keep the reviewer's voice but polish grammar.
- Rewrite extras with a different angle when the same phrase repeats.
- Rewrite harsh/hearsay wording professionally while keeping the fact.
- Maximize coverage WITHIN the rubric: fill related definition rows with
  fresh angles when facts honestly fit; prefer filled related cells over blanks.
- For true gaps with ZERO related definition fact: Good = "" and Can Improve = "".
- DELETE any boilerplate; never invent Customer / Budget / Empathy / etc.
- Keep cells to 1–3 short sentences; keep summary/overall concise.

If an item lacks evidence for its definition after max spreading:
- Strip unsupported / off-rubric claims and all boilerplate.
- Set good to "" and can_improve to "".

Correct the feedback so every Good / Can Improve / summary / overall line
matches the official definitions, with zero boilerplate and no invented facts.
"""

    result = structured_llm.invoke(prompt)
    return {"validation_result": result.model_dump()}


def finalize_feedback(state: FeedbackState):
    return {"final_feedback": state["validation_result"]["corrected_feedback"]}


# ---------------------------------------------------------------------------
# Sheet helpers
# ---------------------------------------------------------------------------


def build_sheet_table(final_feedback: dict) -> pd.DataFrame:
    items = final_feedback["feedback"]
    by_subarea = {item["sub_area"].strip().lower(): item for item in items}

    rows = [
        {
            "Area": "Overall",
            "Sub Areas": "",
            "Previous Year": "",
            "Good": final_feedback.get("employee_summary", ""),
            "Can Improve": final_feedback.get("overall_feedback", ""),
        }
    ]

    for category, subareas in REVIEW_STRUCTURE[1:]:
        for sub_area in subareas:
            item = by_subarea.get(sub_area.strip().lower(), {})
            rows.append(
                {
                    "Area": category,
                    "Sub Areas": sub_area,
                    "Previous Year": "",
                    "Good": item.get("good", ""),
                    "Can Improve": item.get("can_improve", ""),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["Area", "Sub Areas", "Previous Year", "Good", "Can Improve"],
    )


def map_previous_year_feedback(
    sheet_df: pd.DataFrame, previous_year_text: str = ""
) -> pd.DataFrame:
    if not (previous_year_text or "").strip():
        return sheet_df

    structured_llm = llm.with_structured_output(PreviousYearMapping)

    prompt = f"""
Map the previous-year review into the current review-sheet sub-areas.

SUB-AREAS:
{SHEET_SUBAREAS}

PREVIOUS-YEAR REVIEW:
{previous_year_text}

Rules:
1. Use only information present in the previous-year review.
2. Do not invent or infer achievements.
3. Return exactly one row for each of these sub-areas.
4. Use an empty string when no matching information exists.
"""

    result = structured_llm.invoke(prompt)

    mapping = {
        row["sub_area"].strip().lower(): row.get("previous_year", "")
        for row in result.rows
    }

    output = sheet_df.copy()
    for i in range(len(output)):
        sub_area = str(output.at[i, "Sub Areas"]).strip().lower()
        if sub_area:
            output.at[i, "Previous Year"] = mapping.get(sub_area, "")

    return output


def build_final_sheet(
    final_feedback: dict, previous_year_text: str = ""
) -> pd.DataFrame:
    sheet_df = build_sheet_table(final_feedback)
    return map_previous_year_feedback(sheet_df, previous_year_text)


# ---------------------------------------------------------------------------
# Compile graph once (module-level) for stable MemorySaver across Streamlit reruns
# ---------------------------------------------------------------------------

_builder = StateGraph(FeedbackState)
_builder.add_node("analyze_evidence", analyze_evidence)
_builder.add_node("generate_missing_questions", generate_missing_questions)
_builder.add_node("ask_reviewer", ask_reviewer)
_builder.add_node("merge_reviewer_information", merge_reviewer_information)
_builder.add_node("analyze_reviewer_voice", analyze_reviewer_voice)
_builder.add_node("generate_feedback", generate_sheet_feedback)
_builder.add_node("validate_feedback", validate_feedback)
_builder.add_node("finalize_feedback", finalize_feedback)

_builder.add_edge(START, "analyze_evidence")
_builder.add_edge("analyze_evidence", "generate_missing_questions")
_builder.add_edge("generate_missing_questions", "ask_reviewer")
_builder.add_edge("ask_reviewer", "merge_reviewer_information")
_builder.add_edge("merge_reviewer_information", "analyze_reviewer_voice")
_builder.add_edge("analyze_reviewer_voice", "generate_feedback")
_builder.add_edge("generate_feedback", "validate_feedback")
_builder.add_edge("validate_feedback", "finalize_feedback")
_builder.add_edge("finalize_feedback", END)

_memory = MemorySaver()
sheet_feedback_graph = _builder.compile(checkpointer=_memory)


NODE_STATUS_LABELS = {
    "analyze_evidence": "Analyzing evidence…",
    "generate_missing_questions": "Generating follow-up questions…",
    "ask_reviewer": "Preparing questions for you…",
    "merge_reviewer_information": "Merging your answers…",
    "analyze_reviewer_voice": "Matching your writing style…",
    "generate_feedback": "Writing review in your voice (Pro)…",
    "validate_feedback": "Validating feedback…",
    "finalize_feedback": "Finalizing…",
}


def stream_review(input_data, thread_id: str, on_node=None) -> dict:
    """
    Stream graph updates until interrupt or completion.

    on_node(node_name, label) is called for each completed node.
    Returns a state dict; includes "__interrupt__" when paused for HITL.
    """
    config = {"configurable": {"thread_id": thread_id}}
    for chunk in sheet_feedback_graph.stream(
        input_data,
        config=config,
        stream_mode="updates",
    ):
        for node_name in chunk:
            if node_name == "__interrupt__":
                continue
            if on_node:
                on_node(
                    node_name,
                    NODE_STATUS_LABELS.get(node_name, f"Running {node_name}…"),
                )

    snapshot = sheet_feedback_graph.get_state(config)
    result = dict(snapshot.values)

    interrupts = list(getattr(snapshot, "interrupts", ()) or ())
    if not interrupts:
        for task in snapshot.tasks:
            if getattr(task, "interrupts", None):
                interrupts.extend(task.interrupts)
    if interrupts:
        result["__interrupt__"] = interrupts

    return result


def start_review(state: dict, thread_id: str, on_node=None) -> dict:
    """Run until interrupt (questions) or completion."""
    return stream_review(state, thread_id, on_node=on_node)


def resume_review(answers: str, thread_id: str, on_node=None) -> dict:
    """Resume after human answers the interrupt questions."""
    return stream_review(Command(resume=answers), thread_id, on_node=on_node)


def get_interrupt_questions(result: dict) -> list[str] | None:
    """Return questions from an interrupt payload, or None if not interrupted."""
    if "__interrupt__" not in result:
        return None
    payload = result["__interrupt__"][0].value
    return list(payload.get("questions", []))
