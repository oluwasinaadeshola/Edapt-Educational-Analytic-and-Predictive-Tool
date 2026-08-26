"""
Step 4 — Generative AI Intervention Engine.

Drafts personalised, empathetic intervention emails for at-risk students
using OpenAI (preferred) or Hugging Face Inference API (free alternative).
Falls back to a structured template when no API key is configured.
"""

from __future__ import annotations

import os
from textwrap import dedent

import pandas as pd

from config import HUGGINGFACE_MODEL, OPENAI_MODEL


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_risk_context(row: pd.Series, concurrent_subjects: int | None = None) -> str:
    """Convert engineered features into a plain-language risk brief for the LLM."""
    flags = []

    if concurrent_subjects is not None and concurrent_subjects > 0:
        flags.append(
            f"Currently enrolled in {concurrent_subjects} concurrent subject(s) this trimester"
        )

    slope = row.get("GPA_TRAJECTORY_SLOPE")
    if pd.notna(slope) and slope < -2:
        flags.append(f"GPA trajectory is declining (slope: {slope:.2f} points/trimester)")

    early = row.get("EARLY_WARNING_AVG")
    if pd.notna(early) and early < 50:
        flags.append(f"Early assessment average is below pass threshold ({early:.1f}%)")

    early_fails = row.get("EARLY_WARNING_FAILS", 0)
    if early_fails and early_fails > 0:
        flags.append(f"Failed {int(early_fails)} early-weighted assessment(s) (10–20% tasks)")

    failed = row.get("TOTAL_FAILED_UNITS", 0)
    if failed and failed > 0:
        flags.append(f"{int(failed)} unit(s) failed this trimester")

    load_spike = row.get("STUDY_LOAD_SPIKE")
    if pd.notna(load_spike) and load_spike > 1:
        flags.append(f"Study load spike detected (+{load_spike:.0f} subjects vs prior average)")

    peer = row.get("PEER_PERCENTILE")
    if pd.notna(peer) and peer < 0.25:
        flags.append(f"Ranking in bottom quartile of class group ({peer:.0%} percentile)")

    max_attempt = row.get("MAX_ATTEMPT", 1)
    if max_attempt >= 2:
        flags.append(f"Currently on attempt #{int(max_attempt)} for at least one unit")

    risk_prob = row.get("FAILURE_RISK_PROB")
    if pd.notna(risk_prob):
        flags.append(f"Model-estimated failure risk: {risk_prob:.0%}")

    if not flags:
        flags.append("General academic concern flagged by predictive model")

    return "\n".join(f"- {f}" for f in flags)


def _workload_guidance(concurrent_subjects: int) -> str:
    """Actionable workload-management tips scaled to subject count."""
    if concurrent_subjects <= 1:
        return (
            "With a focused single-subject load, prioritise depth over breadth — "
            "use weekly checkpoints to stay ahead of assessment deadlines."
        )
    return (
        f"With {concurrent_subjects} subjects running in parallel, realistic pacing matters. "
        f"I recommend a fixed weekly timetable: allocate dedicated blocks per subject, "
        f"prioritise the two highest-weight assessments due next, and use the student "
        f"support desk to discuss load-balancing options if deadlines cluster."
    )


def _system_prompt(concurrent_subjects: int | None = None) -> str:
    workload_note = ""
    if concurrent_subjects is not None and concurrent_subjects > 0:
        workload_note = (
            f"\n        IMPORTANT: The student is managing {concurrent_subjects} concurrent "
            f"subjects this trimester. You MUST explicitly acknowledge this heavy workload "
            f"in the email and offer realistic, practical support for managing that pressure "
            f"(e.g. time-blocking, prioritisation, extension options, or study skills support)."
        )

    return dedent(
        f"""
        You are an experienced academic advisor assistant. Write a warm, empathetic
        email draft that a teacher can send to a struggling student. The tone should
        be supportive — never punitive. Include:
        1. A brief acknowledgement of their situation (without revealing raw scores harshly)
        2. Two specific, actionable study strategies tied to the risk factors
        3. An invitation to book a 1:1 support session
        4. A closing that reinforces belief in their ability to succeed
        Keep the email under 280 words. Do not include a subject line.{workload_note}
        """
    ).strip()


def _user_prompt(
    student_id: str,
    row: pd.Series,
    concurrent_subjects: int | None = None,
) -> str:
    risk_brief = _build_risk_context(row, concurrent_subjects=concurrent_subjects)
    avg_mark = _safe_float(row.get("TRIMESTER_AVG_MARK"), default=float("nan"))
    avg_text = f"{avg_mark:.1f}%" if pd.notna(avg_mark) else "N/A"

    workload_line = ""
    if concurrent_subjects is not None:
        workload_line = f"Concurrent subjects this trimester: {concurrent_subjects}\n"

    return dedent(
        f"""
        Student reference: {student_id}
        Trimester: {row.get('STUDYPERIOD', 'N/A')}
        Current average mark: {avg_text}
        Age group: {row.get('AGEGROUP', 'N/A')}
        {workload_line}
        Identified risk factors:
        {risk_brief}

        Draft the intervention email now.
        """
    ).strip()


def _generate_openai(user_prompt: str, concurrent_subjects: int | None = None) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt(concurrent_subjects)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=450,
    )
    return response.choices[0].message.content.strip()


def _generate_huggingface(user_prompt: str, concurrent_subjects: int | None = None) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ.get("HUGGINGFACE_API_TOKEN"))
    prompt = f"{_system_prompt(concurrent_subjects)}\n\n{user_prompt}\n\nEmail draft:"
    result = client.text_generation(
        prompt,
        model=HUGGINGFACE_MODEL,
        max_new_tokens=450,
        temperature=0.7,
    )
    return result.strip()


def _generate_template(
    student_id: str,
    row: pd.Series,
    concurrent_subjects: int | None = None,
) -> str:
    """Offline fallback — no API key required."""
    risk_brief = _build_risk_context(row, concurrent_subjects=concurrent_subjects)
    avg = _safe_float(row.get("TRIMESTER_AVG_MARK"))
    subjects = concurrent_subjects or int(row.get("STUDY_LOAD_INTENSITY", 0) or 0)
    workload_para = ""
    if subjects > 0:
        workload_para = dedent(
            f"""
        I also want to acknowledge that you are currently balancing **{subjects} subjects**
        this trimester. That is a significant workload, and it is completely understandable
        if you are feeling stretched. {_workload_guidance(subjects)}
        """
        ).strip()

    return dedent(
        f"""
        Dear Student ({student_id}),

        I hope this message finds you well. I wanted to reach out personally because
        I have been reviewing progress this trimester and I am here to support you.

        {workload_para}

        I noticed a few patterns that often appear when students are juggling competing
        priorities — and every one of them is something we can work through together:

        {risk_brief}

        Here are two strategies that have helped students in similar situations:
        1. **Front-load your week** — block two 45-minute study sessions before mid-week
           for your highest-weight assessments, especially early tasks worth 10–20%.
        2. **Use office hours early** — bring one specific question from your weakest unit;
           a 15-minute conversation often saves hours of confusion later.

        Please reply to this email or book a 1:1 session through the student portal.
        Your current average of {avg:.1f}% shows real capability — with targeted support,
        I genuinely believe you can turn this trajectory around.

        Warm regards,
        [Your Name]
        Academic Advisor
        """
    ).strip()


def generate_intervention_email(
    student_id: str,
    row: pd.Series,
    provider: str = "auto",
    concurrent_subjects: int | None = None,
) -> tuple[str, str]:
    """
    Generate an intervention email draft.

    Parameters
    ----------
    provider : "auto" | "openai" | "huggingface" | "template"
    concurrent_subjects : number of unique subjects in the student's current trimester

    Returns
    -------
    (email_text, provider_used)
    """
    if concurrent_subjects is None:
        load_val = row.get("STUDY_LOAD_INTENSITY")
        if pd.notna(load_val):
            try:
                concurrent_subjects = int(load_val)
            except (TypeError, ValueError):
                concurrent_subjects = None

    user_prompt = _user_prompt(student_id, row, concurrent_subjects=concurrent_subjects)

    if provider == "template":
        return (
            _generate_template(student_id, row, concurrent_subjects=concurrent_subjects),
            "Template (offline)",
        )

    if provider in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        try:
            return (
                _generate_openai(user_prompt, concurrent_subjects=concurrent_subjects),
                f"OpenAI ({OPENAI_MODEL})",
            )
        except Exception:
            if provider == "openai":
                raise

    if provider in ("auto", "huggingface") and os.environ.get("HUGGINGFACE_API_TOKEN"):
        try:
            return (
                _generate_huggingface(user_prompt, concurrent_subjects=concurrent_subjects),
                f"Hugging Face ({HUGGINGFACE_MODEL})",
            )
        except Exception:
            if provider == "huggingface":
                raise

    return (
        _generate_template(student_id, row, concurrent_subjects=concurrent_subjects),
        "Template (offline — set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN for AI)",
    )
