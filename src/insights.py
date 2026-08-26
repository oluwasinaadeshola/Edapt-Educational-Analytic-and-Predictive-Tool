"""
Step 3 — Macro Insights for Academic Teams.

Administrative analytics: sinking subjects and re-attempt correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_sinking_subjects(
    raw: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Identify SUBJECTCODEs whose mean mark is significantly below the cohort.

    Uses a one-sample t-test against the global mean; flags subjects where
    p-value < alpha and mean is lower than overall average.
    """
    subject_stats = (
        raw.groupby("SUBJECTCODE")["MARKPERCENT"]
        .agg(Mean_Mark="mean", Std_Dev="std", Assessment_Count="count")
        .reset_index()
    )

    global_mean = raw["MARKPERCENT"].mean()
    subject_stats["Global_Mean"] = round(global_mean, 2)
    subject_stats["Gap_vs_Global"] = round(subject_stats["Mean_Mark"] - global_mean, 2)

    p_values = []
    for code, grp in raw.groupby("SUBJECTCODE"):
        if len(grp) < 30:
            p_values.append(1.0)
            continue
        _, p = stats.ttest_1samp(grp["MARKPERCENT"], global_mean, alternative="less")
        p_values.append(p)

    subject_stats["P_Value"] = np.round(p_values, 4)
    subject_stats["Is_Sinking"] = (
        (subject_stats["P_Value"] < alpha)
        & (subject_stats["Mean_Mark"] < global_mean)
    )
    subject_stats = subject_stats.sort_values("Mean_Mark").reset_index(drop=True)
    subject_stats["Mean_Mark"] = subject_stats["Mean_Mark"].round(2)
    subject_stats["Std_Dev"] = subject_stats["Std_Dev"].round(2)
    return subject_stats


def compute_reattempt_correlation(units: pd.DataFrame) -> dict:
    """
    Compare failure rates for first-attempt vs re-attempt (ATTEMPTNUMBER >= 2) students.
    """
    first_attempt = units[units["ATTEMPTNUMBER"] == 1]
    re_attempt = units[units["ATTEMPTNUMBER"] >= 2]

    first_fail_rate = (
        1 - first_attempt["UNIT_PASSED"].mean() if len(first_attempt) else 0
    )
    re_fail_rate = (
        1 - re_attempt["UNIT_PASSED"].mean() if len(re_attempt) else 0
    )

    # Student-level view: students who ever re-attempted
    reattempt_students = units.loc[units["ATTEMPTNUMBER"] >= 2, "STUDENTID_MASKED"].unique()
    student_fail = (
        units.groupby("STUDENTID_MASKED")["UNIT_PASSED"]
        .apply(lambda s: 1 - s.mean())
        .reset_index(name="FAIL_RATE")
    )
    reattempt_fail = student_fail[
        student_fail["STUDENTID_MASKED"].isin(reattempt_students)
    ]["FAIL_RATE"].mean()
    no_reattempt_fail = student_fail[
        ~student_fail["STUDENTID_MASKED"].isin(reattempt_students)
    ]["FAIL_RATE"].mean()

    return {
        "first_attempt_fail_pct": round(first_fail_rate * 100, 2),
        "reattempt_fail_pct": round(re_fail_rate * 100, 2),
        "reattempt_student_fail_pct": round(reattempt_fail * 100, 2),
        "no_reattempt_student_fail_pct": round(no_reattempt_fail * 100, 2),
        "reattempt_unit_count": len(re_attempt),
        "first_attempt_unit_count": len(first_attempt),
        "reattempt_student_count": len(reattempt_students),
    }


def trimester_enrolment_trends(raw: pd.DataFrame) -> pd.DataFrame:
    """Enrolment volume per trimester for the admin dashboard."""
    return (
        raw.groupby("STUDYPERIOD")["STUDENTID_MASKED"]
        .nunique()
        .reset_index(name="Unique_Students")
        .sort_values("STUDYPERIOD")
    )


def gender_performance_gap(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean mark by gender for equity monitoring."""
    return (
        raw.groupby("GENDERCODE")["MARKPERCENT"]
        .agg(Mean_Mark="mean", Count="count")
        .reset_index()
        .round(2)
    )
