"""
Student Deep-Dive Profile helpers.

Builds per-student progress trajectories, workload metrics, and summary cards
for the teacher-facing profile view.
"""

from __future__ import annotations

import pandas as pd

from config import PASS_MARK_THRESHOLD, WORKLOAD_OVERLOAD_THRESHOLD


def list_students(raw: pd.DataFrame) -> list[str]:
    """Return sorted unique student IDs, safely handling missing values."""
    if raw.empty or "STUDENTID_MASKED" not in raw.columns:
        return []
    ids = raw["STUDENTID_MASKED"].dropna().astype(str).unique().tolist()
    return sorted(ids)


def compute_cohort_avg_study_load(raw: pd.DataFrame) -> float:
    """
    Dataset-wide average number of unique subjects per student per trimester.
    Used as the benchmark for workload spike detection.
    """
    if raw.empty:
        return float(WORKLOAD_OVERLOAD_THRESHOLD)

    df = raw.copy()
    df["STUDYPERIOD"] = pd.to_numeric(df["STUDYPERIOD"], errors="coerce")
    loads = (
        df.dropna(subset=["STUDENTID_MASKED", "STUDYPERIOD", "SUBJECTCODE"])
        .groupby(["STUDENTID_MASKED", "STUDYPERIOD"])["SUBJECTCODE"]
        .nunique()
    )
    if loads.empty:
        return float(WORKLOAD_OVERLOAD_THRESHOLD)
    return round(float(loads.mean()), 2)


def build_mark_trajectory(raw: pd.DataFrame, student_id: str) -> pd.DataFrame:
    """
    Average MARKPERCENT per chronological STUDYPERIOD for one student.

    Returns columns: STUDYPERIOD, MARKPERCENT, PASS_STATUS
    """
    empty = pd.DataFrame(columns=["STUDYPERIOD", "MARKPERCENT", "PASS_STATUS"])

    if not student_id or raw.empty:
        return empty

    student = raw[raw["STUDENTID_MASKED"].astype(str) == str(student_id)].copy()
    if student.empty:
        return empty

    student["STUDYPERIOD"] = pd.to_numeric(student["STUDYPERIOD"], errors="coerce")
    student["MARKPERCENT"] = pd.to_numeric(student["MARKPERCENT"], errors="coerce")
    student = student.dropna(subset=["STUDYPERIOD", "MARKPERCENT"])
    if student.empty:
        return empty

    trajectory = (
        student.groupby("STUDYPERIOD", as_index=False)["MARKPERCENT"]
        .mean()
        .rename(columns={"MARKPERCENT": "MARKPERCENT"})
        .sort_values("STUDYPERIOD")
    )
    trajectory["PASS_STATUS"] = trajectory["MARKPERCENT"].apply(
        lambda m: "Passing" if m >= PASS_MARK_THRESHOLD else "Failing"
    )
    return trajectory.reset_index(drop=True)


def compute_current_study_load(
    raw: pd.DataFrame,
    student_id: str,
    cohort_avg: float | None = None,
    threshold: float = WORKLOAD_OVERLOAD_THRESHOLD,
) -> dict:
    """
    Count unique SUBJECTCODEs in the student's most recent STUDYPERIOD.

    Flags overload when count exceeds `threshold` OR the cohort average.
    """
    default = {
        "student_id": student_id,
        "latest_period": None,
        "concurrent_subjects": 0,
        "cohort_avg_load": cohort_avg or float(threshold),
        "overload_threshold": threshold,
        "is_overload": False,
        "overload_reason": None,
        "subject_codes": [],
    }

    if not student_id or raw.empty:
        return default

    cohort_avg = cohort_avg if cohort_avg is not None else compute_cohort_avg_study_load(raw)
    default["cohort_avg_load"] = cohort_avg

    student = raw[raw["STUDENTID_MASKED"].astype(str) == str(student_id)].copy()
    if student.empty:
        return default

    student["STUDYPERIOD"] = pd.to_numeric(student["STUDYPERIOD"], errors="coerce")
    student = student.dropna(subset=["STUDYPERIOD"])
    if student.empty:
        return default

    latest_period = student["STUDYPERIOD"].max()
    latest = student[student["STUDYPERIOD"] == latest_period]
    concurrent = int(latest["SUBJECTCODE"].nunique())
    subject_codes = sorted(latest["SUBJECTCODE"].dropna().astype(str).unique().tolist())

    exceeds_fixed = concurrent > threshold
    exceeds_cohort = concurrent > cohort_avg
    is_overload = exceeds_fixed or exceeds_cohort

    reasons = []
    if exceeds_fixed:
        reasons.append(f"exceeds fixed threshold ({threshold:.0f} subjects)")
    if exceeds_cohort:
        reasons.append(f"exceeds cohort average ({cohort_avg:.1f} subjects)")

    return {
        "student_id": student_id,
        "latest_period": float(latest_period),
        "concurrent_subjects": concurrent,
        "cohort_avg_load": cohort_avg,
        "overload_threshold": threshold,
        "is_overload": is_overload,
        "overload_reason": "; ".join(reasons) if reasons else None,
        "subject_codes": subject_codes,
    }


def build_student_summary(
    raw: pd.DataFrame,
    features: pd.DataFrame,
    student_id: str,
    cohort_avg: float | None = None,
) -> dict:
    """
    Combine trajectory, workload, and latest engineered features for one student.
    """
    trajectory = build_mark_trajectory(raw, student_id)
    workload = compute_current_study_load(raw, student_id, cohort_avg=cohort_avg)

    feature_row = None
    if not features.empty and student_id:
        student_features = features[
            features["STUDENTID_MASKED"].astype(str) == str(student_id)
        ].sort_values("STUDYPERIOD")
        if not student_features.empty:
            feature_row = student_features.iloc[-1]

    latest_mark = None
    trend_label = "Insufficient data"
    if not trajectory.empty:
        latest_mark = float(trajectory["MARKPERCENT"].iloc[-1])
        if len(trajectory) >= 2:
            delta = latest_mark - float(trajectory["MARKPERCENT"].iloc[-2])
            if delta >= 2:
                trend_label = "Improving"
            elif delta <= -2:
                trend_label = "Declining"
            else:
                trend_label = "Stable"

    return {
        "trajectory": trajectory,
        "workload": workload,
        "feature_row": feature_row,
        "latest_mark": latest_mark,
        "trend_label": trend_label,
        "trimesters_recorded": len(trajectory),
    }
