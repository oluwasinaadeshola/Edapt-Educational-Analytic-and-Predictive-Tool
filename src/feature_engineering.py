"""
Step 1 — Data Transformation & Feature Engineering.

Transforms transactional assessment rows into one row per (student, trimester),
with engineered features for Academic Momentum, Behavioral Red Flags, and
Structural Pressures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    EARLY_WARNING_WEIGHT_MAX,
    EARLY_WARNING_WEIGHT_MIN,
    FIRST_ASSESSMENT_ID,
    PASS_MARK_THRESHOLD,
    SUBJECT_FAMILIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_family(code: str) -> str:
    """Map SUBJECTCODE prefix to a subject family (MBA, ICT, Accounting, …)."""
    prefix = "".join(c for c in str(code) if c.isalpha())[:3].upper()
    for key, label in SUBJECT_FAMILIES.items():
        if prefix.startswith(key):
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Unit-level aggregation (student × subject × trimester × attempt)
# ---------------------------------------------------------------------------

def build_unit_records(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse assessment rows to one record per unit enrolment.
    Each record carries the final weighted mark and early-warning flag.
    """
    raw = raw.copy()
    raw["SUBJECT_FAMILY"] = raw["SUBJECTCODE"].map(_subject_family)

    # Early Warning: first assessment (ID=1) with weighting 10–20 %
    early_mask = (
        (raw["STUDYPACKAGEASSESSMENTID"] == FIRST_ASSESSMENT_ID)
        & (raw["WEIGHTING"] >= EARLY_WARNING_WEIGHT_MIN)
        & (raw["WEIGHTING"] <= EARLY_WARNING_WEIGHT_MAX)
    )
    early = (
        raw.loc[early_mask]
        .groupby(["STUDENTID_MASKED", "SUBJECTCODE", "STUDYPERIOD", "ATTEMPTNUMBER"])
        .agg(EARLY_WARNING_MARK=("MARKPERCENT", "first"))
        .reset_index()
    )

    unit_keys = [
        "STUDENTID_MASKED",
        "SUBJECTCODE",
        "STUDYPERIOD",
        "ATTEMPTNUMBER",
        "CLASSGROUP",
        "GENDERCODE",
        "AGEGROUP",
        "COUNTRY_MASKED",
        "SUBJECT_FAMILY",
    ]

    # Vectorised weighted final mark (faster than row-wise apply on 300k+ rows)
    raw["_WEIGHTED_MARK"] = raw["MARKPERCENT"] * raw["WEIGHTING"]
    units = (
        raw.groupby(unit_keys, as_index=False)
        .agg(
            _WEIGHTED_SUM=("_WEIGHTED_MARK", "sum"),
            _WEIGHT_TOTAL=("WEIGHTING", "sum"),
        )
    )
    units["UNIT_FINAL_MARK"] = units["_WEIGHTED_SUM"] / units["_WEIGHT_TOTAL"].replace(0, np.nan)
    units = units.drop(columns=["_WEIGHTED_SUM", "_WEIGHT_TOTAL"])
    units["UNIT_PASSED"] = (units["UNIT_FINAL_MARK"] >= PASS_MARK_THRESHOLD).astype(int)
    units = units.merge(
        early,
        on=["STUDENTID_MASKED", "SUBJECTCODE", "STUDYPERIOD", "ATTEMPTNUMBER"],
        how="left",
    )
    return units


# ---------------------------------------------------------------------------
# Subject difficulty (global rating used by Linear Regression)
# ---------------------------------------------------------------------------

def compute_subject_difficulty(units: pd.DataFrame) -> pd.DataFrame:
    """Lower mean mark → higher difficulty rating (inverted percentile)."""
    diff = (
        units.groupby("SUBJECTCODE")["UNIT_FINAL_MARK"]
        .mean()
        .reset_index(name="SUBJECT_MEAN_MARK")
    )
    diff["SUBJECT_DIFFICULTY"] = 100 - diff["SUBJECT_MEAN_MARK"]
    return diff


# ---------------------------------------------------------------------------
# Trimester-level feature table (one row per student per trimester)
# ---------------------------------------------------------------------------

def build_trimester_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature-engineering pipeline.
    Returns a DataFrame indexed logically by (STUDENTID_MASKED, STUDYPERIOD).
    """
    units = build_unit_records(raw)
    subject_diff = compute_subject_difficulty(units)

    trimester_keys = ["STUDENTID_MASKED", "STUDYPERIOD"]

    # --- Per-trimester aggregates ---
    tri = (
        units.groupby(trimester_keys, as_index=False)
        .agg(
            TRIMESTER_AVG_MARK=("UNIT_FINAL_MARK", "mean"),
            TRIMESTER_MIN_MARK=("UNIT_FINAL_MARK", "min"),
            TOTAL_UNITS=("SUBJECTCODE", "nunique"),
            TOTAL_FAILED_UNITS=("UNIT_PASSED", lambda s: int((s == 0).sum())),
            MAX_ATTEMPT=("ATTEMPTNUMBER", "max"),
            AVG_ATTEMPT=("ATTEMPTNUMBER", "mean"),
            EARLY_WARNING_AVG=("EARLY_WARNING_MARK", "mean"),
            EARLY_WARNING_FAILS=(
                "EARLY_WARNING_MARK",
                lambda s: int((s < PASS_MARK_THRESHOLD).sum()),
            ),
            GENDERCODE=("GENDERCODE", "first"),
            AGEGROUP=("AGEGROUP", "first"),
            COUNTRY_MASKED=("COUNTRY_MASKED", "first"),
        )
    )

    # Study Load Intensity
    tri["STUDY_LOAD_INTENSITY"] = tri["TOTAL_UNITS"]

    # Subject Consistency — mean mark per family within trimester
    family_pivot = (
        units.groupby(trimester_keys + ["SUBJECT_FAMILY"])["UNIT_FINAL_MARK"]
        .mean()
        .unstack(fill_value=np.nan)
    )
    family_pivot.columns = [f"CONSISTENCY_{c.upper().replace(' ', '_')}" for c in family_pivot.columns]
    tri = tri.merge(family_pivot.reset_index(), on=trimester_keys, how="left")

    # ICT vs MBA spread (Subject Consistency metric)
    ict_col = next((c for c in tri.columns if "ICT" in c), None)
    mba_col = next((c for c in tri.columns if "MBA" in c), None)
    if ict_col and mba_col:
        tri["SUBJECT_CONSISTENCY_ICT_MBA_GAP"] = tri[ict_col] - tri[mba_col]
    else:
        tri["SUBJECT_CONSISTENCY_ICT_MBA_GAP"] = np.nan

    # Peer Benchmarking — percentile rank vs CLASSGROUP average
    class_avg = (
        units.groupby(["STUDYPERIOD", "CLASSGROUP"])["UNIT_FINAL_MARK"]
        .mean()
        .reset_index(name="CLASSGROUP_AVG")
    )
    student_class = (
        units.groupby(trimester_keys + ["CLASSGROUP"])["UNIT_FINAL_MARK"]
        .mean()
        .reset_index(name="STUDENT_CLASS_AVG")
    )
    peer = student_class.merge(class_avg, on=["STUDYPERIOD", "CLASSGROUP"])
    peer["PEER_DELTA"] = peer["STUDENT_CLASS_AVG"] - peer["CLASSGROUP_AVG"]
    peer["PEER_PERCENTILE"] = peer.groupby(["STUDYPERIOD", "CLASSGROUP"])[
        "STUDENT_CLASS_AVG"
    ].rank(pct=True)
    tri = tri.merge(
        peer.groupby(trimester_keys).agg(
            PEER_DELTA=("PEER_DELTA", "mean"),
            PEER_PERCENTILE=("PEER_PERCENTILE", "mean"),
        ).reset_index(),
        on=trimester_keys,
        how="left",
    )

    # Sort trimesters chronologically per student
    tri = tri.sort_values(trimester_keys).reset_index(drop=True)

    # --- Academic Momentum: GPA Trajectory (slope over preceding trimesters) ---
    tri["GPA_TRAJECTORY_SLOPE"] = np.nan
    tri["PRIOR_WEIGHTED_AVG"] = np.nan

    for student_id, grp in tri.groupby("STUDENTID_MASKED"):
        idx = grp.index.tolist()
        marks = grp["TRIMESTER_AVG_MARK"].values
        for i, row_idx in enumerate(idx):
            if i == 0:
                continue
            prior_marks = marks[:i]
            prior_periods = np.arange(len(prior_marks), dtype=float)
            tri.loc[row_idx, "PRIOR_WEIGHTED_AVG"] = np.average(prior_marks)
            if len(prior_marks) >= 2:
                slope, _, _, _, _ = stats.linregress(prior_periods, prior_marks)
                tri.loc[row_idx, "GPA_TRAJECTORY_SLOPE"] = slope

    # Study Load Spike — current load vs student's historical mean
    tri["STUDY_LOAD_SPIKE"] = np.nan
    for student_id, grp in tri.groupby("STUDENTID_MASKED"):
        loads = grp["STUDY_LOAD_INTENSITY"].values
        idx = grp.index.tolist()
        for i, row_idx in enumerate(idx):
            if i == 0:
                continue
            hist_mean = loads[:i].mean()
            tri.loc[row_idx, "STUDY_LOAD_SPIKE"] = loads[i] - hist_mean

    # --- Target: next trimester performance (for ML) ---
    tri["NEXT_TRIMESTER_AVG_MARK"] = tri.groupby("STUDENTID_MASKED")[
        "TRIMESTER_AVG_MARK"
    ].shift(-1)
    tri["NEXT_TRIMESTER_FAILED"] = (
        tri.groupby("STUDENTID_MASKED")["TOTAL_FAILED_UNITS"].shift(-1)
    )
    tri["NEXT_PASS_FAIL"] = (
        tri["NEXT_TRIMESTER_AVG_MARK"] >= PASS_MARK_THRESHOLD
    ).astype(float)

    # Subject difficulty for the hardest unit taken that trimester
    units_with_diff = units.merge(subject_diff, on="SUBJECTCODE")
    hardest = (
        units_with_diff.groupby(trimester_keys)["SUBJECT_DIFFICULTY"]
        .max()
        .reset_index(name="MAX_SUBJECT_DIFFICULTY")
    )
    tri = tri.merge(hardest, on=trimester_keys, how="left")

    tri["HAS_HISTORY"] = tri["PRIOR_WEIGHTED_AVG"].notna()
    return tri


def get_feature_summary(features: pd.DataFrame) -> dict:
    """Quick stats for the UI overview panel."""
    return {
        "students": features["STUDENTID_MASKED"].nunique(),
        "trimester_records": len(features),
        "trimesters": features["STUDYPERIOD"].nunique(),
        "ml_ready_rows": int(features["HAS_HISTORY"].sum()),
        "avg_failed_units": round(features["TOTAL_FAILED_UNITS"].mean(), 2),
    }
