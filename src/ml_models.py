"""
Step 2 — Machine Learning Model Comparison.

Trains three distinct models using past trimesters to forecast subsequent
performance, then returns evaluation metrics for the UI comparison table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    name: str
    task: str
    metrics: dict[str, float]
    notes: str


def _prepare_ml_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with prior history and a known next-trimester outcome."""
    ml = features[
        features["HAS_HISTORY"]
        & features["NEXT_TRIMESTER_AVG_MARK"].notna()
        & features["NEXT_PASS_FAIL"].notna()
    ].copy()
    ml["NEXT_PASS_FAIL"] = ml["NEXT_PASS_FAIL"].astype(int)
    return ml


def train_and_compare_models(
    features: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[list[ModelResult], pd.DataFrame, dict]:
    """
    Train Logistic Regression, Linear Regression, and Random Forest.

    Returns
    -------
    results : list of ModelResult for the comparison table
    comparison_df : tidy DataFrame ready for st.dataframe()
    artifacts : fitted models + test predictions for downstream use
    """
    ml = _prepare_ml_frame(features)
    if len(ml) < 50:
        raise ValueError(
            f"Not enough ML-ready rows ({len(ml)}). "
            "Need students with at least two trimesters of history."
        )

    # ------------------------------------------------------------------ #
    # 1. Logistic Regression — binary Pass/Fail
    #    Features: Avg Mark, Total Failed Units, Age Group
    # ------------------------------------------------------------------ #
    log_features = ["TRIMESTER_AVG_MARK", "TOTAL_FAILED_UNITS", "AGEGROUP"]
    X_log = ml[log_features]
    y_log = ml["NEXT_PASS_FAIL"]

    X_log_train, X_log_test, y_log_train, y_log_test = train_test_split(
        X_log, y_log, test_size=test_size, random_state=random_state, stratify=y_log
    )

    log_pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["TRIMESTER_AVG_MARK", "TOTAL_FAILED_UNITS"]),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["AGEGROUP"]),
                    ]
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    log_pipe.fit(X_log_train, y_log_train)
    log_pred = log_pipe.predict(X_log_test)
    log_prob = log_pipe.predict_proba(X_log_test)[:, 1]

    log_metrics = {
        "Accuracy": round(accuracy_score(y_log_test, log_pred), 4),
        "F1 Score": round(f1_score(y_log_test, log_pred, zero_division=0), 4),
    }
    if len(np.unique(y_log_test)) > 1:
        log_metrics["ROC-AUC"] = round(roc_auc_score(y_log_test, log_prob), 4)

    # ------------------------------------------------------------------ #
    # 2. Linear Regression — continuous next-trimester mark
    #    Features: prior weighted average, subject difficulty rating
    # ------------------------------------------------------------------ #
    lin_features = ["PRIOR_WEIGHTED_AVG", "MAX_SUBJECT_DIFFICULTY"]
    X_lin = ml[lin_features].dropna()
    y_lin = ml.loc[X_lin.index, "NEXT_TRIMESTER_AVG_MARK"]

    X_lin_train, X_lin_test, y_lin_train, y_lin_test = train_test_split(
        X_lin, y_lin, test_size=test_size, random_state=random_state
    )

    lin_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    lin_pipe.fit(X_lin_train, y_lin_train)
    lin_pred = lin_pipe.predict(X_lin_test)

    lin_metrics = {
        "R² Score": round(r2_score(y_lin_test, lin_pred), 4),
        "MAE": round(mean_absolute_error(y_lin_test, lin_pred), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_lin_test, lin_pred))), 4),
    }

    # ------------------------------------------------------------------ #
    # 3. Random Forest — nonlinear risk from COUNTRY × GENDER × ATTEMPT
    # ------------------------------------------------------------------ #
    rf_features = ["COUNTRY_MASKED", "GENDERCODE", "MAX_ATTEMPT"]
    X_rf = ml[rf_features]
    y_rf = ml["NEXT_PASS_FAIL"]

    X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(
        X_rf, y_rf, test_size=test_size, random_state=random_state, stratify=y_rf
    )

    rf_pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(handle_unknown="ignore"), rf_features),
                    ]
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_rf_train, y_rf_train)
    rf_pred = rf_pipe.predict(X_rf_test)
    rf_prob = rf_pipe.predict_proba(X_rf_test)[:, 1]

    rf_metrics = {
        "Accuracy": round(accuracy_score(y_rf_test, rf_pred), 4),
        "F1 Score": round(f1_score(y_rf_test, rf_pred, zero_division=0), 4),
    }
    if len(np.unique(y_rf_test)) > 1:
        rf_metrics["ROC-AUC"] = round(roc_auc_score(y_rf_test, rf_prob), 4)

    # ------------------------------------------------------------------ #
    # Assemble comparison table
    # ------------------------------------------------------------------ #
    results = [
        ModelResult(
            name="Logistic Regression",
            task="Pass/Fail Classification",
            metrics=log_metrics,
            notes="Features: Avg Mark, Total Failed Units, Age Group",
        ),
        ModelResult(
            name="Linear Regression",
            task="Next Trimester Mark (continuous)",
            metrics=lin_metrics,
            notes="Features: Prior Weighted Avg, Subject Difficulty",
        ),
        ModelResult(
            name="Random Forest",
            task="Pass/Fail (nonlinear risk)",
            metrics=rf_metrics,
            notes="Features: Country × Gender × Attempt Number",
        ),
    ]

    rows = []
    for r in results:
        row = {"Model": r.name, "Task": r.task, "Notes": r.notes}
        row.update(r.metrics)
        rows.append(row)
    comparison_df = pd.DataFrame(rows)

    # Risk scores for all ML-ready rows (used by GenAI tab)
    ml = ml.copy()
    log_input = ml[log_features]
    pass_idx = int(np.where(log_pipe.named_steps["clf"].classes_ == 1)[0][0])
    ml["FAILURE_RISK_PROB"] = 1 - log_pipe.predict_proba(log_input)[:, pass_idx]

    artifacts = {
        "logistic": log_pipe,
        "linear": lin_pipe,
        "random_forest": rf_pipe,
        "risk_scores": ml,
        "train_size": len(ml),
    }

    return results, comparison_df, artifacts
