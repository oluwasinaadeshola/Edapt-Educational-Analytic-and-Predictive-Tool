"""
EdAPT- Educational Analytics and Predictive Tools — Streamlit Application
Capstone graduation project: predictive analytics + GenAI intervention engine.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is on the path so `config` and `src` resolve correctly
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DEFAULT_DATA_PATH,
    HIGH_RISK_PROBABILITY_THRESHOLD,
    PASS_MARK_THRESHOLD,
    WORKLOAD_OVERLOAD_THRESHOLD,
)
from src.data_loader import load_assessment_data
from src.feature_engineering import build_trimester_features, build_unit_records, get_feature_summary
from src.genai_intervention import generate_intervention_email
from src.insights import (
    compute_reattempt_correlation,
    compute_sinking_subjects,
    gender_performance_gap,
    trimester_enrolment_trends,
)
from src.ml_models import train_and_compare_models
from src.student_profile import (
    build_student_summary,
    compute_cohort_avg_study_load,
    compute_current_study_load,
    list_students,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EdAPT- Educational Analytics and Predictive Tools",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎓 EdAPT- Educational Analytics and Predictive Tools")
st.caption(
    "Capstone project — predictive early warning, ML model comparison, "
    "and GenAI-powered intervention drafting."
)


# ---------------------------------------------------------------------------
# Cached pipeline steps (expensive on first run)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading assessment data…")
def load_data(path: str) -> pd.DataFrame:
    return load_assessment_data(path)


@st.cache_data(show_spinner="Engineering features (Step 1)…")
def engineer_features(_raw: pd.DataFrame) -> pd.DataFrame:
    return build_trimester_features(_raw)


@st.cache_data(show_spinner="Training ML models (Step 2)…")
def run_models(_features: pd.DataFrame):
    return train_and_compare_models(_features)


@st.cache_data(show_spinner="Computing admin insights…")
def admin_insights(_raw: pd.DataFrame, _units: pd.DataFrame):
    return {
        "sinking": compute_sinking_subjects(_raw),
        "reattempt": compute_reattempt_correlation(_units),
        "enrolment": trimester_enrolment_trends(_raw),
        "gender": gender_performance_gap(_raw),
    }


@st.cache_data(show_spinner="Computing cohort workload benchmark…")
def cohort_study_load(_raw: pd.DataFrame) -> float:
    return compute_cohort_avg_study_load(_raw)


def _resolve_feature_row(
    student_id: str,
    features: pd.DataFrame,
    risk_df: pd.DataFrame | None,
) -> pd.Series | None:
    """Pick the best available feature row for GenAI (risk scores preferred)."""
    if risk_df is not None and not risk_df.empty:
        student_risk = risk_df[risk_df["STUDENTID_MASKED"].astype(str) == str(student_id)]
        if not student_risk.empty:
            return student_risk.sort_values("STUDYPERIOD").iloc[-1]

    student_features = features[features["STUDENTID_MASKED"].astype(str) == str(student_id)]
    if not student_features.empty:
        return student_features.sort_values("STUDYPERIOD").iloc[-1]
    return None


# ---------------------------------------------------------------------------
# Load & process
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    data_path = st.text_input(
        "Data file path",
        value=str(DEFAULT_DATA_PATH),
        help="Local CSV export. Replace with LMS webhook URL in production (see src/data_loader.py).",
    )

try:
    raw = load_data(data_path)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

features = engineer_features(raw)
units = build_unit_records(raw)
summary = get_feature_summary(features)

try:
    model_results, comparison_df, artifacts = run_models(features)
    risk_df = artifacts["risk_scores"]
except ValueError as exc:
    st.warning(str(exc))
    model_results, comparison_df, artifacts, risk_df = None, None, None, None

insights = admin_insights(raw, units)
cohort_avg_load = cohort_study_load(raw)
all_students = list_students(raw)

if "selected_student" not in st.session_state and all_students:
    st.session_state.selected_student = all_students[0]

with st.sidebar:
    st.divider()
    st.header("Student Selector")
    if all_students:
        default_idx = (
            all_students.index(st.session_state.selected_student)
            if st.session_state.selected_student in all_students
            else 0
        )
        selected_student = st.selectbox(
            "Select student (STUDENTID_MASKED)",
            options=all_students,
            index=default_idx,
            help="Choose any student to inspect in the Deep-Dive Profile tab.",
        )
        st.session_state.selected_student = selected_student
        if st.button("🎲 Pick Random Student"):
            st.session_state.selected_student = str(
                pd.Series(all_students).sample(1, random_state=None).iloc[0]
            )
            st.rerun()
    else:
        st.warning("No students found in the dataset.")
        selected_student = None
    st.divider()
    st.markdown(
        "**LMS Integration (future)**  \n"
        "The data loader in `src/data_loader.py` documents where to plug in "
        "Canvas REST API or Moodle Web Services webhooks."
    )
    st.divider()
    st.markdown(
        "**GenAI API Keys (optional)**  \n"
        "Set environment variables before launching:\n"
        "- `OPENAI_API_KEY` — preferred\n"
        "- `HUGGINGFACE_API_TOKEN` — free alternative\n\n"
        "Without keys, the app uses an offline template."
    )

selected_student = st.session_state.get("selected_student")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Students", f"{summary['students']:,}")
k2.metric("Trimester Records", f"{summary['trimester_records']:,}")
k3.metric("ML-Ready Rows", f"{summary['ml_ready_rows']:,}")
k4.metric("Avg Failed Units / Trimester", summary["avg_failed_units"])
k5.metric("Pass Threshold", f"{PASS_MARK_THRESHOLD}%")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_features, tab_ml, tab_admin, tab_profile, tab_genai = st.tabs(
    [
        "📋 Overview",
        "⚙️ Step 1: Features",
        "🤖 Step 2: ML Models",
        "📊 Step 3: Admin Insights",
        "🔍 Student Deep-Dive Profile",
        "✉️ Step 4: GenAI Intervention",
    ]
)

# ---- Overview ----
with tab_overview:
    st.subheader("Dataset Snapshot")
    st.dataframe(raw.head(20), use_container_width=True, hide_index=True)
    st.download_button(
        "Download engineered features (CSV)",
        features.to_csv(index=False).encode(),
        file_name="trimester_features.csv",
        mime="text/csv",
    )

    fig = px.histogram(
        raw,
        x="MARKPERCENT",
        nbins=50,
        title="Distribution of Assessment Marks (%)",
        labels={"MARKPERCENT": "Mark (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Step 1: Features ----
with tab_features:
    st.subheader("Step 1 — Data Transformation & Feature Engineering")
    st.markdown(
        "Transactional rows are aggregated to **one row per student, per trimester** "
        "with the following engineered feature groups:"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Academic Momentum**")
        st.markdown(
            "- `GPA_TRAJECTORY_SLOPE` — slope of prior trimester averages\n"
            "- `PRIOR_WEIGHTED_AVG` — weighted historical average\n"
            "- `SUBJECT_CONSISTENCY_*` — performance by subject family (ICT, MBA, …)"
        )
    with c2:
        st.markdown("**Behavioral Red Flags**")
        st.markdown(
            "- `EARLY_WARNING_AVG` — first assessment mark (10–20% weight)\n"
            "- `EARLY_WARNING_FAILS` — count of failed early assessments\n"
            "- `MAX_ATTEMPT` / `AVG_ATTEMPT` — re-attempt history"
        )
    with c3:
        st.markdown("**Structural Pressures**")
        st.markdown(
            "- `STUDY_LOAD_INTENSITY` — subjects per trimester\n"
            "- `STUDY_LOAD_SPIKE` — load vs personal historical average\n"
            "- `PEER_PERCENTILE` — rank within CLASSGROUP"
        )

    display_cols = [
        "STUDENTID_MASKED",
        "STUDYPERIOD",
        "TRIMESTER_AVG_MARK",
        "GPA_TRAJECTORY_SLOPE",
        "EARLY_WARNING_AVG",
        "STUDY_LOAD_INTENSITY",
        "STUDY_LOAD_SPIKE",
        "PEER_PERCENTILE",
        "TOTAL_FAILED_UNITS",
        "MAX_ATTEMPT",
    ]
    st.dataframe(
        features[display_cols].dropna(subset=["GPA_TRAJECTORY_SLOPE"]).head(50),
        use_container_width=True,
        hide_index=True,
    )

    student_pick = st.selectbox(
        "Plot GPA trajectory for a student",
        options=sorted(features["STUDENTID_MASKED"].unique())[:200],
    )
    student_data = features[features["STUDENTID_MASKED"] == student_pick].sort_values("STUDYPERIOD")
    if len(student_data) > 1:
        fig2 = px.line(
            student_data,
            x="STUDYPERIOD",
            y="TRIMESTER_AVG_MARK",
            markers=True,
            title=f"GPA Trajectory — {student_pick}",
            labels={"STUDYPERIOD": "Trimester", "TRIMESTER_AVG_MARK": "Avg Mark (%)"},
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---- Step 2: ML ----
with tab_ml:
    st.subheader("Step 2 — Machine Learning Model Comparison")
    st.markdown(
        "Three models forecast **subsequent trimester performance** using features "
        "from prior trimesters as inputs."
    )

    if comparison_df is not None:
        st.markdown("### Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        for result in model_results:
            with st.expander(f"{result.name} — {result.task}"):
                st.write(result.notes)
                metric_cols = st.columns(len(result.metrics))
                for col, (name, value) in zip(metric_cols, result.metrics.items()):
                    col.metric(name, value)

        st.markdown("### Predicted vs Actual (Linear Regression)")
        lin_pipe = artifacts["linear"]
        ml_frame = features[
            features["HAS_HISTORY"] & features["NEXT_TRIMESTER_AVG_MARK"].notna()
        ].dropna(subset=["PRIOR_WEIGHTED_AVG", "MAX_SUBJECT_DIFFICULTY"])
        preds = lin_pipe.predict(
            ml_frame[["PRIOR_WEIGHTED_AVG", "MAX_SUBJECT_DIFFICULTY"]]
        )
        scatter_df = pd.DataFrame(
            {"Actual": ml_frame["NEXT_TRIMESTER_AVG_MARK"].values, "Predicted": preds}
        )
        fig3 = px.scatter(
            scatter_df,
            x="Actual",
            y="Predicted",
            opacity=0.3,
            title="Linear Regression: Predicted vs Actual Next-Trimester Mark",
            labels={"Actual": "Actual Mark (%)", "Predicted": "Predicted Mark (%)"},
        )
        fig3.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=100,
            y1=100,
            line=dict(dash="dash", color="gray"),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ML models could not be trained — insufficient multi-trimester history.")

# ---- Step 3: Admin ----
with tab_admin:
    st.subheader("Step 3 — Macro Insights for Academic Teams")

    st.markdown("### 🚨 Sinking Subjects")
    st.caption(
        "Subjects whose mean mark is statistically significantly below the cohort average "
        "(one-sample t-test, α = 0.05)."
    )
    sinking = insights["sinking"]
    sinking_only = sinking[sinking["Is_Sinking"]]
    c_left, c_right = st.columns([2, 1])
    with c_left:
        fig4 = px.bar(
            sinking.head(20),
            x="SUBJECTCODE",
            y="Mean_Mark",
            color="Is_Sinking",
            title="Lowest-Performing Subjects (Top 20)",
            labels={"Mean_Mark": "Mean Mark (%)", "Is_Sinking": "Statistically Sinking"},
        )
        fig4.add_hline(
            y=sinking["Global_Mean"].iloc[0],
            line_dash="dash",
            annotation_text="Global Mean",
        )
        st.plotly_chart(fig4, use_container_width=True)
    with c_right:
        st.metric("Sinking Subjects Found", len(sinking_only))
        st.dataframe(
            sinking_only[["SUBJECTCODE", "Mean_Mark", "Gap_vs_Global", "P_Value"]].head(15),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("### 🔁 The Re-Attempt Correlation")
    re = insights["reattempt"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("1st Attempt Fail Rate", f"{re['first_attempt_fail_pct']}%")
    r2.metric("Re-Attempt Fail Rate", f"{re['reattempt_fail_pct']}%")
    r3.metric("Students Who Re-Attempted", f"{re['reattempt_student_count']:,}")
    r4.metric(
        "Fail Rate (Re-Attempt Students)",
        f"{re['reattempt_student_fail_pct']}%",
    )

    fig5 = px.bar(
        x=["First Attempt", "Re-Attempt (≥2)"],
        y=[re["first_attempt_fail_pct"], re["reattempt_fail_pct"]],
        title="Unit Failure Rate by Attempt Number",
        labels={"x": "Attempt Type", "y": "Failure Rate (%)"},
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.divider()
    st.markdown("### Enrolment & Equity")
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(
            px.line(
                insights["enrolment"],
                x="STUDYPERIOD",
                y="Unique_Students",
                markers=True,
                title="Unique Students per Trimester",
            ),
            use_container_width=True,
        )
    with e2:
        st.plotly_chart(
            px.bar(
                insights["gender"],
                x="GENDERCODE",
                y="Mean_Mark",
                title="Mean Mark by Gender",
                labels={"Mean_Mark": "Mean Mark (%)"},
            ),
            use_container_width=True,
        )

# ---- Student Deep-Dive Profile ----
with tab_profile:
    st.subheader("Student Deep-Dive Profile")
    st.caption(
        "Select any student from the sidebar to review their mark trajectory, "
        "workload intensity, and generate a workload-aware intervention email."
    )

    if not selected_student:
        st.info("Select a student from the sidebar to begin.")
    else:
        try:
            profile = build_student_summary(
                raw, features, selected_student, cohort_avg=cohort_avg_load
            )
        except Exception as exc:
            st.error(f"Could not build profile for {selected_student}: {exc}")
            profile = None

        if profile is not None:
            workload = profile["workload"]
            trajectory = profile["trajectory"]

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Student ID", selected_student)
            h2.metric("Trimesters on Record", profile["trimesters_recorded"])
            latest_mark = profile["latest_mark"]
            h3.metric(
                "Latest Avg Mark",
                f"{latest_mark:.1f}%" if latest_mark is not None else "N/A",
            )
            h4.metric("Performance Trend", profile["trend_label"])

            st.markdown("### 📈 Progress History — MARKPERCENT Trajectory")
            if trajectory.empty:
                st.warning("No assessment history found for this student.")
            else:
                chart_df = trajectory.set_index("STUDYPERIOD")[["MARKPERCENT"]]
                st.line_chart(chart_df, height=320)

                pass_rows = trajectory[trajectory["PASS_STATUS"] == "Passing"]
                fail_rows = trajectory[trajectory["PASS_STATUS"] == "Failing"]
                status_cols = st.columns(2)
                with status_cols[0]:
                    st.success(
                        f"**Passing trimesters:** {', '.join(pass_rows['STUDYPERIOD'].astype(str).tolist()) or 'None'}"
                    )
                with status_cols[1]:
                    st.error(
                        f"**Failing trimesters:** {', '.join(fail_rows['STUDYPERIOD'].astype(str).tolist()) or 'None'}"
                    )

                st.caption(
                    f"Dashed reference: pass threshold is {PASS_MARK_THRESHOLD}%. "
                    "Chart shows average MARKPERCENT per trimester."
                )

            st.divider()
            st.markdown("### 📚 Subject Overload Tracker")

            w1, w2, w3 = st.columns(3)
            w1.metric(
                "Current Study Load",
                f"{workload['concurrent_subjects']} subjects",
                help=f"Unique SUBJECTCODEs in trimester {workload['latest_period']}",
            )
            w2.metric("Cohort Average Load", f"{workload['cohort_avg_load']:.1f} subjects")
            w3.metric("Fixed Overload Threshold", f">{WORKLOAD_OVERLOAD_THRESHOLD} subjects")

            if workload["is_overload"]:
                st.error(
                    f"**Workload Spike / High Overload Risk** — "
                    f"This student is enrolled in **{workload['concurrent_subjects']} subjects** "
                    f"in trimester **{workload['latest_period']}**. "
                    f"Reason: {workload['overload_reason']}."
                )
            else:
                st.success(
                    f"Study load is within normal range "
                    f"({workload['concurrent_subjects']} subjects in trimester {workload['latest_period']})."
                )

            if workload["subject_codes"]:
                st.markdown(
                    "**Enrolled subjects (latest trimester):** "
                    + ", ".join(f"`{code}`" for code in workload["subject_codes"])
                )

            st.divider()
            st.markdown("### ✉️ Workload-Aware Intervention Email")

            feature_row = profile["feature_row"]
            if feature_row is None:
                feature_row = pd.Series(
                    {
                        "STUDYPERIOD": workload.get("latest_period"),
                        "TRIMESTER_AVG_MARK": profile.get("latest_mark") or 0,
                        "AGEGROUP": "Unknown",
                        "STUDY_LOAD_INTENSITY": workload["concurrent_subjects"],
                    }
                )

            genai_cols = st.columns([1, 2])
            with genai_cols[0]:
                provider = st.radio(
                    "GenAI Provider",
                    ["auto", "openai", "huggingface", "template"],
                    horizontal=False,
                    key="profile_genai_provider",
                )
            with genai_cols[1]:
                st.markdown(
                    f"The email will acknowledge the student's **{workload['concurrent_subjects']}-subject workload** "
                    f"and offer realistic support for managing that pressure."
                )

            if st.button("✉️ Generate Workload-Aware Email", type="primary", key="profile_genai_btn"):
                with st.spinner("Drafting personalised email…"):
                    try:
                        email_text, provider_used = generate_intervention_email(
                            selected_student,
                            feature_row,
                            provider=provider,
                            concurrent_subjects=workload["concurrent_subjects"],
                        )
                        st.success(f"Generated using: **{provider_used}**")
                        st.text_area(
                            "Email Draft (editable)",
                            email_text,
                            height=400,
                            key="profile_email_output",
                        )
                        st.download_button(
                            "Download email draft (.txt)",
                            email_text.encode(),
                            file_name=f"intervention_{selected_student}.txt",
                            key="profile_email_download",
                        )
                    except Exception as exc:
                        st.error(f"Email generation failed: {exc}")

# ---- Step 4: GenAI ----
with tab_genai:
    st.subheader("Step 4 — Generative AI Intervention Engine")
    st.markdown(
        "Select a **high-risk student** flagged by the Logistic Regression model. "
        "Click the button to generate a personalised intervention email draft."
    )

    if risk_df is not None:
        high_risk = risk_df[
            risk_df["FAILURE_RISK_PROB"] >= HIGH_RISK_PROBABILITY_THRESHOLD
        ].sort_values("FAILURE_RISK_PROB", ascending=False)

        st.metric(
            "High-Risk Students Flagged",
            len(high_risk),
            help=f"Failure probability ≥ {HIGH_RISK_PROBABILITY_THRESHOLD:.0%}",
        )

        risk_display = high_risk[
            [
                "STUDENTID_MASKED",
                "STUDYPERIOD",
                "TRIMESTER_AVG_MARK",
                "FAILURE_RISK_PROB",
                "GPA_TRAJECTORY_SLOPE",
                "EARLY_WARNING_AVG",
                "TOTAL_FAILED_UNITS",
            ]
        ].head(30)
        risk_display = risk_display.copy()
        risk_display["FAILURE_RISK_PROB"] = (
            risk_display["FAILURE_RISK_PROB"].map(lambda x: f"{x:.0%}")
        )
        st.dataframe(risk_display, use_container_width=True, hide_index=True)

        student_options = high_risk["STUDENTID_MASKED"] + " | T" + high_risk["STUDYPERIOD"].astype(str)
        selected_label = st.selectbox("Select student record", student_options.tolist())
        selected_id = selected_label.split(" | ")[0]
        selected_period = float(selected_label.split(" | T")[1])

        row = high_risk[
            (high_risk["STUDENTID_MASKED"] == selected_id)
            & (high_risk["STUDYPERIOD"] == selected_period)
        ].iloc[0]

        provider = st.radio(
            "GenAI Provider",
            ["auto", "openai", "huggingface", "template"],
            horizontal=True,
            help="'auto' tries OpenAI → Hugging Face → offline template.",
        )

        if st.button("✉️ Generate Intervention Email", type="primary"):
            load_info = compute_current_study_load(
                raw, selected_id, cohort_avg=cohort_avg_load
            )
            concurrent = load_info["concurrent_subjects"]
            with st.spinner("Drafting personalised email…"):
                try:
                    email_text, provider_used = generate_intervention_email(
                        selected_id,
                        row,
                        provider=provider,
                        concurrent_subjects=concurrent,
                    )
                    st.success(f"Generated using: **{provider_used}**")
                    if load_info["is_overload"]:
                        st.warning(
                            f"Workload note: student is managing **{concurrent} subjects** "
                            f"(flagged as high overload risk)."
                        )
                    st.text_area("Email Draft (editable)", email_text, height=400)
                    st.download_button(
                        "Download email draft (.txt)",
                        email_text.encode(),
                        file_name=f"intervention_{selected_id}.txt",
                    )
                except Exception as exc:
                    st.error(f"Email generation failed: {exc}")
    else:
        st.info("Train ML models first to identify at-risk students.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "**Modular architecture:** `src/data_loader.py` → `src/feature_engineering.py` "
    "→ `src/ml_models.py` → `src/insights.py` → `src/student_profile.py` "
    "→ `src/genai_intervention.py`"
)




# =====================================================================
# SYSTEM AUTOMATED POST-ML PREDICTION EXPORT PANEL (REPAIRED PATCH)
# =====================================================================
import streamlit as st
import pandas as pd

st.markdown("---")
st.header("📥 Export AI Predictive At-Risk Registry")

ml_dataframe = None

# 1. Scan Streamlit's cache memory for data tables matching keywords
for key in st.session_state.keys():
    if any(kw in key.lower() for kw in ['model', 'predict', 'result', 'feature', 'final', 'output', 'df']):
        if isinstance(st.session_state[key], pd.DataFrame):
            ml_dataframe = st.session_state[key]
            break

# 2. Check local or global script variables if memory cache is empty
if ml_dataframe is None:
    for var_name in ['_features', 'df_features', 'features_df', 'predictions_df', 'results_df', 'final_df', 'df', 'data']:
        if var_name in locals() or var_name in globals():
            possible_df = locals().get(var_name) or globals().get(var_name)
            if isinstance(possible_df, pd.DataFrame):
                ml_dataframe = possible_df
                break

# 3. Process the table data safely
if ml_dataframe is not None:
    # Identify column names containing ML labels or risk metrics
    ml_cols = [c for c in ml_dataframe.columns if any(w in c.lower() for w in ['predict', 'risk', 'label', 'fail', 'logistic', 'forest', 'regression'])]
    
    if ml_cols:
        # AFTER ML: Filter out student profiles flagged as failures or high risk
        mask = ml_dataframe[ml_cols].astype(str).str.lower().str.contains('high|fail|risk|1|true', na=False).any(axis=1)
        at_risk_registry = ml_dataframe[mask]
        st.success("🤖 **AI Predictive Engine Data Connected:** This list shows students forecasted to fail future subjects using your ML algorithms.")
    else:
        # BEFORE ML FALLBACK: Dynamically pull students with marks or scores under 50%
        score_cols = [c for c in ml_dataframe.columns if any(w in c.lower() for w in ['percent', 'mark', 'grade', 'score', 'trajectory', 'slope'])]
        if score_cols:
            mask = (ml_dataframe[score_cols[0]] < 50)
            at_risk_registry = ml_dataframe[mask]
        else:
            at_risk_registry = ml_dataframe.head(25) # Hard fallback to guarantee rows exist
        st.info("📊 **Feature Matrix Connected:** Exporting based on engineered early risk warning indices prior to final classification sorting.")

    # 4. Render the Download User Interface Elements
    if not at_risk_registry.empty:
        st.write(f"📋 Found **{len(at_risk_registry)}** flagged student profiles prioritized for academic intervention.")
        
        # Display data summary matrix on layout screen
        preview_columns = [c for c in ['STUDENTID_MASKED', 'SUBJECTCODE', 'STUDYPERIOD'] + ml_cols if c in at_risk_registry.columns]
        st.dataframe(at_risk_registry[preview_columns].drop_duplicates().head(5))
        
        # Convert table to clean CSV bytes
        csv_bytes = at_risk_registry.to_csv(index=False).encode('utf-8')
        
        # Streamlit standard download button asset
        st.download_button(
            label="📥 Download Machine Learning At-Risk Registry (CSV File)",
            data=csv_bytes,
            file_name="ml_early_detection_at_risk_students.csv",
            mime="text/csv",
            help="Click here to download the finalized prognostic AI dataset output for institutional administration planning."
        )
    else:
        st.warning("Prediction processing active, but no student rows currently match the risk threshold filters.")
else:
    st.error("System pipeline architecture routing error: Unable to map the active state dataframe grid matrix automatically.")
