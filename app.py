"""
EdAPT - Educational Analytics and Predictive Tools
User-Friendly Interface - Designed for Teachers and Academic Staff
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is on the path
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
from src.ml_models_2 import train_and_compare_models
from src.student_profile import (
    build_student_summary,
    compute_cohort_avg_study_load,
    compute_current_study_load,
    list_students,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EDAPT - Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern, Clean Design
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0 1rem;
    }
    
    /* Welcome header */
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .welcome-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    .welcome-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Card styling */
    .card-modern {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
        height: 100%;
    }
    .card-modern:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .card-modern h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-modern .big-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0.2rem 0;
    }
    .card-modern .sub-text {
        color: #718096;
        font-size: 0.85rem;
    }
    
    /* Status badges */
    .badge-high {
        background: #fed7d7;
        color: #c53030;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background: #feebc8;
        color: #c05621;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background: #c6f6d5;
        color: #276749;
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Upload area */
    .upload-box {
        border: 3px dashed #cbd5e0;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
        background: #fafafa;
        cursor: pointer;
    }
    .upload-box:hover {
        border-color: #667eea;
        background: #f7fafc;
    }
    .upload-box .icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .upload-box .title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2d3748;
    }
    .upload-box .sub {
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f7fafc;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        color: #4a5568;
    }
    .stTabs [aria-selected="true"] {
        background: white !important;
        color: #667eea !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* Metric cards in overview */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #edf2f7;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f7fafc;
    }
    
    /* Progress indicators */
    .progress-success {
        color: #48bb78;
        font-weight: 600;
    }
    .progress-warning {
        color: #ed8936;
        font-weight: 600;
    }
    .progress-danger {
        color: #fc8181;
        font-weight: 600;
    }
    
    /* Tooltip style */
    .tooltip {
        color: #718096;
        font-size: 0.8rem;
        cursor: help;
        border-bottom: 1px dashed #cbd5e0;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .btn-primary button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        color: #a0aec0;
        font-size: 0.85rem;
        border-top: 1px solid #edf2f7;
        margin-top: 2rem;
    }
    
    /* Status messages */
    .status-success {
        background: #c6f6d5;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: #276749;
        border-left: 4px solid #48bb78;
    }
    .status-info {
        background: #bee3f8;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: #2b6cb0;
        border-left: 4px solid #3182ce;
    }
    .status-warning {
        background: #feebc8;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        color: #c05621;
        border-left: 4px solid #dd6b20;
    }
    
    /* Simple steps */
    .step {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 1rem;
        background: #f7fafc;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .step-number {
        background: #667eea;
        color: white;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Functions
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading your data...")
def load_data(path: str) -> pd.DataFrame:
    return load_assessment_data(path)

@st.cache_data(show_spinner="Analyzing student patterns...")
def engineer_features(_raw: pd.DataFrame) -> pd.DataFrame:
    return build_trimester_features(_raw)

@st.cache_data(show_spinner="Training prediction models...")
def run_models(_features: pd.DataFrame):
    return train_and_compare_models(_features)

@st.cache_data(show_spinner="Calculating institutional insights...")
def admin_insights(_raw: pd.DataFrame, _units: pd.DataFrame):
    return {
        "sinking": compute_sinking_subjects(_raw),
        "reattempt": compute_reattempt_correlation(_units),
        "enrolment": trimester_enrolment_trends(_raw),
        "gender": gender_performance_gap(_raw),
    }

@st.cache_data(show_spinner="Computing workload benchmarks...")
def cohort_study_load(_raw: pd.DataFrame) -> float:
    return compute_cohort_avg_study_load(_raw)

# ---------------------------------------------------------------------------
# Welcome Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="welcome-header">
    <h1>🎓 EDAPT</h1>
    <p>Educational Analytics & Predictive Tool — Your early warning system for student success</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar - Simple Upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Step 1: Upload Your Data")
    
    # Simple file uploader with nice styling
    uploaded_file = st.file_uploader(
        "Choose your student data file (CSV format)",
        type=['csv'],
        help="Upload the CSV file from your institution's data system"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state['uploaded_data_path'] = tmp_file.name
            st.session_state['uploaded_file_name'] = uploaded_file.name
        
        # Success message
        st.markdown(f"""
        <div class="status-success">
            ✅ <strong>File uploaded successfully!</strong><br>
            <span style="font-size:0.85rem;">{uploaded_file.name} ({len(uploaded_file.getvalue()) / 1024:.1f} KB)</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # If data is loaded, show student selector
    if "data_loaded" in st.session_state and st.session_state.data_loaded:
        st.markdown("### 👤 Step 2: Select a Student")
        
        all_students = st.session_state.get("all_students", [])
        if all_students:
            selected_student = st.selectbox(
                "Choose a student to view their profile",
                options=all_students,
                help="Select any student to see their detailed performance"
            )
            st.session_state.selected_student = selected_student
    
    st.divider()
    
    # Help section
    with st.expander("❓ Need Help?"):
        st.markdown("""
        **How to use EDAPT:**
        
        1. **Upload** your CSV data file above
        2. **Wait** for the analysis to complete
        3. **Explore** the tabs to see insights
        4. **Find** at-risk students who need help
        
        **Need your data file?**
        Contact your IT department for the student data export.
        """)

# ---------------------------------------------------------------------------
# Main Content - Load Data
# ---------------------------------------------------------------------------
data_path = st.session_state.get('uploaded_data_path', str(DEFAULT_DATA_PATH))

try:
    raw = load_data(data_path)
    st.session_state.data_loaded = True
except FileNotFoundError:
    st.session_state.data_loaded = False
    # Show friendly upload prompt
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <h2 style="color:#4a5568; font-weight:600;">No Data Loaded Yet</h2>
        <p style="color:#718096; font-size:1.1rem; max-width:500px; margin:0 auto;">
            Please upload your student data CSV file using the file uploader in the sidebar.
        </p>
        <div style="margin-top:2rem; display:flex; gap:0.5rem; justify-content:center; flex-wrap:wrap;">
            <span style="background:#edf2f7; padding:0.5rem 1rem; border-radius:8px; font-size:0.9rem;">📁 Click "Browse files"</span>
            <span style="background:#edf2f7; padding:0.5rem 1rem; border-radius:8px; font-size:0.9rem;">⬆️ Select your CSV</span>
            <span style="background:#edf2f7; padding:0.5rem 1rem; border-radius:8px; font-size:0.9rem;">✅ Wait for analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Process Data
# ---------------------------------------------------------------------------
features = engineer_features(raw)
units = build_unit_records(raw)
summary = get_feature_summary(features)

try:
    model_results, comparison_df, artifacts = run_models(features)
    risk_df = artifacts["risk_scores"]
except ValueError:
    model_results, comparison_df, artifacts, risk_df = None, None, None, None

insights = admin_insights(raw, units)
cohort_avg_load = cohort_study_load(raw)
all_students = list_students(raw)
st.session_state.all_students = all_students

if "selected_student" not in st.session_state and all_students:
    st.session_state.selected_student = all_students[0]

# ---------------------------------------------------------------------------
# KPI Cards - Simple Overview
# ---------------------------------------------------------------------------
st.markdown("### 📊 Your Data at a Glance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{summary['students']:,}</div>
        <div class="label">🎓 Students</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{summary['trimester_records']:,}</div>
        <div class="label">📅 Study Periods</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{summary['ml_ready_rows']:,}</div>
        <div class="label">🤖 Records Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Calculate pass rate
    pass_rate = (raw['MARKPERCENT'] >= 50).mean() * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="value" style="color: {'#48bb78' if pass_rate > 70 else '#ed8936' if pass_rate > 50 else '#fc8181'}">{pass_rate:.1f}%</div>
        <div class="label">✅ Overall Pass Rate</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="value">{raw['SUBJECTCODE'].nunique()}</div>
        <div class="label">📚 Subjects</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs - Simplified Labels
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview",
    "🔍 Find At-Risk Students",
    "📊 Subject Insights", 
    "👤 Student Profile",
    "🤖 Predictions",
    "✉️ Intervention Emails"
])

# ---------------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### 📋 Overview of Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="background:#f7fafc; padding:1rem; border-radius:12px; margin-bottom:1rem;">
            <p style="margin:0; color:#4a5568; font-weight:500;">📌 What you're looking at:</p>
            <p style="margin:0.5rem 0 0 0; color:#718096; font-size:0.95rem;">
                This dashboard helps you find students who may need extra support.
                The system has analyzed your data and identified patterns that predict student success or struggle.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple chart - Mark Distribution
        st.markdown("#### 📈 Student Marks Distribution")
        fig = px.histogram(
            raw,
            x="MARKPERCENT",
            nbins=30,
            title="",
            labels={"MARKPERCENT": "Student Mark (%)"},
            color_discrete_sequence=["#667eea"]
        )
        fig.add_vline(x=50, line_dash="dash", line_color="#fc8181", annotation_text="Pass Threshold (50%)")
        fig.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("💡 Students scoring below 50% may need additional support. Hover over the bars to see exact numbers.")
    
    with col2:
        st.markdown("#### 🔍 Quick Insights")
        
        # Calculate some simple insights
        total_students = summary['students']
        high_risk_count = len(risk_df[risk_df["FAILURE_RISK_PROB"] >= HIGH_RISK_PROBABILITY_THRESHOLD]) if risk_df is not None else 0
        
        st.markdown(f"""
        <div class="card-modern">
            <h3>⚠️ At-Risk Students</h3>
            <div class="big-number" style="color: {'#fc8181' if high_risk_count > 0 else '#48bb78'}">{high_risk_count:,}</div>
            <div class="sub-text">Students flagged as needing immediate attention</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show latest period
        latest_period = raw['STUDYPERIOD'].max()
        st.markdown(f"""
        <div class="card-modern">
            <h3>📅 Latest Data</h3>
            <div class="big-number">{latest_period}</div>
            <div class="sub-text">Most recent study period analyzed</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data
        st.markdown("#### 📄 Data Preview")
        st.dataframe(raw.head(5), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 2: Find At-Risk Students
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### 🔍 Find Students Who Need Support")
    
    if risk_df is not None:
        high_risk = risk_df[
            risk_df["FAILURE_RISK_PROB"] >= HIGH_RISK_PROBABILITY_THRESHOLD
        ].sort_values("FAILURE_RISK_PROB", ascending=False)
        
        st.markdown(f"""
        <div style="background:#f7fafc; padding:1rem 1.5rem; border-radius:12px; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <span style="font-weight:600; font-size:1.1rem;">🚨 High-Risk Students</span>
                <span style="color:#718096; margin-left:0.5rem;">Students with the highest chance of failing</span>
            </div>
            <span class="badge-high">{len(high_risk)} students</span>
        </div>
        """, unsafe_allow_html=True)
        
        if len(high_risk) > 0:
            # Display high-risk students in a friendly table
            display_df = high_risk.head(20)[
                ["STUDENTID_MASKED", "TRIMESTER_AVG_MARK", "FAILURE_RISK_PROB", "TOTAL_FAILED_UNITS"]
            ].copy()
            display_df.columns = ["Student ID", "Current Mark", "Risk Level", "Failed Units"]
            display_df["Risk Level"] = display_df["Risk Level"].apply(lambda x: f"{x:.0%}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.caption("💡 These students should be contacted for support. Visit the 'Intervention Emails' tab to generate messages.")
            
            # ================================================================
            # 📥 DOWNLOAD ALL AT-RISK STUDENTS (ADD THIS SECTION)
            # ================================================================
            st.divider()
            st.markdown("### 📥 Export All At-Risk Students")
            
            # Create a copy with ALL at-risk students (not just the first 20)
            all_at_risk = high_risk.copy()
            
            # Add priority levels based on risk probability
            all_at_risk["RISK_LEVEL"] = all_at_risk["FAILURE_RISK_PROB"].apply(
                lambda x: "🔴 High" if x >= 0.8 else "🟡 Medium" if x >= 0.65 else "🟢 Low"
            )
            all_at_risk["PRIORITY"] = all_at_risk["FAILURE_RISK_PROB"].apply(
                lambda x: "🔴 URGENT" if x >= 0.8 else "🟡 Monitor" if x >= 0.65 else "🟢 Review"
            )
            
            # Select key columns for export (only those that exist)
            export_cols = [
                "STUDENTID_MASKED",
                "STUDYPERIOD", 
                "TRIMESTER_AVG_MARK",
                "FAILURE_RISK_PROB",
                "RISK_LEVEL",
                "PRIORITY",
                "TOTAL_FAILED_UNITS",
                "MAX_ATTEMPT",
                "EARLY_WARNING_AVG",
            ]
            # Only include columns that actually exist
            export_cols = [col for col in export_cols if col in all_at_risk.columns]
            export_df = all_at_risk[export_cols].copy()
            
            # Format risk as percentage
            export_df["FAILURE_RISK_PROB"] = export_df["FAILURE_RISK_PROB"].apply(lambda x: f"{x:.1%}")
            
            # Show the total count
            st.info(f"📋 **{len(export_df)}** at-risk students ready for export")
            
            # Show preview of who will be exported
            st.markdown("#### 📊 Preview (First 10 Students)")
            st.dataframe(export_df.head(10), use_container_width=True, hide_index=True)
            
            # Create columns for download options
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Download button for ALL at-risk students
                csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download All {len(export_df)} At-Risk Students",
                    data=csv_bytes,
                    file_name="edapt_all_at_risk_students.csv",
                    mime="text/csv",
                    help=f"Download all {len(export_df)} at-risk students for intervention planning",
                    use_container_width=True
                )
            
            with col_d2:
                # Filter by priority level
                priority_filter = st.multiselect(
                    "Filter by Priority (Optional)",
                    options=["🔴 URGENT", "🟡 Monitor", "🟢 Review"],
                    default=["🔴 URGENT", "🟡 Monitor", "🟢 Review"],
                    help="Select which priority levels to include in the download"
                )
                
                if priority_filter:
                    filtered_df = export_df[export_df["PRIORITY"].isin(priority_filter)]
                    if not filtered_df.empty:
                        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {len(filtered_df)} Filtered Students",
                            data=filtered_csv,
                            file_name="edapt_filtered_at_risk_students.csv",
                            mime="text/csv",
                            help="Download only the selected priority levels",
                            use_container_width=True
                        )
        else:
            st.success("🎉 No high-risk students found! Great job!")
    else:
        st.info("📊 Run the predictions first to see at-risk students.")
# ---------------------------------------------------------------------------
# TAB 3: Subject Insights
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("### 📊 Subject Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📉 Sinking Subjects")
        st.caption("Subjects where students score significantly lower than average")
        
        sinking = insights["sinking"]
        sink_display = sinking.head(10)[["SUBJECTCODE", "Mean_Mark", "Gap_vs_Global", "Is_Sinking"]].copy()
        sink_display.columns = ["Subject", "Avg Mark", "vs Average", "Sinking?"]
        
        st.dataframe(sink_display, use_container_width=True, hide_index=True)
        st.caption("🔴 Subjects marked as 'Sinking' need attention - consider additional resources.")
    
    with col2:
        st.markdown("#### 🔁 Re-Attempt Impact")
        st.caption("What happens when students retake subjects")
        
        re = insights["reattempt"]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="card-modern" style="text-align:center;">
                <div style="color:#718096; font-size:0.85rem;">First Attempt</div>
                <div style="font-size:2rem; font-weight:700; color:#48bb78;">{re['first_attempt_fail_pct']}%</div>
                <div style="color:#718096; font-size:0.8rem;">fail rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="card-modern" style="text-align:center;">
                <div style="color:#718096; font-size:0.85rem;">Re-Attempt</div>
                <div style="font-size:2rem; font-weight:700; color:#fc8181;">{re['reattempt_fail_pct']}%</div>
                <div style="color:#718096; font-size:0.8rem;">fail rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background:#edf2f7; padding:0.75rem 1rem; border-radius:10px; margin-top:0.5rem;">
            <span style="font-weight:600;">💡 Key Insight:</span>
            <span style="color:#4a5568;">Students who retake subjects are <span style="color:#fc8181; font-weight:700;">{re['reattempt_fail_pct'] / re['first_attempt_fail_pct']:.0f}x</span> more likely to fail again.</span>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TAB 4: Student Profile
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("### 👤 Student Profile")
    
    if not all_students:
        st.info("No students found in the dataset.")
    else:
        student_id = st.session_state.get("selected_student", all_students[0])
        
        try:
            profile = build_student_summary(raw, features, student_id, cohort_avg=cohort_avg_load)
        except Exception:
            profile = None
        
        if profile is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="card-modern" style="text-align:center;">
                    <div style="color:#718096; font-size:0.85rem;">Student</div>
                    <div style="font-size:1.2rem; font-weight:600;">{student_id}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                trend = profile.get("trend_label", "Unknown")
                trend_color = "#48bb78" if trend == "Improving" else "#fc8181" if trend == "Declining" else "#ed8936"
                st.markdown(f"""
                <div class="card-modern" style="text-align:center;">
                    <div style="color:#718096; font-size:0.85rem;">Performance Trend</div>
                    <div style="font-size:1.2rem; font-weight:600; color:{trend_color};">{trend}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                latest = profile.get("latest_mark", "N/A")
                if latest != "N/A":
                    latest = f"{latest:.1f}%"
                    latest_color = "#48bb78" if float(latest.replace("%","")) >= 50 else "#fc8181"
                else:
                    latest_color = "#718096"
                st.markdown(f"""
                <div class="card-modern" style="text-align:center;">
                    <div style="color:#718096; font-size:0.85rem;">Latest Average</div>
                    <div style="font-size:1.2rem; font-weight:600; color:{latest_color};">{latest}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                load = profile.get("workload", {}).get("concurrent_subjects", "N/A")
                load_color = "#fc8181" if load > 3 else "#48bb78"
                st.markdown(f"""
                <div class="card-modern" style="text-align:center;">
                    <div style="color:#718096; font-size:0.85rem;">Subjects Taking</div>
                    <div style="font-size:1.2rem; font-weight:600; color:{load_color};">{load}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Trajectory chart
            trajectory = profile.get("trajectory")
            if trajectory is not None and not trajectory.empty:
                st.markdown("#### 📈 Performance Over Time")
                fig = px.line(
                    trajectory,
                    x="STUDYPERIOD",
                    y="MARKPERCENT",
                    markers=True,
                    title="",
                    labels={"STUDYPERIOD": "Study Period", "MARKPERCENT": "Average Mark (%)"}
                )
                fig.add_hline(y=50, line_dash="dash", line_color="#fc8181", annotation_text="Pass Threshold")
                fig.update_layout(
                    height=300,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Workload info
            workload = profile.get("workload", {})
            if workload.get("is_overload"):
                st.warning(f"⚠️ **Workload Alert:** This student is taking {workload['concurrent_subjects']} subjects, which exceeds the recommended load.")
            else:
                st.success(f"✅ Study load is within normal range ({workload['concurrent_subjects']} subjects).")
        else:
            st.warning("Could not load profile for this student.")

# ---------------------------------------------------------------------------
# TAB 5: Predictions
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("### 🤖 How We Predict Student Success")
    
    if comparison_df is not None:
        st.markdown("""
        <div style="background:#f7fafc; padding:1rem; border-radius:12px; margin-bottom:1.5rem;">
            <p style="margin:0; color:#4a5568;">
                The system uses <strong>three different methods</strong> to predict student success.
                Each method looks at different factors and gives us a more complete picture.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple model comparison
        col1, col2, col3 = st.columns(3)
        
        models = [
            ("📊 Logistic Regression", "Pass/Fail", "Looks at marks, failed units, age group"),
            ("📈 Linear Regression", "Exact Mark", "Looks at prior marks, subject difficulty"),
            ("🌲 Random Forest", "Pass/Fail", "Looks at country, gender, attempts")
        ]
        
        for col, (name, task, desc) in zip([col1, col2, col3], models):
            with col:
                st.markdown(f"""
                <div class="card-modern">
                    <h3>{name}</h3>
                    <p style="font-size:0.9rem; color:#4a5568;"><strong>Task:</strong> {task}</p>
                    <p style="font-size:0.85rem; color:#718096;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show comparison table
        st.markdown("#### 📊 Model Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        #st.caption("💡 Higher accuracy numbers mean more reliable predictions. Random Forest typically performs best.")
        
        # Show prediction chart if available
        if artifacts is not None and "linear" in artifacts:
            st.markdown("#### 📈 Predicted vs Actual Marks")
            lin_pipe = artifacts["linear"]
            ml_frame = features[
                features["HAS_HISTORY"] & features["NEXT_TRIMESTER_AVG_MARK"].notna()
            ].dropna(subset=["PRIOR_WEIGHTED_AVG", "MAX_SUBJECT_DIFFICULTY"])
            
            if len(ml_frame) > 0:
                preds = lin_pipe.predict(
                    ml_frame[["PRIOR_WEIGHTED_AVG", "MAX_SUBJECT_DIFFICULTY"]]
                )
                scatter_df = pd.DataFrame(
                    {"Actual": ml_frame["NEXT_TRIMESTER_AVG_MARK"].values, "Predicted": preds}
                )
                fig = px.scatter(
                    scatter_df,
                    x="Actual",
                    y="Predicted",
                    opacity=0.4,
                    title="",
                    labels={"Actual": "Actual Mark (%)", "Predicted": "Predicted Mark (%)"}
                )
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=100,
                    y1=100,
                    line=dict(dash="dash", color="gray")
                )
                fig.update_layout(
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 Points close to the diagonal line are perfectly predicted. The closer the points cluster to the line, the better the model.")
    else:
        st.info("📊 Train the prediction models to see results here.")

# ---------------------------------------------------------------------------
# TAB 6: Intervention Emails
# ---------------------------------------------------------------------------
with tab6:
    st.markdown("### ✉️ Generate Support Emails for At-Risk Students")
    
    st.markdown("""
    <div style="background:#f7fafc; padding:1rem; border-radius:12px; margin-bottom:1.5rem;">
        <p style="margin:0; color:#4a5568;">
            💡 Select a student and generate a personalised email to offer them support. 
            The email will be tailored to their specific situation and risk factors.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if risk_df is not None:
        high_risk = risk_df[
            risk_df["FAILURE_RISK_PROB"] >= HIGH_RISK_PROBABILITY_THRESHOLD
        ].sort_values("FAILURE_RISK_PROB", ascending=False)
        
        if len(high_risk) > 0:
            st.info(f"📋 {len(high_risk)} students are flagged as needing support. Select one below to generate an email.")
            
            # Student selector
            student_options = high_risk["STUDENTID_MASKED"].tolist()
            selected_student = st.selectbox(
                "Select a student to support",
                options=student_options,
                help="Choose a student from the list of at-risk students"
            )
            
            if selected_student:
                student_data = high_risk[high_risk["STUDENTID_MASKED"] == selected_student].iloc[0]
                
                # Show student's current situation
                st.markdown("#### 📊 Student Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Mark", f"{student_data['TRIMESTER_AVG_MARK']:.1f}%")
                with col2:
                    st.metric("Risk Level", f"{student_data['FAILURE_RISK_PROB']:.0%}")
                with col3:
                    st.metric("Failed Units", f"{student_data['TOTAL_FAILED_UNITS']}")
                
                # Generate email button
                if st.button("✉️ Generate Support Email", type="primary"):
                    load_info = compute_current_study_load(
                        raw, selected_student, cohort_avg=cohort_avg_load
                    )
                    concurrent = load_info["concurrent_subjects"]
                    
                    with st.spinner("Creating a personalised support email..."):
                        try:
                            email_text, provider_used = generate_intervention_email(
                                selected_student,
                                student_data,
                                provider="template",
                                concurrent_subjects=concurrent,
                            )
                            st.success(f"✅ Email generated successfully!")
                            
                            st.markdown("#### 📧 Email Draft")
                            st.text_area(
                                "Review and edit the email below before sending:",
                                email_text,
                                height=400,
                                key="email_draft"
                            )
                            
                            # Download button
                            st.download_button(
                                "📥 Download Email",
                                email_text.encode(),
                                file_name=f"support_email_{selected_student}.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"Could not generate email: {e}")
        else:
            st.success("🎉 No high-risk students found! All students are on track.")
    else:
        st.info("📊 Run the predictions first to identify at-risk students.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    EDAPT - Educational Analytics & Predictive Tool 🎓
</div>
""", unsafe_allow_html=True)