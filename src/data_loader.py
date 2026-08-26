"""
Data ingestion layer with Streamlit Cloud file upload support.

LOCAL MODE (current):
    Reads the anonymised CSV export from the institutional data warehouse.

FUTURE LMS INTEGRATION — replace `load_assessment_data()` with one of:

    Canvas LMS (REST API):
        GET /api/v1/courses/{course_id}/students/submissions
        Authenticate with OAuth2 bearer token; paginate with `page` + `per_page`.
        Map JSON fields: user_id → STUDENTID_MASKED, score → ASSESSMENTMARK, etc.

    Moodle (Web Services):
        core_grades_get_grades  OR  mod_assign_get_submissions
        POST to {moodle_url}/webservice/rest/server.php?wstoken=...&wsfunction=...
        Register a webhook on `core_event` for real-time grade updates.

    Generic webhook receiver (recommended pattern):
        @app.route("/webhook/lms/grades", methods=["POST"])
        def receive_grades():
            payload = request.json          # normalised grade event
            df = normalise_lms_payload(payload)
            append_to_feature_store(df)     # triggers feature_engineering pipeline
            return {"status": "ok"}, 200

    Keep the function signature identical so the rest of the app needs zero changes.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import DEFAULT_DATA_PATH


def load_assessment_data(source: Path | str | None = None) -> pd.DataFrame:
    """
    Load raw transactional assessment rows.
    
    Tries multiple locations:
    1. Uploaded file (via Streamlit file uploader)
    2. User-provided path
    3. Default path from config
    4. Streamlit Cloud mounted path
    5. Current directory
    
    Returns
    -------
    pd.DataFrame with one row per assessment attempt.
    """
    
    # ============================================================
    # PRIORITY 1: Check for uploaded file in session state
    # ============================================================
    if "uploaded_data_path" in st.session_state:
        uploaded_path = Path(st.session_state["uploaded_data_path"])
        if uploaded_path.exists():
            print(f"✅ Using uploaded file: {uploaded_path}")
            try:
                df = pd.read_csv(uploaded_path)
                # Normalise column names
                df.columns = [c.upper() for c in df.columns]
                df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)
                return df
            except Exception as e:
                print(f"⚠️ Error reading uploaded file: {e}")
                # Continue to try other paths
    
    # ============================================================
    # PRIORITY 2: User-provided path
    # ============================================================
    if source is not None:
        path = Path(source)
        if path.exists():
            print(f"✅ Using user-provided path: {path}")
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]
            df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)
            return df
    
    # ============================================================
    # PRIORITY 3: Default path from config
    # ============================================================
    if DEFAULT_DATA_PATH and DEFAULT_DATA_PATH.exists():
        print(f"✅ Using default path: {DEFAULT_DATA_PATH}")
        df = pd.read_csv(DEFAULT_DATA_PATH)
        df.columns = [c.upper() for c in df.columns]
        df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)
        return df
    
    # ============================================================
    # PRIORITY 4: Streamlit Cloud mounted path
    # ============================================================
    cloud_paths = [
        Path("/mount/src/edapt-educational-analytic-and-predictive-tool/Capstone_data_20260324.csv"),
        Path("/mount/src/edapt-educational-analytic-and-predictive-tool/Capstone_data_20260324.csv"),
        Path("/app/Capstone_data_20260324.csv"),
        Path("./Capstone_data_20260324.csv"),
    ]
    
    for path in cloud_paths:
        if path.exists():
            print(f"✅ Found data at Streamlit Cloud path: {path}")
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]
            df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)
            return df
    
    # ============================================================
    # PRIORITY 5: Try just the filename
    # ============================================================
    simple_path = Path("Capstone_data_20260324.csv")
    if simple_path.exists():
        print(f"✅ Found data at: {simple_path}")
        df = pd.read_csv(simple_path)
        df.columns = [c.upper() for c in df.columns]
        df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)
        return df
    
    # ============================================================
    # No file found - raise helpful error
    # ============================================================
    raise FileNotFoundError(
        f"Dataset not found.\n\n"
        f"📂 **How to upload your data:**\n"
        f"1. Look at the left sidebar\n"
        f"2. Find the '📂 Upload Data File' section\n"
        f"3. Click 'Browse files' and select your CSV\n"
        f"4. The app will automatically load it!\n\n"
        f"📋 **Tried these locations:**\n"
        f"  - Uploaded file: {st.session_state.get('uploaded_data_path', 'Not uploaded')}\n"
        f"  - Default path: {DEFAULT_DATA_PATH}\n"
        f"  - Cloud paths: {', '.join(str(p) for p in cloud_paths[:3])}\n"
        f"  - Current directory: {Path.cwd()}\n\n"
        f"💡 **Tip:** If you've uploaded the file, try refreshing the page."
    )


def normalise_lms_payload(payload: dict) -> pd.DataFrame:
    """
    STUB — convert a Canvas/Moodle webhook JSON payload into the same schema
    as the local CSV. Implement field mapping when connecting to a live LMS.

    Example Canvas submission event → row dict:
        {
            "STUDENTID_MASKED": payload["user_id"],
            "ASSESSMENTMARK": payload["score"],
            "MAXMARK": payload["points_possible"],
            ...
        }
    """
    raise NotImplementedError(
        "LMS webhook normalisation not yet configured. "
        "Use load_assessment_data() with a CSV export for now."
    )