"""
Data ingestion layer.

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

from config import DEFAULT_DATA_PATH


def load_assessment_data(source: Path | str | None = None) -> pd.DataFrame:
    """
    Load raw transactional assessment rows.

    Parameters
    ----------
    source : path to CSV file, or None to use the default local export.

    Returns
    -------
    pd.DataFrame with one row per assessment attempt.
    """
    path = Path(source) if source is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place Capstone_data_20260324.csv in the project root, "
            "or point `source` to your LMS export."
        )

    df = pd.read_csv(path)

    # Normalise column names to uppercase for consistency with LMS field mapping
    df.columns = [c.upper() for c in df.columns]

    # STUDYPERIOD stored as float (e.g. 23.2) — keep numeric for ordering
    df["STUDYPERIOD"] = df["STUDYPERIOD"].astype(float)

    return df


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
