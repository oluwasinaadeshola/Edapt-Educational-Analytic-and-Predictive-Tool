import os
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "Capstone_data_20260324.csv")

"""Application configuration and shared constants."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Local file path — replace with LMS webhook endpoint in production.
# See src/data_loader.py for Canvas/Moodle integration notes.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Capstone_data_20260324.csv"

# Academic thresholds
PASS_MARK_THRESHOLD = 50.0  # MARKPERCENT below this = fail
EARLY_WARNING_WEIGHT_MIN = 10.0
EARLY_WARNING_WEIGHT_MAX = 20.0
FIRST_ASSESSMENT_ID = 1  # STUDYPACKAGEASSESSMENTID for first task in a unit

# Subject family prefixes for Subject Consistency feature
SUBJECT_FAMILIES = {
    "MBA": "MBA",
    "ACC": "Accounting",
    "BUS": "Business",
    "ICT": "ICT",
    "FIN": "Finance",
    "LAW": "Law",
    "MGT": "Management",
}

# GenAI provider settings (set OPENAI_API_KEY or HUGGINGFACE_API_TOKEN in env)
OPENAI_MODEL = "gpt-4o-mini"
HUGGINGFACE_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HIGH_RISK_PROBABILITY_THRESHOLD = 0.65

# Workload overload — flag when concurrent subjects exceed this OR cohort average
WORKLOAD_OVERLOAD_THRESHOLD = 3
