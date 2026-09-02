"""
Feature definitions for the EduPulse student-outcome model.

This module is the single source of truth for:
  - which raw dataset columns are numeric vs. nominal-categorical
  - human-readable labels for the UI (so raw column names never leak to users)
  - the target label mapping

Keeping this in one place (rather than duplicated in notebooks / API code)
is what prevents the classic "training preprocessing != serving preprocessing"
bug called out in the project brief.
"""

# Raw target column name in the source CSV
TARGET_COL = "Target"

# Class order used everywhere (model output, API response, UI). Fixed order
# matters for reproducible confusion matrices and consistent probability keys.
CLASS_LABELS = ["Dropout", "Enrolled", "Graduate"]

# Columns that are genuinely nominal categories (encoded as integer *codes*
# in the source data, e.g. "Course" 33 = Biofuel Production Technologies).
# These get one-hot encoded rather than treated as ordinal numbers.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Previous qualification",
]

# Binary flags (already 0/1) — treated as numeric, no encoding needed.
BINARY_FEATURES = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# Continuous / count numeric features — scaled.
NUMERIC_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

ALL_FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES

# Human-readable labels for the UI. Every feature shown to an advisor must
# use one of these instead of the raw dataset column name (brief 4.4).
FEATURE_LABELS = {
    "Marital status": "Marital Status",
    "Application mode": "Application Mode",
    "Application order": "Application Preference Order",
    "Course": "Course",
    "Nacionality": "Nationality",
    "Mother's qualification": "Mother's Education Level",
    "Father's qualification": "Father's Education Level",
    "Mother's occupation": "Mother's Occupation",
    "Father's occupation": "Father's Occupation",
    "Previous qualification": "Previous Qualification Type",
    "Daytime/evening attendance": "Daytime Attendance",
    "Displaced": "Displaced Student",
    "Educational special needs": "Educational Special Needs",
    "Debtor": "Has Outstanding Debt",
    "Tuition fees up to date": "Tuition Fees Up To Date",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship Holder",
    "International": "International Student",
    "Previous qualification (grade)": "Previous Qualification Grade",
    "Admission grade": "Admission Grade",
    "Age at enrollment": "Age at Enrollment",
    "Curricular units 1st sem (credited)": "1st Semester Units Credited",
    "Curricular units 1st sem (enrolled)": "1st Semester Units Enrolled",
    "Curricular units 1st sem (evaluations)": "1st Semester Units Evaluated",
    "Curricular units 1st sem (approved)": "1st Semester Units Approved",
    "Curricular units 1st sem (grade)": "1st Semester Average Grade",
    "Curricular units 1st sem (without evaluations)": "1st Semester Units w/o Evaluation",
    "Curricular units 2nd sem (credited)": "2nd Semester Units Credited",
    "Curricular units 2nd sem (enrolled)": "2nd Semester Units Enrolled",
    "Curricular units 2nd sem (evaluations)": "2nd Semester Units Evaluated",
    "Curricular units 2nd sem (approved)": "2nd Semester Units Approved",
    "Curricular units 2nd sem (grade)": "2nd Semester Average Grade",
    "Curricular units 2nd sem (without evaluations)": "2nd Semester Units w/o Evaluation",
    "Unemployment rate": "Regional Unemployment Rate",
    "Inflation rate": "Regional Inflation Rate",
    "GDP": "Regional GDP Growth",
}


def human_label(feature: str) -> str:
    """Return the UI-facing label for a raw feature name."""
    return FEATURE_LABELS.get(feature, feature)

# ---------------------------------------------------------------------------
# API field mapping: clean snake_case names (what the frontend/API contract
# uses, matching the style of the brief's example payload in section 7.1)
# <-> raw dataset column names (what the trained pipeline expects internally).
#
# This is the ONE place that translation happens. schemas/prediction.py uses
# this to build the request model; services/prediction_service.py uses it to
# convert an incoming request back into a raw-column DataFrame row before
# calling the pipeline.
# ---------------------------------------------------------------------------
API_FIELD_TO_RAW_COLUMN = {
    "marital_status": "Marital status",
    "application_mode": "Application mode",
    "application_order": "Application order",
    "course": "Course",
    "nationality": "Nacionality",
    "mothers_qualification": "Mother's qualification",
    "fathers_qualification": "Father's qualification",
    "mothers_occupation": "Mother's occupation",
    "fathers_occupation": "Father's occupation",
    "previous_qualification": "Previous qualification",
    "daytime_evening_attendance": "Daytime/evening attendance",
    "displaced": "Displaced",
    "educational_special_needs": "Educational special needs",
    "debtor": "Debtor",
    "tuition_fees_up_to_date": "Tuition fees up to date",
    "gender": "Gender",
    "scholarship_holder": "Scholarship holder",
    "international": "International",
    "previous_qualification_grade": "Previous qualification (grade)",
    "admission_grade": "Admission grade",
    "age_at_enrollment": "Age at enrollment",
    "curricular_units_1st_sem_credited": "Curricular units 1st sem (credited)",
    "curricular_units_1st_sem_enrolled": "Curricular units 1st sem (enrolled)",
    "curricular_units_1st_sem_evaluations": "Curricular units 1st sem (evaluations)",
    "curricular_units_1st_sem_approved": "Curricular units 1st sem (approved)",
    "curricular_units_1st_sem_grade": "Curricular units 1st sem (grade)",
    "curricular_units_1st_sem_without_evaluations": "Curricular units 1st sem (without evaluations)",
    "curricular_units_2nd_sem_credited": "Curricular units 2nd sem (credited)",
    "curricular_units_2nd_sem_enrolled": "Curricular units 2nd sem (enrolled)",
    "curricular_units_2nd_sem_evaluations": "Curricular units 2nd sem (evaluations)",
    "curricular_units_2nd_sem_approved": "Curricular units 2nd sem (approved)",
    "curricular_units_2nd_sem_grade": "Curricular units 2nd sem (grade)",
    "curricular_units_2nd_sem_without_evaluations": "Curricular units 2nd sem (without evaluations)",
    "unemployment_rate": "Unemployment rate",
    "inflation_rate": "Inflation rate",
    "gdp": "GDP",
}

# Reverse lookup, built automatically so it can never drift out of sync with
# the forward mapping above.
RAW_COLUMN_TO_API_FIELD = {v: k for k, v in API_FIELD_TO_RAW_COLUMN.items()}