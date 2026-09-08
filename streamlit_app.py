"""
EduPulse — Streamlit demo

A lightweight, free-to-host demo of the EduPulse student dropout-risk model.
This reuses the SAME trained pipeline and the SAME feature/explain code as
the FastAPI + React app (ml/src/*) — it is not a re-implementation of the
model, just a different, simpler front door onto it. The original
FastAPI + React app (backend/, frontend/) is left untouched.

Why this file exists: FastAPI + React needs a host that can run a
long-lived server process, and every free tier tried for that required
adding a card for verification. Streamlit Community Cloud hosts this file
directly from GitHub for free, no card, the same way the other two ML
projects in this account are already deployed.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Make ml/src importable so this app reuses the EXACT SAME feature
# definitions and SHAP logic as training/serving — never a second copy
# that could drift out of sync.
REPO_ROOT = Path(__file__).resolve().parent
ML_SRC_DIR = REPO_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from features import API_FIELD_TO_RAW_COLUMN, human_label  # noqa: E402
from explain import explain_instance  # noqa: E402

ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "model_pipeline.joblib"

st.set_page_config(page_title="EduPulse — Student Dropout Risk", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifact():
    """Load the trained pipeline once per app session, not per request."""
    if not ARTIFACT_PATH.exists():
        return None, f"Model artifact not found at {ARTIFACT_PATH}."
    try:
        artifact = joblib.load(ARTIFACT_PATH)
        return artifact, None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not load the model artifact: {exc}"


@st.cache_resource
def load_shap_background():
    """Build the SHAP background sample once. Only Logistic Regression's
    LinearExplainer needs this; tree models don't, but building it is cheap
    and harmless either way since it's cached."""
    try:
        from data import load_xy

        artifact, err = load_artifact()
        if artifact is None:
            return None, err
        preprocessor = artifact["pipeline"].named_steps["preprocess"]
        X, _ = load_xy()
        sample = X.sample(100, random_state=42)
        return preprocessor.transform(sample), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not build SHAP background sample: {exc}"


def compute_risk_band(dropout_probability: float) -> str:
    """Same thresholds as the FastAPI backend's prediction_service.py —
    a product decision, not a statistically derived boundary."""
    if dropout_probability >= 0.60:
        return "High"
    elif dropout_probability >= 0.30:
        return "Medium"
    return "Low"


def run_prediction(artifact, raw_row_df: pd.DataFrame):
    """Mirrors backend/app/services/prediction_service.py::run_prediction,
    reimplemented here (not imported) since that module is wired into the
    FastAPI app package rather than being a standalone function."""
    pipeline = artifact["pipeline"]
    class_labels = artifact["class_labels"]
    is_xgb = artifact["is_xgb"]
    label_encoder = artifact["label_encoder"]

    if is_xgb:
        pred_encoded = pipeline.predict(raw_row_df)[0]
        predicted_class = label_encoder.inverse_transform([pred_encoded])[0]
        proba_raw = pipeline.predict_proba(raw_row_df)[0]
        proba_columns = label_encoder.inverse_transform(range(len(proba_raw)))
    else:
        predicted_class = pipeline.predict(raw_row_df)[0]
        proba_raw = pipeline.predict_proba(raw_row_df)[0]
        proba_columns = pipeline.named_steps["clf"].classes_

    col_index = {c: i for i, c in enumerate(proba_columns)}
    probabilities = {label: round(float(proba_raw[col_index[label]]), 4) for label in class_labels}

    dropout_prob = probabilities.get("Dropout", 0.0)
    risk_band = compute_risk_band(dropout_prob)

    return {
        "prediction": str(predicted_class),
        "risk_band": risk_band,
        "probabilities": probabilities,
    }


# Sensible defaults taken from the API's own example payload
# (backend/app/schemas/prediction.py, StudentFeaturesRequest.Config) —
# most of these are UCI numeric codes an end user has no way to know
# off-hand, so a realistic starting point matters more than a blank 0.
DEFAULTS = {
    "marital_status": 1,
    "application_mode": 17,
    "application_order": 1,
    "course": 9254,
    "nationality": 1,
    "mothers_qualification": 1,
    "fathers_qualification": 1,
    "mothers_occupation": 1,
    "fathers_occupation": 1,
    "previous_qualification": 1,
    "daytime_evening_attendance": 1,
    "displaced": 0,
    "educational_special_needs": 0,
    "debtor": 0,
    "tuition_fees_up_to_date": 1,
    "gender": 0,
    "scholarship_holder": 0,
    "international": 0,
    "previous_qualification_grade": 126.0,
    "admission_grade": 122.6,
    "age_at_enrollment": 20,
    "curricular_units_1st_sem_credited": 0,
    "curricular_units_1st_sem_enrolled": 6,
    "curricular_units_1st_sem_evaluations": 6,
    "curricular_units_1st_sem_approved": 5,
    "curricular_units_1st_sem_grade": 12.4,
    "curricular_units_1st_sem_without_evaluations": 0,
    "curricular_units_2nd_sem_credited": 0,
    "curricular_units_2nd_sem_enrolled": 6,
    "curricular_units_2nd_sem_evaluations": 6,
    "curricular_units_2nd_sem_approved": 4,
    "curricular_units_2nd_sem_grade": 11.8,
    "curricular_units_2nd_sem_without_evaluations": 0,
    "unemployment_rate": 10.8,
    "inflation_rate": 1.4,
    "gdp": 1.74,
}

BINARY_LABELS = {
    "daytime_evening_attendance": ("Evening", "Daytime"),
    "displaced": ("No", "Yes"),
    "educational_special_needs": ("No", "Yes"),
    "debtor": ("No", "Yes"),
    "tuition_fees_up_to_date": ("No", "Yes"),
    "gender": ("Female", "Male"),
    "scholarship_holder": ("No", "Yes"),
    "international": ("No", "Yes"),
}


def render_form():
    """One st.form covering all 36 model inputs, grouped into sections so
    the page doesn't read as one undifferentiated wall of fields."""
    values = {}
    with st.form("student_features"):
        st.subheader("Background & Application")
        c1, c2 = st.columns(2)
        with c1:
            values["marital_status"] = st.number_input(
                "Marital Status (UCI code)", min_value=1, max_value=6, value=DEFAULTS["marital_status"]
            )
            values["application_mode"] = st.number_input(
                "Application Mode (UCI code)", min_value=0, value=DEFAULTS["application_mode"]
            )
            values["application_order"] = st.number_input(
                "Application Preference Order", min_value=0, value=DEFAULTS["application_order"]
            )
            values["course"] = st.number_input("Course (UCI code)", min_value=0, value=DEFAULTS["course"])
            values["nationality"] = st.number_input(
                "Nationality (UCI code)", min_value=0, value=DEFAULTS["nationality"]
            )
            values["previous_qualification"] = st.number_input(
                "Previous Qualification Type (UCI code)", min_value=0, value=DEFAULTS["previous_qualification"]
            )
        with c2:
            values["mothers_qualification"] = st.number_input(
                "Mother's Education Level (UCI code)", min_value=0, value=DEFAULTS["mothers_qualification"]
            )
            values["fathers_qualification"] = st.number_input(
                "Father's Education Level (UCI code)", min_value=0, value=DEFAULTS["fathers_qualification"]
            )
            values["mothers_occupation"] = st.number_input(
                "Mother's Occupation (UCI code)", min_value=0, value=DEFAULTS["mothers_occupation"]
            )
            values["fathers_occupation"] = st.number_input(
                "Father's Occupation (UCI code)", min_value=0, value=DEFAULTS["fathers_occupation"]
            )
            values["previous_qualification_grade"] = st.number_input(
                "Previous Qualification Grade (0-200)",
                min_value=0.0, max_value=200.0, value=DEFAULTS["previous_qualification_grade"],
            )
            values["admission_grade"] = st.number_input(
                "Admission Grade (0-200)", min_value=0.0, max_value=200.0, value=DEFAULTS["admission_grade"]
            )
        values["age_at_enrollment"] = st.number_input(
            "Age at Enrollment", min_value=15, max_value=100, value=DEFAULTS["age_at_enrollment"]
        )

        st.subheader("Status Flags")
        flag_fields = [
            "daytime_evening_attendance", "displaced", "educational_special_needs", "debtor",
            "tuition_fees_up_to_date", "gender", "scholarship_holder", "international",
        ]
        flag_cols = st.columns(4)
        for i, field in enumerate(flag_fields):
            no_label, yes_label = BINARY_LABELS[field]
            with flag_cols[i % 4]:
                choice = st.radio(
                    human_label(API_FIELD_TO_RAW_COLUMN[field]),
                    options=[0, 1],
                    format_func=lambda v, nl=no_label, yl=yes_label: nl if v == 0 else yl,
                    index=DEFAULTS[field],
                    key=f"flag_{field}",
                )
                values[field] = choice

        st.subheader("1st Semester Performance")
        s1_cols = st.columns(3)
        s1_fields = [
            ("curricular_units_1st_sem_credited", "Units Credited", 0.0, None),
            ("curricular_units_1st_sem_enrolled", "Units Enrolled", 0.0, None),
            ("curricular_units_1st_sem_evaluations", "Units Evaluated", 0.0, None),
            ("curricular_units_1st_sem_approved", "Units Approved", 0.0, None),
            ("curricular_units_1st_sem_grade", "Average Grade (0-20)", 0.0, 20.0),
            ("curricular_units_1st_sem_without_evaluations", "Units w/o Evaluation", 0.0, None),
        ]
        for i, (field, label, lo, hi) in enumerate(s1_fields):
            with s1_cols[i % 3]:
                values[field] = st.number_input(label, min_value=lo, max_value=hi, value=float(DEFAULTS[field]), key=field)

        st.subheader("2nd Semester Performance")
        s2_cols = st.columns(3)
        s2_fields = [
            ("curricular_units_2nd_sem_credited", "Units Credited", 0.0, None),
            ("curricular_units_2nd_sem_enrolled", "Units Enrolled", 0.0, None),
            ("curricular_units_2nd_sem_evaluations", "Units Evaluated", 0.0, None),
            ("curricular_units_2nd_sem_approved", "Units Approved", 0.0, None),
            ("curricular_units_2nd_sem_grade", "Average Grade (0-20)", 0.0, 20.0),
            ("curricular_units_2nd_sem_without_evaluations", "Units w/o Evaluation", 0.0, None),
        ]
        for i, (field, label, lo, hi) in enumerate(s2_fields):
            with s2_cols[i % 3]:
                values[field] = st.number_input(label, min_value=lo, max_value=hi, value=float(DEFAULTS[field]), key=field)

        st.subheader("Macroeconomic Context")
        m1, m2, m3 = st.columns(3)
        with m1:
            values["unemployment_rate"] = st.number_input(
                "Regional Unemployment Rate (%)", value=DEFAULTS["unemployment_rate"]
            )
        with m2:
            values["inflation_rate"] = st.number_input("Regional Inflation Rate (%)", value=DEFAULTS["inflation_rate"])
        with m3:
            values["gdp"] = st.number_input("Regional GDP Growth", value=DEFAULTS["gdp"])

        submitted = st.form_submit_button("Predict dropout risk")
    return submitted, values


def main():
    st.title("🎓 EduPulse — Student Dropout Risk")
    st.caption(
        "A free Streamlit demo of the same trained model used by the EduPulse "
        "FastAPI + React app. Enter a student's details below to see the "
        "predicted outcome, risk band, and the top factors behind it."
    )

    artifact, load_err = load_artifact()
    if load_err:
        st.error(f"Could not load the model: {load_err}")
        st.stop()

    submitted, values = render_form()
    if not submitted:
        return

    try:
        raw_row = {
            raw_col: values[api_field] for api_field, raw_col in API_FIELD_TO_RAW_COLUMN.items()
        }
        raw_row_df = pd.DataFrame([raw_row])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not prepare the input for the model: {exc}")
        return

    try:
        result = run_prediction(artifact, raw_row_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    st.divider()
    st.subheader("Result")

    risk_color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}.get(result["risk_band"], "⚪")
    col_a, col_b = st.columns(2)
    col_a.metric("Predicted Outcome", result["prediction"])
    col_b.metric("Dropout Risk Band", f"{risk_color} {result['risk_band']}")

    st.write("**Class probabilities**")
    proba_df = pd.DataFrame(
        {"Outcome": list(result["probabilities"].keys()), "Probability": list(result["probabilities"].values())}
    ).set_index("Outcome")
    st.bar_chart(proba_df)

    st.write("**Top factors behind this prediction**")
    try:
        clf_type = type(artifact["pipeline"].named_steps["clf"]).__name__
        background = None
        if clf_type == "LogisticRegression":
            background, bg_err = load_shap_background()
            if bg_err:
                st.warning(f"Explanation unavailable: {bg_err}")
                background = None
        if background is not None or clf_type != "LogisticRegression":
            explanation = explain_instance(artifact, raw_row_df, top_n=5, background_data=background)
            factors_df = pd.DataFrame(explanation["top_factors"])
            st.dataframe(factors_df, width="stretch", hide_index=True)
            st.caption(explanation["note"])
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not compute an explanation for this prediction: {exc}")


if __name__ == "__main__":
    main()
