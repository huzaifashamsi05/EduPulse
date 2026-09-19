"""
EduPulse — Streamlit demo

A free-to-host, luxury-grade demo of the EduPulse student dropout-risk
model. This reuses the SAME trained pipeline and the SAME feature/explain
code as the FastAPI + React app (ml/src/*) — it is not a re-implementation
of the model, just a richer, more polished front door onto it. The original
FastAPI + React app (backend/, frontend/) is left untouched.

Why this file exists: FastAPI + React needs a host that can run a
long-lived server process, and every free tier tried for that required
adding a card for verification. Streamlit Community Cloud hosts this file
directly from GitHub for free, no card, the same way the other two ML
projects in this account are already deployed.

This version adds: a dark luxury theme, a sidebar-driven three-page layout
(Predict / Model Insights / About & Responsible Use), one-click example
presets, a risk gauge, richer Plotly visuals for probabilities and SHAP
factors, a live model-comparison dashboard sourced straight from
ml/reports/model_comparison_report.json, and an in-session prediction
history — without changing a single line of the underlying ML logic.
"""
import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make ml/src importable so this app reuses the EXACT SAME feature
# definitions and SHAP logic as training/serving — never a second copy
# that could drift out of sync.
REPO_ROOT = Path(__file__).resolve().parent
ML_SRC_DIR = REPO_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from features import API_FIELD_TO_RAW_COLUMN, human_label  # noqa: E402
from explain import explain_instance, global_feature_importance  # noqa: E402

ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "model_pipeline.joblib"
REPORT_PATH = REPO_ROOT / "ml" / "reports" / "model_comparison_report.json"

st.set_page_config(
    page_title="EduPulse — Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# LUXURY THEME
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg-0: #05070d;
    --bg-1: #0a0f1c;
    --accent-1: #a78bfa;
    --accent-2: #38bdf8;
    --gold: #f0c975;
    --danger: #ef4444;
    --warn: #f59e0b;
    --ok: #22c55e;
    --text-hi: #f2f6ff;
    --text-lo: #97a2c2;
    --border: rgba(148, 163, 184, 0.14);
}

html, body { font-family: 'Inter', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Sora', 'Segoe UI Emoji', 'Noto Color Emoji', 'Apple Color Emoji', sans-serif; }

/* Never let the custom font stack override Streamlit's own ligature icon fonts
   (this is what makes icons like "upload" / "arrow_right" render as literal text) */
[data-testid="stIconMaterial"],
[class*="material-symbols"],
span[data-icon],
[data-testid="stFileUploaderDropzoneIcon"] svg,
[data-testid*="Icon"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(167,139,250,0.12) 0%, transparent 42%),
        radial-gradient(circle at 92% 8%, rgba(56,189,248,0.10) 0%, transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(240,201,117,0.06) 0%, transparent 45%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 55%, var(--bg-0) 100%);
}
#MainMenu, footer, header { visibility: hidden; }

.hero-wrap {
    padding: 2.6rem 2.4rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(56,189,248,0.08));
    border: 1px solid var(--border);
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    display: inline-block; font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--gold); font-weight: 700;
    background: rgba(240,201,117,0.12); border: 1px solid rgba(240,201,117,0.3);
    padding: 0.32rem 0.85rem; border-radius: 999px; margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.03em;
    color: var(--text-hi); margin: 0 0 0.55rem 0; line-height: 1.15;
}
.hero-title span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { color: var(--text-lo); font-size: 1.02rem; max-width: 720px; margin: 0; }
.hero-badges { margin-top: 1.2rem; display: flex; gap: 0.6rem; flex-wrap: wrap; }
.hero-badge {
    font-size: 0.78rem; color: var(--text-hi);
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    padding: 0.38rem 0.85rem; border-radius: 999px;
}
.hero-badge.gold { border-color: rgba(240,201,117,0.4); color: var(--gold); }

.section-label {
    font-size: 0.75rem; letter-spacing: 0.13em; text-transform: uppercase;
    color: var(--accent-2); font-weight: 700; margin: 1.3rem 0 0.4rem 0;
}
.section-title { font-size: 1.35rem; font-weight: 700; color: var(--text-hi); margin: 0 0 1rem 0; }

section[data-testid="stSidebar"] { background: var(--bg-0); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-hi) !important; font-weight: 600; }
.sidebar-brand {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.3rem;
    color: var(--text-hi); padding: 0.4rem 0 1rem 0; letter-spacing: -0.02em;
}
.sidebar-brand span { color: var(--accent-1); }

/* Sidebar nav - card style radio */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.65rem; display: flex; flex-direction: column; margin-top: 0.2rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1rem 0.9rem 1.2rem;
    margin: 0 !important;
    cursor: pointer;
    box-shadow: 0 2px 10px -4px rgba(0,0,0,0.4);
    transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    border-color: rgba(167,139,250,0.45);
    transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    border-color: transparent;
    box-shadow: 0 10px 26px -8px rgba(167,139,250,0.55);
    transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
    color: #0b0620 !important; font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    color: var(--text-hi); font-size: 0.93rem; font-weight: 600; letter-spacing: -0.01em;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
section[data-testid="stSidebar"] .stRadio > label { display: none; }

.stButton>button {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    color: #0b0620; font-weight: 700; border: none; border-radius: 10px;
    padding: 0.6rem 1.6rem;
    box-shadow: 0 8px 24px -8px rgba(167,139,250,0.5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 10px 28px -6px rgba(167,139,250,0.65); }
button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important; color: var(--text-hi) !important;
    border: 1px solid var(--border) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
    border: 1px solid var(--border); border-radius: 16px; padding: 1.1rem 1.2rem;
}
[data-testid="stMetricLabel"] { color: var(--text-lo) !important; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; font-family: 'Sora', sans-serif; }

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
[data-testid="stExpander"] {
    border: 1px solid var(--border); border-radius: 16px; background: rgba(255,255,255,0.02);
    margin-bottom: 0.8rem;
}
[data-testid="stExpander"] summary { font-weight: 700; color: var(--text-hi); }

p, .stMarkdown, label { color: var(--text-lo); }
h1, h2, h3 { color: var(--text-hi); }

.result-card {
    border-radius: 20px; padding: 1.8rem 2rem; border: 1px solid var(--border);
    margin: 0.6rem 0 1.2rem 0;
}
.result-card.high { background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(239,68,68,0.03)); border-color: rgba(239,68,68,0.35); }
.result-card.medium { background: linear-gradient(135deg, rgba(245,158,11,0.16), rgba(245,158,11,0.03)); border-color: rgba(245,158,11,0.35); }
.result-card.low { background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(34,197,94,0.03)); border-color: rgba(34,197,94,0.35); }
.result-headline { font-family: 'Sora', sans-serif; font-size: 1.9rem; font-weight: 800; color: var(--text-hi); margin: 0; }
.result-band {
    display: inline-block; font-weight: 700; font-size: 0.85rem;
    padding: 0.35rem 0.9rem; border-radius: 999px; margin-top: 0.5rem;
}
.result-band.high { background: rgba(239,68,68,0.2); color: #fca5a5; }
.result-band.medium { background: rgba(245,158,11,0.2); color: #fcd34d; }
.result-band.low { background: rgba(34,197,94,0.2); color: #86efac; }
.result-note {
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.08);
    color: var(--text-lo); font-size: 0.88rem;
}

.preset-strip { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1rem; }

.footer-note {
    margin-top: 2.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border);
    color: var(--text-lo); font-size: 0.82rem; text-align: center;
}
.footer-note a { color: var(--accent-2); text-decoration: none; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c9d4ea"),
    margin=dict(l=10, r=10, t=45, b=10),
)


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


@st.cache_data
def load_report():
    """Load the model comparison report generated by ml/src/train.py. This
    powers the Model Insights page and the hero badges — never hardcoded."""
    try:
        with open(REPORT_PATH) as f:
            return json.load(f), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Model report not found or unreadable: {exc}"


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

# One-click example presets. "Strong Performer" nudges the defaults toward a
# clearly healthy profile; "At-Risk Profile" is the exact scenario called out
# in the README's manual integration test (2nd-sem approved units -> 0
# flips the dropout probability from ~19% to ~93%), so the demo's headline
# claim is always reproducible with one click, not just in a screenshot.
PRESET_STRONG_PERFORMER = dict(DEFAULTS)
PRESET_STRONG_PERFORMER.update({
    "admission_grade": 152.0,
    "curricular_units_1st_sem_approved": 6,
    "curricular_units_1st_sem_grade": 15.5,
    "curricular_units_2nd_sem_approved": 6,
    "curricular_units_2nd_sem_grade": 15.8,
    "tuition_fees_up_to_date": 1,
    "debtor": 0,
    "scholarship_holder": 1,
})

PRESET_AT_RISK = dict(DEFAULTS)
PRESET_AT_RISK.update({
    "admission_grade": 96.0,
    "curricular_units_1st_sem_approved": 1,
    "curricular_units_1st_sem_grade": 8.5,
    "curricular_units_2nd_sem_approved": 0,
    "curricular_units_2nd_sem_grade": 0.0,
    "tuition_fees_up_to_date": 0,
    "debtor": 1,
    "scholarship_holder": 0,
})

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

INT_FIELDS = {
    "marital_status", "application_mode", "application_order", "course", "nationality",
    "mothers_qualification", "fathers_qualification", "mothers_occupation", "fathers_occupation",
    "previous_qualification", "age_at_enrollment",
}

FIELD_KEY_PREFIX = "epf_"


def field_key(api_field: str) -> str:
    return FIELD_KEY_PREFIX + api_field


def apply_preset(preset: dict):
    """Push preset values straight into widget session_state BEFORE the
    widgets are instantiated on the next run — the standard Streamlit
    pattern for programmatically setting form defaults after first load."""
    for api_field, value in preset.items():
        key = field_key(api_field)
        if api_field in INT_FIELDS or api_field in BINARY_LABELS:
            st.session_state[key] = int(value)
        else:
            st.session_state[key] = float(value)
    st.session_state["ep_result"] = None
    st.rerun()


def render_hero(report):
    badges = [
        '<span class="hero-badge gold">🎓 Decision-support, not automation</span>',
        '<span class="hero-badge">🧪 UCI Student Dropout dataset · 4,424 records</span>',
    ]
    if report:
        model_name = report.get("selected_model", "").replace("_", " ").title()
        f1 = report.get("test_set_metrics_final_model", {}).get("f1_macro")
        auc = report.get("test_set_metrics_final_model", {}).get("roc_auc_ovr_macro")
        if model_name:
            badges.insert(0, f'<span class="hero-badge gold">🏆 {model_name}</span>')
        if f1 is not None:
            badges.append(f'<span class="hero-badge">📈 Macro-F1 {f1:.2f}</span>')
        if auc is not None:
            badges.append(f'<span class="hero-badge">🎯 ROC-AUC {auc:.2f}</span>')
    st.markdown(f"""
    <div class="hero-wrap">
        <span class="hero-eyebrow">Machine Learning · Early Intervention Platform</span>
        <div class="hero-title">🎓 EduPulse <span>Student Success Intelligence</span></div>
        <p class="hero-sub">Predicts a student's likely academic outcome — Dropout, Enrolled, or
        Graduate — explains why the model reached that call with SHAP, and gives an academic
        advisor the context to reach out sooner. This demo runs the exact same trained pipeline
        as the full FastAPI + React app.</p>
        <div class="hero-badges">{''.join(badges)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_nav():
    st.sidebar.markdown('<div class="sidebar-brand">Edu<span>Pulse</span></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<p class="section-label" style="margin-top:0;">Navigate</p>', unsafe_allow_html=True)
    choice = st.sidebar.radio(
        "Navigate",
        ["🎯  Predict a Student", "📊  Model Insights", "ℹ️  About & Responsible Use"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "EduPulse is a decision-support prototype for human academic advisors. "
        "It must never be used to automatically remove, penalize, rank, or "
        "discipline students."
    )
    st.sidebar.markdown(
        '<p style="font-size:0.78rem;">'
        '<a href="https://github.com/huzaifashamsi05/EduPulse" target="_blank">📦 Full source on GitHub</a><br>'
        '<a href="https://portfolio-showcase-api-server-gilt.vercel.app" target="_blank">🧑‍💻 Developer portfolio</a>'
        '</p>',
        unsafe_allow_html=True,
    )
    if "Predict" in choice:
        return "predict"
    elif "Insights" in choice:
        return "insights"
    return "about"


def render_presets_bar():
    st.markdown('<p class="section-label" style="margin-top:0;">Quick Start</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title" style="margin-bottom:0.6rem;">Try an example, or fill in your own student below</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🟢 Load Strong-Performer Example", width="stretch"):
            apply_preset(PRESET_STRONG_PERFORMER)
    with c2:
        if st.button("🔴 Load At-Risk Example", width="stretch"):
            apply_preset(PRESET_AT_RISK)
    with c3:
        if st.button("↺ Reset to Defaults", width="stretch", type="secondary"):
            apply_preset(DEFAULTS)


def render_form():
    """One st.form covering all 36 model inputs, grouped into expandable
    luxury cards so the page doesn't read as one undifferentiated wall of
    fields."""
    values = {}
    with st.form("student_features"):
        with st.expander("👤 Background & Application", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                values["marital_status"] = st.number_input(
                    "Marital Status (UCI code)", min_value=1, max_value=6,
                    value=DEFAULTS["marital_status"], key=field_key("marital_status"),
                )
                values["application_mode"] = st.number_input(
                    "Application Mode (UCI code)", min_value=0,
                    value=DEFAULTS["application_mode"], key=field_key("application_mode"),
                )
                values["application_order"] = st.number_input(
                    "Application Preference Order", min_value=0,
                    value=DEFAULTS["application_order"], key=field_key("application_order"),
                )
                values["course"] = st.number_input(
                    "Course (UCI code)", min_value=0,
                    value=DEFAULTS["course"], key=field_key("course"),
                )
                values["nationality"] = st.number_input(
                    "Nationality (UCI code)", min_value=0,
                    value=DEFAULTS["nationality"], key=field_key("nationality"),
                )
                values["previous_qualification"] = st.number_input(
                    "Previous Qualification Type (UCI code)", min_value=0,
                    value=DEFAULTS["previous_qualification"], key=field_key("previous_qualification"),
                )
            with c2:
                values["mothers_qualification"] = st.number_input(
                    "Mother's Education Level (UCI code)", min_value=0,
                    value=DEFAULTS["mothers_qualification"], key=field_key("mothers_qualification"),
                )
                values["fathers_qualification"] = st.number_input(
                    "Father's Education Level (UCI code)", min_value=0,
                    value=DEFAULTS["fathers_qualification"], key=field_key("fathers_qualification"),
                )
                values["mothers_occupation"] = st.number_input(
                    "Mother's Occupation (UCI code)", min_value=0,
                    value=DEFAULTS["mothers_occupation"], key=field_key("mothers_occupation"),
                )
                values["fathers_occupation"] = st.number_input(
                    "Father's Occupation (UCI code)", min_value=0,
                    value=DEFAULTS["fathers_occupation"], key=field_key("fathers_occupation"),
                )
                values["previous_qualification_grade"] = st.number_input(
                    "Previous Qualification Grade (0-200)",
                    min_value=0.0, max_value=200.0, value=DEFAULTS["previous_qualification_grade"],
                    key=field_key("previous_qualification_grade"),
                )
                values["admission_grade"] = st.number_input(
                    "Admission Grade (0-200)", min_value=0.0, max_value=200.0,
                    value=DEFAULTS["admission_grade"], key=field_key("admission_grade"),
                )
                values["age_at_enrollment"] = st.number_input(
                    "Age at Enrollment", min_value=15, max_value=100,
                    value=DEFAULTS["age_at_enrollment"], key=field_key("age_at_enrollment"),
                )

        with st.expander("🚩 Status Flags", expanded=True):
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
                        key=field_key(field),
                    )
                    values[field] = choice

        with st.expander("📘 1st Semester Performance", expanded=True):
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
                    values[field] = st.number_input(
                        label, min_value=lo, max_value=hi, value=float(DEFAULTS[field]), key=field_key(field)
                    )

        with st.expander("📗 2nd Semester Performance", expanded=True):
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
                    values[field] = st.number_input(
                        label, min_value=lo, max_value=hi, value=float(DEFAULTS[field]), key=field_key(field)
                    )

        with st.expander("🌍 Macroeconomic Context", expanded=False):
            m1, m2, m3 = st.columns(3)
            with m1:
                values["unemployment_rate"] = st.number_input(
                    "Regional Unemployment Rate (%)", value=DEFAULTS["unemployment_rate"],
                    key=field_key("unemployment_rate"),
                )
            with m2:
                values["inflation_rate"] = st.number_input(
                    "Regional Inflation Rate (%)", value=DEFAULTS["inflation_rate"],
                    key=field_key("inflation_rate"),
                )
            with m3:
                values["gdp"] = st.number_input(
                    "Regional GDP Growth", value=DEFAULTS["gdp"], key=field_key("gdp"),
                )

        submitted = st.form_submit_button("🔍 Predict Student Outcome", type="primary", width="stretch")
    return submitted, values


def render_gauge(dropout_prob: float, risk_band: str):
    color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}.get(risk_band, "#38bdf8")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=dropout_prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#f2f6ff"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#97a2c2"},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": "rgba(148,163,184,0.2)",
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.18)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.18)"},
                {"range": [60, 100], "color": "rgba(239,68,68,0.18)"},
            ],
        },
        title={"text": "Dropout Probability", "font": {"size": 14, "color": "#97a2c2"}},
    ))
    fig.update_layout(height=260, **PLOTLY_DARK)
    st.plotly_chart(fig, width="stretch")


def render_result(result: dict, artifact, raw_row_df: pd.DataFrame):
    cls = result["risk_band"].lower()
    st.markdown('<p class="section-label">Result</p><p class="section-title">Prediction</p>', unsafe_allow_html=True)

    col_gauge, col_card = st.columns([1, 1.4])
    with col_gauge:
        render_gauge(result["probabilities"].get("Dropout", 0.0), result["risk_band"])
    with col_card:
        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="result-headline">{result['prediction']}</div>
            <span class="result-band {cls}">Dropout Risk: {result['risk_band']}</span>
            <div class="result-note">
                🎯 This is a decision-support signal for a human advisor — never an automated
                decision about the student.
            </div>
        </div>
        """, unsafe_allow_html=True)

        proba_df = pd.DataFrame({
            "Outcome": list(result["probabilities"].keys()),
            "Probability": list(result["probabilities"].values()),
        })
        fig = px.bar(
            proba_df, x="Probability", y="Outcome", orientation="h", text="Probability",
            color="Outcome",
            color_discrete_map={"Dropout": "#ef4444", "Enrolled": "#f59e0b", "Graduate": "#22c55e"},
        )
        fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", height=220, **PLOTLY_DARK)
        st.plotly_chart(fig, width="stretch")

    st.markdown('<p class="section-label">Explainability</p><p class="section-title">Top factors behind this prediction</p>', unsafe_allow_html=True)
    try:
        clf_type = type(artifact["pipeline"].named_steps["clf"]).__name__
        background = None
        if clf_type == "LogisticRegression":
            background, bg_err = load_shap_background()
            if bg_err:
                st.warning(f"Explanation unavailable: {bg_err}")
                background = None
        if background is not None or clf_type != "LogisticRegression":
            explanation = explain_instance(artifact, raw_row_df, top_n=6, background_data=background)
            factors_df = pd.DataFrame(explanation["top_factors"])
            factors_df["signed_impact"] = factors_df.apply(
                lambda r: r["impact"] if str(r["direction"]).startswith("toward") else -r["impact"], axis=1
            )
            factors_df = factors_df.sort_values("signed_impact")
            fig2 = px.bar(
                factors_df, x="signed_impact", y="feature", orientation="h",
                color="signed_impact",
                color_continuous_scale=["#ef4444", "#38363f", "#22c55e"],
                labels={"signed_impact": "SHAP contribution", "feature": ""},
            )
            fig2.update_layout(coloraxis_showscale=False, height=320, **PLOTLY_DARK)
            st.plotly_chart(fig2, width="stretch")
            st.caption(f"ℹ️ {explanation['note']} Bars pointing right push toward the predicted outcome; bars pointing left push away from it.")

            history_entry = {
                "Prediction": result["prediction"],
                "Risk Band": result["risk_band"],
                "Dropout %": f"{result['probabilities'].get('Dropout', 0):.0%}",
                "Top Factor": factors_df.iloc[-1]["feature"] if len(factors_df) else "—",
            }
            st.session_state.setdefault("ep_history", [])
            st.session_state["ep_history"].insert(0, history_entry)
            st.session_state["ep_history"] = st.session_state["ep_history"][:10]

            report_lines = [
                "EduPulse — Student Outcome Prediction Report", "",
                f"Predicted outcome: {result['prediction']}",
                f"Dropout risk band: {result['risk_band']}", "",
                "Class probabilities:",
            ]
            for k, v in result["probabilities"].items():
                report_lines.append(f"  {k}: {v:.1%}")
            report_lines += ["", "Top contributing factors:"]
            for _, row in factors_df.iloc[::-1].iterrows():
                report_lines.append(f"  {row['feature']} — {row['direction'].replace('_', ' ')} (impact {row['impact']:.3f})")
            report_lines += ["", explanation["note"], "", "This is a decision-support tool. It must never automate punitive decisions."]
            st.download_button(
                "⬇️ Download this prediction as a report",
                "\n".join(report_lines).encode("utf-8"),
                "edupulse_prediction_report.txt",
                "text/plain",
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not compute an explanation for this prediction: {exc}")


def render_history():
    history = st.session_state.get("ep_history", [])
    if not history:
        return
    st.markdown('<p class="section-label">Session Log</p><p class="section-title">Recent predictions (this session only)</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
    if st.button("🗑️ Clear history"):
        st.session_state["ep_history"] = []
        st.rerun()


def render_model_insights(report):
    st.markdown('<p class="section-label">Model Insights</p><p class="section-title">How the deployed model was chosen and how it performs</p>', unsafe_allow_html=True)
    if not report:
        st.info("Model comparison report not found in this deployment — insights are unavailable, but predictions still work.")
        return

    final = report.get("test_set_metrics_final_model", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", report.get("selected_model", "—").replace("_", " ").title())
    m2.metric("Test Accuracy", f"{final.get('accuracy', 0):.1%}")
    m3.metric("Macro-F1", f"{final.get('f1_macro', 0):.3f}")
    m4.metric("ROC-AUC (macro)", f"{final.get('roc_auc_ovr_macro', 0):.3f}")

    baseline = report.get("baseline_majority_class", {})
    st.caption(
        f"ℹ️ {report.get('selection_rationale', '')} A majority-class baseline scores "
        f"{baseline.get('accuracy', 0):.0%} accuracy but only {baseline.get('f1_macro', 0):.2f} "
        "macro-F1 — the deployed model substantially improves on it."
    )

    st.markdown('<p class="section-label">Model Comparison</p><p class="section-title">5-fold cross-validation on the training split</p>', unsafe_allow_html=True)
    cv = report.get("cv_comparison", {})
    if cv:
        rows = []
        for name, metrics in cv.items():
            rows.append({
                "Model": name.replace("_", " ").title(),
                "Accuracy": metrics.get("test_accuracy", {}).get("mean"),
                "Macro-F1": metrics.get("test_f1_macro", {}).get("mean"),
                "Precision (macro)": metrics.get("test_precision_macro", {}).get("mean"),
                "Recall (macro)": metrics.get("test_recall_macro", {}).get("mean"),
                "Fit Time (s)": metrics.get("fit_time_sec"),
            })
        cv_df = pd.DataFrame(rows)
        st.dataframe(
            cv_df.style.format({
                "Accuracy": "{:.1%}", "Macro-F1": "{:.3f}",
                "Precision (macro)": "{:.3f}", "Recall (macro)": "{:.3f}", "Fit Time (s)": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        fig = px.bar(
            cv_df, x="Model", y="Macro-F1", color="Model", text="Macro-F1",
            color_discrete_sequence=["#a78bfa", "#38bdf8", "#f0c975"],
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(showlegend=False, height=320, **PLOTLY_DARK)
        st.plotly_chart(fig, width="stretch")

    col_cm, col_fi = st.columns(2)
    with col_cm:
        st.markdown('<p class="section-label">Confusion Matrix</p><p class="section-title">Final model, held-out test set</p>', unsafe_allow_html=True)
        cm = final.get("confusion_matrix")
        labels = report.get("dataset", {}).get("class_labels", ["Dropout", "Enrolled", "Graduate"])
        if cm:
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Purples",
                labels=dict(x="Predicted", y="Actual", color="Count"),
            )
            fig_cm.update_layout(height=340, **PLOTLY_DARK)
            st.plotly_chart(fig_cm, width="stretch")

    with col_fi:
        st.markdown('<p class="section-label">Global Drivers</p><p class="section-title">Top features across all students</p>', unsafe_allow_html=True)
        artifact, err = load_artifact()
        if artifact is not None:
            try:
                importances = global_feature_importance(artifact, top_n=10)
                fi_df = pd.DataFrame(importances)
                fig_fi = px.bar(
                    fi_df[::-1], x="importance", y="feature", orientation="h",
                    color_discrete_sequence=["#38bdf8"],
                    labels={"importance": "Relative importance", "feature": ""},
                )
                fig_fi.update_layout(height=340, **PLOTLY_DARK)
                st.plotly_chart(fig_fi, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not compute global feature importance: {exc}")
        else:
            st.warning(err)

    with st.expander("📄 Per-class precision / recall / F1 (final model, test set)"):
        per_class = final.get("per_class", {})
        rows = []
        for cls_name in report.get("dataset", {}).get("class_labels", []):
            stats = per_class.get(cls_name, {})
            rows.append({
                "Class": cls_name,
                "Precision": stats.get("precision"),
                "Recall": stats.get("recall"),
                "F1-score": stats.get("f1-score"),
                "Support": stats.get("support"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    dataset = report.get("dataset", {})
    st.caption(
        f"ℹ️ Trained on {dataset.get('n_total', '—')} records "
        f"({dataset.get('n_train', '—')} train / {dataset.get('n_val', '—')} val / "
        f"{dataset.get('n_test', '—')} test), model version {report.get('model_version', '—')}."
    )


def render_about(report):
    st.markdown('<p class="section-label">About</p><p class="section-title">Responsible use & limitations</p>', unsafe_allow_html=True)
    st.markdown("""
EduPulse is an AI-powered student success and early-intervention platform. It predicts a
student's likely academic outcome (Dropout / Enrolled / Graduate), explains why the model
reached that prediction, and presents both to an academic advisor — with the explicit
understanding that predictions support human review and must never automate punitive
decisions.
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label" style="margin-top:0;">Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
- **Source:** UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success*
- **DOI:** 10.24432/C5MC89
- **Size:** 4,424 records, 36 features, 3-class target
- **Class balance:** ~50% Graduate, ~32% Dropout, ~18% Enrolled
""")
    with c2:
        st.markdown('<p class="section-label" style="margin-top:0;">System</p>', unsafe_allow_html=True)
        st.markdown("""
- React (Vite) + FastAPI full app; this page is the free-hosted Streamlit demo
- Same serialized preprocessing + model pipeline as production
- SHAP explainer auto-selected by model type (Tree vs. Linear)
- Model selected by **macro-F1**, not raw accuracy, due to class imbalance
""")

    st.markdown('<p class="section-label">Limitations</p>', unsafe_allow_html=True)
    st.markdown("""
- Test accuracy is roughly 74–78% and macro-F1 roughly 0.70 — a useful decision-support
  signal, not a certainty.
- The "Enrolled" outcome is the hardest class to predict (fewest training examples).
- The model was trained on one historical dataset from one country's higher-education
  system; accuracy is not guaranteed to generalize elsewhere.
- Global feature importance can be inflated for high-cardinality categorical features.
- **EduPulse must not be used to automatically remove, penalize, reject, rank, or
  discipline students.** Predictions and explanations describe patterns in historical
  data — they do not prove causation, and should never replace a human conversation
  with the student.
""")

    st.markdown(f"""
    <div class="footer-note">
        Built as a solo capstone project by Muhammad Huzaifa Shamsi ·
        <a href="https://github.com/huzaifashamsi05/EduPulse" target="_blank">Source on GitHub</a> ·
        <a href="https://portfolio-showcase-api-server-gilt.vercel.app" target="_blank">Portfolio</a>
    </div>
    """, unsafe_allow_html=True)


def main():
    report, _report_err = load_report()
    render_hero(report)
    mode = render_sidebar_nav()

    artifact, load_err = load_artifact()
    if load_err:
        st.error(f"Could not load the model: {load_err}")
        st.stop()

    if mode == "predict":
        render_presets_bar()
        submitted, values = render_form()

        if submitted:
            try:
                raw_row = {
                    raw_col: values[api_field] for api_field, raw_col in API_FIELD_TO_RAW_COLUMN.items()
                }
                raw_row_df = pd.DataFrame([raw_row])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not prepare the input for the model: {exc}")
                raw_row_df = None

            if raw_row_df is not None:
                try:
                    result = run_prediction(artifact, raw_row_df)
                    st.session_state["ep_result"] = result
                    st.session_state["ep_result_row"] = raw_row_df
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Prediction failed: {exc}")
                    st.session_state["ep_result"] = None

        if st.session_state.get("ep_result"):
            st.divider()
            render_result(st.session_state["ep_result"], artifact, st.session_state["ep_result_row"])

        st.divider()
        render_history()

    elif mode == "insights":
        render_model_insights(report)

    else:
        render_about(report)


if __name__ == "__main__":
    main()
