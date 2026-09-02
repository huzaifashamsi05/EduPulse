"""
Prediction service: turns a validated StudentFeaturesRequest into a full
prediction response by running it through the already-loaded model pipeline.

Kept separate from the route function in api/predict.py (brief 7.3: "Keep
model code out of route files as much as possible") — the route's job is
just HTTP plumbing; this file's job is the actual prediction logic, and it
could be unit-tested or reused (e.g. by /predict/batch later) without
touching FastAPI at all.
"""
from app.core.model_loader import get_artifact, get_report


def compute_risk_band(dropout_probability: float) -> str:
    """
    Simple, documented thresholds converting Dropout probability into a
    human-facing risk band (brief 3.2: 'the UI may map Dropout probability
    to a risk band while the underlying model remains multi-class').

    These thresholds are a product decision, not a statistically derived
    boundary — worth stating plainly in the README/report rather than
    presenting them as if they were scientifically optimal.
    """
    if dropout_probability >= 0.60:
        return "High"
    elif dropout_probability >= 0.30:
        return "Medium"
    else:
        return "Low"


def run_prediction(request):
    """
    request: a validated StudentFeaturesRequest (already passed Pydantic checks).
    Returns a dict matching the brief's /predict response contract (section 7.2),
    minus top_factors (that's /explain's job, added in the next step).
    """
    artifact = get_artifact()
    report = get_report()
    pipeline = artifact["pipeline"]
    class_labels = artifact["class_labels"]
    is_xgb = artifact["is_xgb"]
    label_encoder = artifact["label_encoder"]

    X_row = request.to_raw_dataframe_row()

    if is_xgb:
        pred_encoded = pipeline.predict(X_row)[0]
        predicted_class = label_encoder.inverse_transform([pred_encoded])[0]
        proba_raw = pipeline.predict_proba(X_row)[0]
        proba_columns = label_encoder.inverse_transform(range(len(proba_raw)))
    else:
        predicted_class = pipeline.predict(X_row)[0]
        proba_raw = pipeline.predict_proba(X_row)[0]
        proba_columns = pipeline.named_steps["clf"].classes_

    # Reorder probabilities into the fixed CLASS_LABELS order so the API
    # response is always {"Dropout": x, "Enrolled": y, "Graduate": z}
    # regardless of what order the underlying sklearn model happens to use.
    col_index = {c: i for i, c in enumerate(proba_columns)}
    probabilities = {
        label: round(float(proba_raw[col_index[label]]), 4)
        for label in class_labels
    }

    dropout_prob = probabilities.get("Dropout", 0.0)
    risk_band = compute_risk_band(dropout_prob)

    model_version = report.get("model_version") if report else "unknown"

    return {
        "prediction": str(predicted_class),
        "risk_band": risk_band,
        "probabilities": probabilities,
        "model_version": model_version,
    }
