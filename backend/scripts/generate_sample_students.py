"""
Generates a sample "student roster" for the Dashboard/Students pages.

Why precompute this instead of predicting live on every dashboard load:
the Dashboard needs to show hundreds of students at once (brief 4.3:
"table of highest-risk students", "outcome distribution", etc.) — running
the model hundreds of times on every page load would be wasteful and slow.
So we predict once here, save the results, and the API just reads this file.

This does NOT replace /predict (which still does live, on-demand inference
for a single student — the brief's required "New Prediction" flow). This
is sample/demo data standing in for what would be a real student database
in production (brief 6.1 storage note: "CSV-only is acceptable for the
demo if endpoints are clean").
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = REPO_ROOT / "ml" / "src"
sys.path.insert(0, str(ML_SRC))

from data import load_xy, RANDOM_SEED
from features import CLASS_LABELS, ALL_FEATURES, RAW_COLUMN_TO_API_FIELD

ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "model_pipeline.joblib"
OUTPUT_PATH = REPO_ROOT / "backend" / "app" / "data" / "students_sample.json"

SAMPLE_SIZE = 300


def compute_risk_band(dropout_probability: float) -> str:
    if dropout_probability >= 0.60:
        return "High"
    elif dropout_probability >= 0.30:
        return "Medium"
    else:
        return "Low"


def main():
    artifact = joblib.load(ARTIFACT_PATH)
    pipeline = artifact["pipeline"]
    is_xgb = artifact["is_xgb"]
    label_encoder = artifact["label_encoder"]

    X, y = load_xy()
    sample = X.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    if is_xgb:
        pred_encoded = pipeline.predict(sample)
        predictions = label_encoder.inverse_transform(pred_encoded)
        proba = pipeline.predict_proba(sample)
        proba_columns = label_encoder.inverse_transform(range(proba.shape[1]))
    else:
        predictions = pipeline.predict(sample)
        proba = pipeline.predict_proba(sample)
        proba_columns = pipeline.named_steps["clf"].classes_

    col_index = {c: i for i, c in enumerate(proba_columns)}

    students = []
    for i in range(len(sample)):
        probs = {
            label: round(float(proba[i, col_index[label]]), 4)
            for label in CLASS_LABELS
        }
        dropout_prob = probs.get("Dropout", 0.0)

        # Store raw feature values under clean API field names, so this
        # roster's shape matches what /predict and /explain expect if the
        # frontend ever wants to re-run a "what-if" on a stored student.
        raw_row = sample.iloc[i]
        features_api_shape = {
            RAW_COLUMN_TO_API_FIELD[col]: (
                float(raw_row[col]) if isinstance(raw_row[col], (int, float)) else raw_row[col]
            )
            for col in ALL_FEATURES
        }

        students.append({
            "student_id": f"S{1000 + i}",
            "course": int(raw_row["Course"]),
            "age_at_enrollment": int(raw_row["Age at enrollment"]),
            "prediction": str(predictions[i]),
            "risk_band": compute_risk_band(dropout_prob),
            "probabilities": probs,
            "features": features_api_shape,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(students, f, indent=2)

    print(f"Saved {len(students)} sample students to {OUTPUT_PATH}")
    # Quick sanity summary
    from collections import Counter
    print("Prediction distribution:", Counter(s["prediction"] for s in students))
    print("Risk band distribution:", Counter(s["risk_band"] for s in students))


if __name__ == "__main__":
    main()
