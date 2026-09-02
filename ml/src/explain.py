"""
Explainability for the EduPulse model (brief 5.5).

Uses SHAP TreeExplainer on the fitted XGBoost estimator. Because the
preprocessing pipeline one-hot encodes categorical features, a single
original feature (e.g. "Course") can map to many encoded dummy columns
(e.g. "cat__Course_33", "cat__Course_9254", ...). We aggregate SHAP
contributions back to the ORIGINAL feature so the UI can show human
-readable reasons instead of one-hot dummy names.

Important framing (brief 5.5): these are model contribution values, not
causal claims. Every consumer of this module must surface that caveat.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import shap

from features import human_label, CLASS_LABELS

NON_CAUSAL_NOTE = (
    "Model explanations describe model behavior; they do not prove causal relationships."
)


def _origin_feature_name(encoded_name: str) -> str:
    """
    Map an encoded column name back to its original raw feature name.
    e.g. 'num__Admission grade' -> 'Admission grade'
         'cat__Course_9254'     -> 'Course'
         'bin__Debtor'          -> 'Debtor'
    """
    if "__" not in encoded_name:
        return encoded_name
    prefix, rest = encoded_name.split("__", 1)
    if prefix == "cat":
        # OneHotEncoder names as "<original_col>_<category_value>"
        # original col names may themselves contain underscores/spaces/parens,
        # so we match against known categorical columns instead of blind splitting.
        from features import CATEGORICAL_FEATURES
        for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            if rest.startswith(col + "_"):
                return col
        return rest
    return rest


def _build_explainer(clf, background_data=None):
    """
    Pick the right SHAP explainer for whichever model family won model
    selection. Model selection is CV-score-based (train.py) and can vary
    slightly across machines/library versions when scores are near-tied
    (e.g. Logistic Regression vs XGBoost differing by ~0.0002 f1_macro) —
    so explain.py must not assume the winner is always tree-based.
    """
    model_type = type(clf).__name__
    if model_type in ("XGBClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(clf), "tree"
    elif model_type == "LogisticRegression":
        # LinearExplainer needs a background dataset to estimate the
        # expected value it measures contributions against.
        if background_data is None:
            raise ValueError("LogisticRegression explainer needs background_data (a sample of transformed training rows).")
        return shap.LinearExplainer(clf, background_data), "linear"
    else:
        raise ValueError(f"No SHAP explainer configured for model type: {model_type}")


def explain_instance(artifact, X_row: pd.DataFrame, top_n: int = 5, background_data=None):
    """
    Returns top_n aggregated feature contributions for ONE row's predicted class.
    X_row: single-row DataFrame with the raw (untransformed) feature columns.
    background_data: required only when the underlying model is Logistic
        Regression (a small sample of TRANSFORMED training rows, e.g. 100 rows).
    """
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(X_row)
    encoded_names = list(preprocessor.get_feature_names_out())

    explainer, explainer_kind = _build_explainer(clf, background_data)
    raw_shap = explainer.shap_values(X_transformed)

    # xgboost/RF multiclass via TreeExplainer returns array shape
    # (n_samples, n_features, n_classes) in recent shap versions; handle
    # both that and list-of-arrays (older versions). LinearExplainer on a
    # multiclass one-vs-rest LogisticRegression returns a list of arrays,
    # one per class, each (n_samples, n_features).
    #
    # Note: XGBoost was trained on label-ENCODED targets (0/1/2), so
    # clf.predict() returns an integer directly usable as an index.
    # Logistic Regression / Random Forest were trained on the original
    # STRING labels, so clf.predict() returns e.g. "Dropout" and we must
    # look up its position via clf.classes_ instead.
    raw_pred = clf.predict(X_transformed)[0]
    if isinstance(raw_pred, (int, np.integer)):
        pred_class_idx = int(raw_pred)
    else:
        pred_class_idx = list(clf.classes_).index(raw_pred)

    if isinstance(raw_shap, list):
        class_shap = raw_shap[pred_class_idx][0]  # (n_features,)
    elif raw_shap.ndim == 3:
        class_shap = raw_shap[0, :, pred_class_idx]
    else:
        class_shap = raw_shap[0]

    # Aggregate by original feature name
    agg = defaultdict(float)
    for name, val in zip(encoded_names, class_shap):
        origin = _origin_feature_name(name)
        agg[origin] += float(val)

    # Rank by absolute impact
    ranked = sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    predicted_class_name = CLASS_LABELS[pred_class_idx]
    factors = [
        {
            "feature": human_label(feat),
            "impact": round(abs(val), 4),
            # Positive SHAP value = pushed the prediction TOWARD the predicted
            # class; negative = pushed AWAY from it (toward the other classes).
            "direction": (
                f"toward_{predicted_class_name.lower()}" if val > 0
                else f"away_from_{predicted_class_name.lower()}"
            ),
        }
        for feat, val in ranked
    ]
    return {
        "predicted_class": CLASS_LABELS[pred_class_idx],
        "top_factors": factors,
        "note": NON_CAUSAL_NOTE,
    }


def global_feature_importance(artifact, top_n: int = 12):
    """Global feature importance, aggregated to original feature names for
    the Model Insights page. Tree models expose gain-based `feature_importances_`;
    Logistic Regression exposes per-class `coef_` instead, so we average the
    absolute coefficient magnitude across classes as a comparable importance score."""
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]

    encoded_names = list(preprocessor.get_feature_names_out())

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        # coef_ shape is (n_classes, n_features) for multinomial LogisticRegression
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        raise ValueError(f"Model type {type(clf).__name__} has no supported importance attribute.")

    agg = defaultdict(float)
    for name, val in zip(encoded_names, importances):
        origin = _origin_feature_name(name)
        agg[origin] += float(val)

    ranked = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    total = sum(agg.values()) or 1.0
    return [
        {"feature": human_label(feat), "importance": round(val / total, 4)}
        for feat, val in ranked
    ]


if __name__ == "__main__":
    import joblib
    from pathlib import Path
    from data import load_xy

    artifact = joblib.load(Path(__file__).resolve().parents[1] / "artifacts" / "model_pipeline.joblib")
    X, y = load_xy()
    sample = X.iloc[[0]]

    clf_type = type(artifact["pipeline"].named_steps["clf"]).__name__
    print(f"Model in this artifact: {clf_type}")

    background = None
    if clf_type == "LogisticRegression":
        # LinearExplainer needs a small transformed background sample.
        preprocessor = artifact["pipeline"].named_steps["preprocess"]
        background = preprocessor.transform(X.sample(100, random_state=42))

    print("=== Local explanation for row 0 ===")
    result = explain_instance(artifact, sample, background_data=background)
    print(result)

    print("\n=== Global feature importance ===")
    for item in global_feature_importance(artifact):
        print(item)