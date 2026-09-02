"""
Explain service: runs SHAP on a validated request and returns the
brief's exact top_factors contract shape (section 7.2), reusing the SAME
explain_instance() function from ml/src/explain.py — not a reimplementation.
This is the same "one source of truth" principle as preprocessing.
"""
from app.core.model_loader import get_artifact, get_shap_background
from explain import explain_instance, global_feature_importance  # from ml/src


def run_explanation(request, top_n: int = 5):
    """
    request: a validated StudentFeaturesRequest.
    Returns: {"top_factors": [...], "note": "..."} matching brief 7.2's
    top_factors array shape.
    """
    artifact = get_artifact()
    background = get_shap_background()  # None for tree models, required for LogisticRegression

    X_row = request.to_raw_dataframe_row()
    result = explain_instance(artifact, X_row, top_n=top_n, background_data=background)

    return {
        "top_factors": result["top_factors"],
        "note": result["note"],
    }


def get_global_importance(top_n: int = 12):
    """Global feature importance for the Model Insights page (brief 5.5:
    'Also provide global feature importance or another global model-insight
    view'). Computed once from the fitted model's coefficients/importances —
    cheap enough to compute on-demand, doesn't need caching."""
    artifact = get_artifact()
    return global_feature_importance(artifact, top_n=top_n)
