"""
Loads the trained model artifact ONCE and keeps it in memory.

Why this file exists on its own (brief 7.3: "Load the model once, not on
every request"): if we called joblib.load() inside the /predict route
itself, every single API request would re-read a multi-megabyte file from
disk and rebuild the model in memory — slow, and conceptually wrong for a
"serving" API. Instead, main.py calls load_model() exactly once when the
server process starts, and every route just reads the already-loaded object
from here.
"""
import sys
import json
from pathlib import Path

import joblib

# Path from backend/app/core/model_loader.py up to the repo root, then into ml/
REPO_ROOT = Path(__file__).resolve().parents[3]
ML_SRC_DIR = REPO_ROOT / "ml" / "src"
ARTIFACT_PATH = REPO_ROOT / "ml" / "artifacts" / "model_pipeline.joblib"
REPORT_PATH = REPO_ROOT / "ml" / "reports" / "model_comparison_report.json"

# Make ml/src importable (features.py, explain.py) so the backend reuses the
# EXACT SAME feature definitions as training — never a second copy that could
# drift out of sync.
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

# Module-level singleton — populated by load_model(), read by every route.
_state = {
    "artifact": None,
    "report": None,
    "shap_background": None,
}


def load_model():
    """Load the model artifact and its training report into memory.
    Called once from main.py's startup event."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            f"Run ml/src/train.py first to generate it."
        )
    _state["artifact"] = joblib.load(ARTIFACT_PATH)

    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            _state["report"] = json.load(f)
    else:
        _state["report"] = None

    # Only Logistic Regression's SHAP explainer (LinearExplainer) needs a
    # background sample; tree models don't. We prepare it here — once, at
    # startup — rather than inside the /explain route, for the same reason
    # the model itself is loaded once: it's relatively expensive to build
    # and never changes between requests.
    clf_type = type(_state["artifact"]["pipeline"].named_steps["clf"]).__name__
    if clf_type == "LogisticRegression":
        from data import load_xy
        preprocessor = _state["artifact"]["pipeline"].named_steps["preprocess"]
        X, _ = load_xy()
        sample = X.sample(100, random_state=42)
        _state["shap_background"] = preprocessor.transform(sample)

    return _state["artifact"]


def get_artifact():
    """Read accessor for routes — raises a clear error if called before startup."""
    if _state["artifact"] is None:
        raise RuntimeError("Model not loaded yet. load_model() must run at app startup.")
    return _state["artifact"]


def get_report():
    return _state["report"]


def get_shap_background():
    """Returns the precomputed SHAP background sample, or None if the loaded
    model doesn't need one (tree-based models don't)."""
    return _state["shap_background"]
