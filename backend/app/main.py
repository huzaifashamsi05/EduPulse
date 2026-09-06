"""
EduPulse backend — FastAPI entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000

Then visit:
    http://127.0.0.1:8000/health          (plain JSON health check)
    http://127.0.0.1:8000/docs            (auto-generated interactive API docs)
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.core.model_loader import load_model, get_artifact, get_report
from app.schemas.prediction import StudentFeaturesRequest, PredictionResponse
from app.services.prediction_service import run_prediction
from app.services.explain_service import run_explanation, get_global_importance
from app.services.students_service import get_students, get_student_by_id
from app.services.analytics_service import get_summary

app = FastAPI(
    title="EduPulse API",
    description="Student success prediction and explainability API",
    version="0.1.0",
)

# Allow the local Vite dev server to call this API (brief 7.3: "Enable CORS
# only for required development/deployment origins" — not a wildcard "*").
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https://.*\.app\.github\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Runs exactly once when the server process starts. This is where the
    model artifact gets loaded into memory — see app/core/model_loader.py
    for why this matters."""
    load_model()
    print("Model artifact loaded successfully.")


@app.get("/health")
def health_check():
    """Basic liveness check — confirms the server process is up and responding.
    The frontend/evaluator calls this first to confirm the backend is reachable
    before trying anything else (brief section 7: 'GET /health')."""
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    """Model name, class labels, and evaluation metrics (brief section 7:
    'GET /model/info'). Reads from the already-loaded artifact and the
    saved training report — never retrains or reloads anything here."""
    artifact = get_artifact()
    report = get_report()

    info = {
        "model_name": artifact["model_name"],
        "class_labels": artifact["class_labels"],
        "model_version": report.get("model_version") if report else None,
        "trained_at": report.get("trained_at") if report else None,
    }

    if report:
        info["test_set_metrics"] = report["test_set_metrics_final_model"]
        info["selection_rationale"] = report["selection_rationale"]

    return info


@app.get("/model/insights")
def model_insights():
    """
    Full model comparison data for the Model Insights page (brief 4.2:
    'Model comparison table, confusion matrix image/data, feature
    importance, metric definitions'). Separate from /model/info to keep
    that endpoint lightweight for the dashboard's model summary card.
    """
    report = get_report()
    if not report:
        raise HTTPException(status_code=404, detail="No training report found.")

    return {
        "cv_comparison": report["cv_comparison"],
        "val_comparison": report["val_comparison"],
        "selected_model": report["selected_model"],
        "selection_rationale": report["selection_rationale"],
        "baseline_majority_class": report["baseline_majority_class"],
        "test_set_metrics": report["test_set_metrics_final_model"],
        "global_feature_importance": get_global_importance(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: StudentFeaturesRequest):
    """
    Run one live prediction (brief section 7: 'POST /predict').

    FastAPI validates `request` against StudentFeaturesRequest BEFORE this
    function body even runs — if a field is missing or out of range, the
    caller gets a 422 error automatically and run_prediction() never executes.
    """
    result = run_prediction(request)
    return result


@app.post("/explain")
def explain(request: StudentFeaturesRequest):
    """
    Return local feature contributions for a prediction (brief section 7:
    'POST /explain'). Takes the SAME request shape as /predict — the
    frontend calls both endpoints for the same student to get the full
    Student Detail page (prediction + explanation).
    """
    result = run_explanation(request)
    return result


@app.get("/students")
def list_students(
    risk_band: str = Query(None, description="Filter by risk band: High, Medium, or Low"),
    course: int = Query(None, description="Filter by exact course code"),
    limit: int = Query(None, description="Max number of results"),
):
    """
    List/filter student records with their stored predictions (brief
    section 7: 'GET /students'). Powers the Students page's searchable
    table (brief 4.2).
    """
    return get_students(risk_band=risk_band, course=course, limit=limit)


@app.get("/students/{student_id}")
def student_detail(student_id: str):
    """Retrieve one student record by ID (brief section 7: 'GET /students/{id}').
    Powers the Student Detail page."""
    student = get_student_by_id(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")
    return student


@app.get("/analytics/summary")
def analytics_summary():
    """
    Dashboard aggregates (brief section 7: 'GET /analytics/summary').
    Covers all of brief 4.3's minimum dashboard widgets: student count,
    outcome distribution, high-risk count/percentage, risk-by-course,
    and the highest-risk students table.
    """
    return get_summary()
from fastapi.staticfiles import StaticFiles
from pathlib import Path
_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
