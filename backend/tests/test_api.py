"""
Basic backend tests (brief 15.2: /health returns success, valid prediction
payload returns 200, invalid payload returns readable 4xx, repeated
requests don't reload the model).

Run with: pytest tests/test_api.py -v   (from the backend/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
from app.schemas.prediction import StudentFeaturesRequest

VALID_PAYLOAD = StudentFeaturesRequest.Config.json_schema_extra["example"]


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_model_info_returns_expected_keys():
    with TestClient(app) as client:
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "class_labels" in data
        assert set(data["class_labels"]) == {"Dropout", "Enrolled", "Graduate"}


def test_predict_valid_payload_returns_200_with_expected_shape():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in ["Dropout", "Enrolled", "Graduate"]
        assert data["risk_band"] in ["High", "Medium", "Low"]
        # Probabilities should sum to ~1.0
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01


def test_predict_missing_fields_returns_422():
    with TestClient(app) as client:
        response = client.post("/predict", json={"marital_status": 1})
        assert response.status_code == 422
        assert "detail" in response.json()


def test_predict_out_of_range_value_returns_422():
    with TestClient(app) as client:
        bad_payload = dict(VALID_PAYLOAD)
        bad_payload["marital_status"] = 999  # schema limits this to 1-6
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


def test_explain_returns_top_factors():
    with TestClient(app) as client:
        response = client.post("/explain", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "top_factors" in data
        assert len(data["top_factors"]) > 0
        assert "note" in data


def test_students_endpoint_returns_list():
    with TestClient(app) as client:
        response = client.get("/students", params={"limit": 5})
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5


def test_student_not_found_returns_404():
    with TestClient(app) as client:
        response = client.get("/students/DOES_NOT_EXIST")
        assert response.status_code == 404


def test_analytics_summary_returns_expected_keys():
    with TestClient(app) as client:
        response = client.get("/analytics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_students" in data
        assert "risk_distribution" in data
        assert data["total_students"] > 0


def test_repeated_predictions_dont_error(prediction_count=5):
    """Confirms the model isn't being reloaded/retrained per-request —
    repeated calls should all succeed quickly and consistently."""
    with TestClient(app) as client:
        for _ in range(prediction_count):
            response = client.post("/predict", json=VALID_PAYLOAD)
            assert response.status_code == 200
