"""
Pydantic schemas for the /predict endpoint.

Why this matters: FastAPI validates every incoming request against
StudentFeaturesRequest BEFORE our code ever runs. If the frontend sends a
string where a number was expected, or leaves a required field out, the
person gets a clear 422 error automatically — we don't have to write that
validation by hand (brief 7.3: "Validate payload types and ranges with
Pydantic schemas").
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_SRC_DIR = REPO_ROOT / "ml" / "src"
if str(ML_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SRC_DIR))

from pydantic import BaseModel, Field
from features import API_FIELD_TO_RAW_COLUMN


class StudentFeaturesRequest(BaseModel):
    """One student's raw feature values, using clean snake_case field names.
    Every field is required — a real prediction needs the full feature set,
    the same way the model saw complete rows during training."""

    marital_status: int = Field(..., ge=1, le=6, description="1=Single, 2=Married, etc. (UCI code)")
    application_mode: int = Field(..., description="UCI application mode code")
    application_order: int = Field(..., ge=0, description="Order of preference for this course, 0=first choice")
    course: int = Field(..., description="UCI course code")
    nationality: int = Field(..., description="UCI nationality code")
    mothers_qualification: int = Field(..., description="UCI qualification code")
    fathers_qualification: int = Field(..., description="UCI qualification code")
    mothers_occupation: int = Field(..., description="UCI occupation code")
    fathers_occupation: int = Field(..., description="UCI occupation code")
    previous_qualification: int = Field(..., description="UCI previous qualification type code")

    daytime_evening_attendance: int = Field(..., ge=0, le=1)
    displaced: int = Field(..., ge=0, le=1)
    educational_special_needs: int = Field(..., ge=0, le=1)
    debtor: int = Field(..., ge=0, le=1)
    tuition_fees_up_to_date: int = Field(..., ge=0, le=1)
    gender: int = Field(..., ge=0, le=1)
    scholarship_holder: int = Field(..., ge=0, le=1)
    international: int = Field(..., ge=0, le=1)

    previous_qualification_grade: float = Field(..., ge=0, le=200)
    admission_grade: float = Field(..., ge=0, le=200)
    age_at_enrollment: int = Field(..., ge=15, le=100)

    curricular_units_1st_sem_credited: float = Field(..., ge=0)
    curricular_units_1st_sem_enrolled: float = Field(..., ge=0)
    curricular_units_1st_sem_evaluations: float = Field(..., ge=0)
    curricular_units_1st_sem_approved: float = Field(..., ge=0)
    curricular_units_1st_sem_grade: float = Field(..., ge=0, le=20)
    curricular_units_1st_sem_without_evaluations: float = Field(..., ge=0)

    curricular_units_2nd_sem_credited: float = Field(..., ge=0)
    curricular_units_2nd_sem_enrolled: float = Field(..., ge=0)
    curricular_units_2nd_sem_evaluations: float = Field(..., ge=0)
    curricular_units_2nd_sem_approved: float = Field(..., ge=0)
    curricular_units_2nd_sem_grade: float = Field(..., ge=0, le=20)
    curricular_units_2nd_sem_without_evaluations: float = Field(..., ge=0)

    unemployment_rate: float
    inflation_rate: float
    gdp: float

    def to_raw_dataframe_row(self):
        """Convert this validated request into a single-row pandas DataFrame
        with RAW dataset column names — the exact shape the trained pipeline
        expects. This is the one function that bridges 'API language' and
        'model language'."""
        import pandas as pd
        data = self.model_dump()
        raw_row = {
            raw_col: data[api_field]
            for api_field, raw_col in API_FIELD_TO_RAW_COLUMN.items()
        }
        return pd.DataFrame([raw_row])

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Matches the brief's API contract (section 7.2)."""
    prediction: str
    risk_band: str
    probabilities: dict
    model_version: str
