"""
Preprocessing pipeline for the EduPulse student-outcome model.

Per brief 6.2: this builds ONE fitted ColumnTransformer that gets serialized
inside the final model Pipeline. The API loads that single artifact and never
re-fits or duplicates this logic — eliminating train/serve skew.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, BINARY_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Binary features are already 0/1 -> pass through unchanged.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    """Human-readable-ish names for each column AFTER transformation, used for SHAP/explainability."""
    return list(preprocessor.get_feature_names_out())
