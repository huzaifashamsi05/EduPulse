"""
Data loading and splitting for the EduPulse student-outcome model.

Leakage-safety notes (per project brief section 3.3 / 5.1):
  - We split BEFORE any preprocessing is fit. No scaler/encoder ever sees
    validation or test rows during `fit`.
  - Stratified splitting is used throughout because the target classes are
    imbalanced (Graduate 50%, Dropout 32%, Enrolled 18%).
  - A fixed random seed is used and recorded for reproducibility.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from features import ALL_FEATURES, TARGET_COL

RANDOM_SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "student_dropout_raw.csv"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_xy():
    df = load_raw()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y


def split_train_val_test(X, y, test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """
    70/15/15 stratified split.
    Test set is held out untouched until final model selection (brief 3.3).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # val_size is a fraction of the ORIGINAL data; rescale relative to train_val
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val, stratify=y_train_val, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X, y = load_xy()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    for name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"\n{name} class distribution:")
        print(split_y.value_counts(normalize=True).round(3))
