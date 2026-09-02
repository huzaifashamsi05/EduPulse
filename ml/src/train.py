"""
Train and compare model families for the EduPulse student-outcome classifier.

Per brief 5.2/5.3:
  - Compare >= 3 genuinely different model families.
  - Use stratified cross-validation for comparison (not a single train/val split).
  - Select the final model with a written rationale based on class-level
    performance, not "highest accuracy wins".
  - Never touch the test set until the final model is chosen.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data import load_xy, split_train_val_test, RANDOM_SEED
from preprocessing import build_preprocessor
from features import CLASS_LABELS

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
ARTIFACTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CV_FOLDS = 5


def build_candidates():
    """Three genuinely different model families, each as a full Pipeline
    (preprocessing + estimator) so CV never leaks preprocessing info."""
    candidates = {
        "logistic_regression": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED,
            )),
        ]),
        "random_forest": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, class_weight="balanced_subsample",
                random_state=RANDOM_SEED, n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline(steps=[
            ("preprocess", build_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                random_state=RANDOM_SEED, eval_metric="mlogloss",
                n_jobs=-1,
            )),
        ]),
    }
    return candidates


def cross_validate_candidates(X_train, y_train):
    """Stratified 5-fold CV comparison on the TRAIN split only."""
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # xgboost needs numeric labels

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    results = {}
    candidates = build_candidates()
    for name, pipe in candidates.items():
        t0 = time.time()
        # LogReg/RF work fine with string labels; xgboost needs encoded labels.
        y_used = y_train_enc if name == "xgboost" else y_train
        scores = cross_validate(pipe, X_train, y_used, cv=skf, scoring=scoring, n_jobs=1)
        elapsed = time.time() - t0
        results[name] = {
            metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for metric, vals in scores.items() if metric.startswith("test_")
        }
        results[name]["fit_time_sec"] = round(elapsed, 1)
        print(f"[{name}] f1_macro={results[name]['test_f1_macro']['mean']:.4f} "
              f"(+/-{results[name]['test_f1_macro']['std']:.4f})  "
              f"acc={results[name]['test_accuracy']['mean']:.4f}  "
              f"({elapsed:.1f}s)")
    return results, le


def evaluate_on_split(pipe, X, y_true_labels, class_labels=CLASS_LABELS, label_encoder=None, is_xgb=False):
    if is_xgb:
        y_pred_enc = pipe.predict(X)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        proba = pipe.predict_proba(X)
        # reorder proba columns to CLASS_LABELS order
        proba_df_cols = label_encoder.inverse_transform(np.arange(proba.shape[1]))
    else:
        y_pred = pipe.predict(X)
        proba = pipe.predict_proba(X)
        proba_df_cols = pipe.named_steps["clf"].classes_

    col_index = {c: i for i, c in enumerate(proba_df_cols)}
    proba_ordered = proba[:, [col_index[c] for c in class_labels]]

    metrics = {
        "accuracy": float(accuracy_score(y_true_labels, y_pred)),
        "f1_macro": float(f1_score(y_true_labels, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true_labels, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true_labels, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true_labels, y_pred, average="macro")),
        "per_class": classification_report(
            y_true_labels, y_pred, labels=class_labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true_labels, y_pred, labels=class_labels).tolist(),
    }
    try:
        y_true_bin = np.array([class_labels.index(v) for v in y_true_labels])
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_true_bin, proba_ordered, multi_class="ovr", average="macro")
        )
    except Exception as e:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_error"] = str(e)

    return metrics, y_pred, proba_ordered


def baseline_metrics(y_train, y_test):
    """Majority-class baseline for comparison (brief 5.3 baseline comparison requirement)."""
    majority = y_train.value_counts().idxmax()
    y_pred = [majority] * len(y_test)
    return {
        "strategy": f"predict majority class ({majority})",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def main():
    print("Loading data...")
    X, y = load_xy()
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}\n")

    print("=== Stratified 5-fold CV on TRAIN split ===")
    cv_results, label_encoder = cross_validate_candidates(X_train, y_train)

    # Select best by f1_macro (imbalance-aware), not raw accuracy.
    best_name = max(cv_results, key=lambda k: cv_results[k]["test_f1_macro"]["mean"])
    print(f"\nSelected by CV f1_macro: {best_name}")

    print("\n=== Fitting all 3 candidates on TRAIN, evaluating on VAL for comparison table ===")
    candidates = build_candidates()
    val_comparison = {}
    fitted_pipes = {}
    y_train_enc = label_encoder.transform(y_train)

    for name, pipe in candidates.items():
        is_xgb = name == "xgboost"
        pipe.fit(X_train, y_train_enc if is_xgb else y_train)
        fitted_pipes[name] = pipe
        metrics, _, _ = evaluate_on_split(
            pipe, X_val, y_val, label_encoder=label_encoder, is_xgb=is_xgb
        )
        val_comparison[name] = {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
        }
        print(f"[{name}] VAL acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f} "
              f"roc_auc={metrics['roc_auc_ovr_macro']}")

    print(f"\n=== Final evaluation of best model ({best_name}) on held-out TEST set ===")
    best_pipe = fitted_pipes[best_name]
    is_xgb = best_name == "xgboost"
    test_metrics, y_pred, proba = evaluate_on_split(
        best_pipe, X_test, y_test, label_encoder=label_encoder, is_xgb=is_xgb
    )
    print(f"TEST accuracy={test_metrics['accuracy']:.4f}  f1_macro={test_metrics['f1_macro']:.4f}")
    print("Confusion matrix (rows=true, cols=pred), order:", CLASS_LABELS)
    print(np.array(test_metrics["confusion_matrix"]))

    baseline = baseline_metrics(y_train, y_test)
    print(f"\nBaseline (majority class): acc={baseline['accuracy']:.4f} f1_macro={baseline['f1_macro']:.4f}")

    # ---- Refit best model on TRAIN+VAL (common practice: maximize data for
    # the deployed model once test set has served its evaluation purpose) ----
    print(f"\nRefitting {best_name} on TRAIN+VAL for the deployed artifact...")
    import pandas as pd
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = pd.concat([y_train, y_val], axis=0)
    final_pipe = build_candidates()[best_name]
    if is_xgb:
        y_trainval_enc = label_encoder.transform(y_trainval)
        final_pipe.fit(X_trainval, y_trainval_enc)
    else:
        final_pipe.fit(X_trainval, y_trainval)

    # Re-evaluate the REFIT model on the untouched test set (final reported numbers)
    final_test_metrics, _, _ = evaluate_on_split(
        final_pipe, X_test, y_test, label_encoder=label_encoder, is_xgb=is_xgb
    )
    print(f"Refit-on-trainval TEST accuracy={final_test_metrics['accuracy']:.4f} "
          f"f1_macro={final_test_metrics['f1_macro']:.4f}")

    # ---- Serialize artifact ----
    artifact = {
        "pipeline": final_pipe,
        "model_name": best_name,
        "class_labels": CLASS_LABELS,
        "is_xgb": is_xgb,
        "label_encoder": label_encoder if is_xgb else None,
        "feature_list": list(X.columns),
    }
    joblib.dump(artifact, ARTIFACTS_DIR / "model_pipeline.joblib")
    print(f"\nSaved artifact to {ARTIFACTS_DIR / 'model_pipeline.joblib'}")

    # ---- Save full report JSON (used by /model/info and Model Insights page) ----
    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "cv_folds": CV_FOLDS,
        "dataset": {
            "n_total": int(len(X)),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "class_labels": CLASS_LABELS,
        },
        "cv_comparison": cv_results,
        "val_comparison": val_comparison,
        "selected_model": best_name,
        "selection_rationale": (
            f"Selected '{best_name}' based on macro-F1 during 5-fold stratified "
            f"cross-validation on the training split, not raw accuracy, because "
            f"macro-F1 weights all three classes equally and this dataset is "
            f"imbalanced (Graduate ~50%, Dropout ~32%, Enrolled ~18%). A model "
            f"that only predicts the majority class well would still score high "
            f"accuracy while badly under-serving 'Enrolled' students, who are "
            f"arguably the group where early intervention matters most."
        ),
        "baseline_majority_class": baseline,
        "test_set_metrics_pre_refit": test_metrics,
        "test_set_metrics_final_model": final_test_metrics,
        "model_version": "2026-08-29-v1",
    }
    with open(REPORTS_DIR / "model_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {REPORTS_DIR / 'model_comparison_report.json'}")


if __name__ == "__main__":
    main()
