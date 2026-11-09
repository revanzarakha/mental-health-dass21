"""Train DASS-21 severity classifiers.

The dataset contains 21 DASS-21 items (0-3 scale) and the three domain
scores (0-42). Severity labels are derived from the published DASS-21
cut-offs, so this project reproduces the scoring rule with a classifier
instead of learning a new clinical signal. See README for limitations.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dass21_deidentified.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
LABELS = [0, 1, 2, 3, 4]
LEVEL_NAMES = ["Normal", "Ringan", "Sedang", "Berat", "Sangat Berat"]
SCORE_COLUMNS = ["Depresi_Score", "Kecemasan_Score", "Stress_Score"]

# DASS-21 severity cut-offs (score scale 0-42).
CUTOFFS = {
    "depresi": [(9, 0), (13, 1), (20, 2), (27, 3)],
    "anxiety": [(7, 0), (9, 1), (14, 2), (19, 3)],
    "stress": [(14, 0), (18, 1), (25, 2), (33, 3)],
}


def severity_label(domain: str, score: float) -> int:
    """Map a DASS-21 domain score to a 0-4 severity label."""
    for cutoff, label in CUTOFFS[domain]:
        if score <= cutoff:
            return label
    return 4


def detect_question_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look like 0-3 DASS-21 items."""
    columns: list[str] = []
    for column in df.columns:
        if column in SCORE_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty and values.between(0, 3).mean() >= 0.85:
            columns.append(column)
    if len(columns) != 21:
        raise ValueError(f"Expected 21 DASS-21 items, found {len(columns)}.")
    return columns


def build_pipelines() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate(domain: str, df: pd.DataFrame, question_columns: list[str]) -> dict:
    score_column = {
        "depresi": "Depresi_Score",
        "anxiety": "Kecemasan_Score",
        "stress": "Stress_Score",
    }[domain]
    y = df[score_column].apply(lambda score: severity_label(domain, score)).astype(int)
    X = df[question_columns].astype(float)

    distribution = {LEVEL_NAMES[k]: int(v) for k, v in y.value_counts().sort_index().items()}

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    results: dict = {
        "samples": int(len(y)),
        "features": len(question_columns),
        "class_distribution": distribution,
        "models": {},
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    for name, pipeline in build_pipelines().items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        report = classification_report(
            y_test,
            predictions,
            labels=LABELS,
            target_names=LEVEL_NAMES,
            output_dict=True,
            zero_division=0,
        )

        results["models"][name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision_macro": round(
                float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "recall_macro": round(
                float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "f1_macro": round(
                float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4
            ),
            "f1_weighted": round(
                float(f1_score(y_test, predictions, average="weighted", zero_division=0)), 4
            ),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "per_class": {
                label: {
                    "precision": round(float(report[label]["precision"]), 4),
                    "recall": round(float(report[label]["recall"]), 4),
                    "f1": round(float(report[label]["f1-score"]), 4),
                    "support": int(report[label]["support"]),
                }
                for label in LEVEL_NAMES
            },
        }

        joblib.dump(
            pipeline,
            MODEL_DIR / f"pipeline_{'logreg' if name == 'logistic_regression' else 'rf'}_{domain}.pkl",
        )

    return results


def write_markdown(results: dict, output_path: Path) -> None:
    lines = [
        "# Evaluation results",
        "",
        "Generated by `python train.py`.",
        "",
        "| Domain | Model | Accuracy | Macro F1 | CV accuracy (3-fold) |",
        "|---|---|---:|---:|---:|",
    ]
    for domain, domain_results in results["domains"].items():
        for model, metrics in domain_results["models"].items():
            lines.append(
                f"| {domain} | {model.replace('_', ' ')} | {metrics['accuracy']:.3f} "
                f"| {metrics['f1_macro']:.3f} "
                f"| {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    question_columns = detect_question_columns(df)

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps(question_columns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results = {
        "dataset": DATA_PATH.name,
        "samples": int(len(df)),
        "questionnaire": "DASS-21",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "domains": {},
    }

    for domain in ("depresi", "anxiety", "stress"):
        print(f"\n=== {domain.upper()} ===")
        domain_results = evaluate(domain, df, question_columns)
        results["domains"][domain] = domain_results
        for model, metrics in domain_results["models"].items():
            print(
                f"{model:20s} accuracy={metrics['accuracy']:.3f} "
                f"macro-F1={metrics['f1_macro']:.3f} "
                f"cv={metrics['cv_accuracy_mean']:.3f}+/-{metrics['cv_accuracy_std']:.3f}"
            )

    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(results, RESULTS_DIR / "metrics.md")
    print(f"\nSaved artifacts to {MODEL_DIR} and {RESULTS_DIR}")


if __name__ == "__main__":
    main()
