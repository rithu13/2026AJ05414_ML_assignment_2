"""Train and persist the five classifiers required for ML Assignment 2."""

from pathlib import Path
import json

import joblib
import pandas as pd

# Scikit-learn packages a reproducible copy of the UCI Wisconsin Diagnostic dataset.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TARGET_COLUMN = "target"


def calculate_metrics(y_true, y_pred, y_score):
    """Return the six evaluation metrics required by the assignment."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def prediction_scores(model, features):
    """Return continuous positive-class scores for AUC calculation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    # SVC exposes ranking scores without the extra cost of probability calibration.
    return model.decision_function(features)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target = dataset.target

    # Stratification preserves the malignant/benign ratio in both partitions.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    # Scaling stays inside each pipeline, so it is fitted only on training data.
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                ),
            ],
            memory=None,
        ),
        # Limit tree complexity to reduce overfitting while retaining interpretable rules.
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=3,
            ccp_alpha=0.0,
            random_state=RANDOM_STATE,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ],
            memory=None,
        ),
        "Naive Bayes": Pipeline(
            [("scaler", StandardScaler()), ("classifier", GaussianNB())],
            memory=None,
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        gamma="scale",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            memory=None,
        ),
    }

    model_files = {}
    metrics = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = prediction_scores(model, x_test)
        metrics[model_name] = calculate_metrics(y_test, predictions, scores)

        filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)
        model_files[model_name] = filename

    # Persist the untouched hold-out rows for evaluation in the Streamlit app.
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    metadata = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "target_column": TARGET_COLUMN,
        "target_labels": {"0": "malignant", "1": "benign"},
        "feature_names": list(features.columns),
        "model_files": model_files,
        "metrics": metrics,
        "split": {
            "random_state": RANDOM_STATE,
            "test_size": 0.2,
            "training_rows": len(x_train),
            "test_rows": len(x_test),
        },
    }
    with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    results = pd.DataFrame(metrics).T
    print("\nModel evaluation on the held-out test set:\n")
    print(results.to_string())
    print(f"\nBest model by F1: {results['F1'].idxmax()}")
    print(f"Saved {len(models)} models and {len(test_data)} test rows in {BASE_DIR}")


if __name__ == "__main__":
    main()
