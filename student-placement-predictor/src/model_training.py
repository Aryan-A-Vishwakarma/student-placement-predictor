"""
model_training.py

Trains and evaluates two classification models (Logistic Regression and
Random Forest) on the cleaned, feature-engineered student dataset, then
saves the better-performing model to models/model.pkl for later use in
prediction.py and the Streamlit app.

Run: python src/model_training.py
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from data_preprocessing import prepare_dataset, build_preprocessor

RANDOM_STATE = 42


def train_and_evaluate():
    # 1. Load fully cleaned + feature-engineered data
    X, y, _ = prepare_dataset()

    # 2. Train-test split (80/20).
    #    stratify=y ensures both the train and test sets keep roughly the
    #    same Placed/Not-Placed ratio as the full dataset - important
    #    because a random split alone could accidentally put too many
    #    "Placed" students in one set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

    results = {}
    fitted_pipelines = {}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    for name, model in models.items():
        # Pipeline bundles preprocessing + model together. Crucially, when
        # we call pipeline.fit(X_train, y_train), the scaler/encoder inside
        # build_preprocessor() is fit ONLY on X_train - it never sees
        # X_test until .predict()/.predict_proba() is called on it. This
        # is how we avoid data leakage through preprocessing.
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        cm = confusion_matrix(y_test, y_pred)

        results[name] = metrics
        fitted_pipelines[name] = pipeline

        print(f"\n=== {name} ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("Confusion Matrix (rows=actual, cols=predicted) [Not Placed, Placed]:")
        print(cm)
        print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

    # 3. Pick the final model based on F1-Score (balances precision and
    #    recall - a reasonable default when there's no strong business
    #    reason to prioritize one over the other).
    final_model_name = max(results, key=lambda name: results[name]["F1-Score"])
    print(f"\nSelected final model: {final_model_name} (highest F1-Score)")

    final_pipeline = fitted_pipelines[final_model_name]

    # 4. Save the trained pipeline (preprocessing + model together, so
    #    prediction.py doesn't need to re-implement any preprocessing logic).
    with open("models/model.pkl", "wb") as f:
        pickle.dump({"pipeline": final_pipeline, "model_name": final_model_name}, f)
    print("Saved final model to models/model.pkl")

    # 5. Save metrics + which model won, as JSON, so README generation
    #    and the notebook can reference the exact same real numbers.
    with open("models/results.json", "w") as f:
        json.dump({
            "results": results,
            "final_model": final_model_name,
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
        }, f, indent=2)
    print("Saved metrics to models/results.json")

    return results, final_model_name, final_pipeline, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_and_evaluate()
