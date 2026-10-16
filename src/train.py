"""Train the wine cultivar classifier and save it with its metadata.

Writes two files to MODEL_DIR (default: model/):
    model.pkl         scikit-learn pipeline (StandardScaler + LogisticRegression)
    model_info.json   features, classes, metrics and versions (served by /model-info)

If test accuracy is below MIN_ACCURACY (default 0.90) the script exits with an error. Because the Docker image runs this script during the build, a bad model fails the build and never reaches Cloud Run.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLASS_NAMES = ["cultivar_1", "cultivar_2", "cultivar_3"]


def load_data():
    data = load_wine()
    features = [name.replace("/", "_") for name in data.feature_names]
    return data.data, data.target, features


def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def feature_stats(X, features):
    """Min / max / mean per feature, used by the dashboard for slider ranges."""
    return {
        name: {
            "min": round(float(X[:, i].min()), 3),
            "max": round(float(X[:, i].max()), 3),
            "mean": round(float(X[:, i].mean()), 3),
        }
        for i, name in enumerate(features)
    }


def run_training(model_dir=None, min_accuracy=None):
    model_dir = Path(model_dir or os.getenv("MODEL_DIR", "model"))
    if min_accuracy is None:
        min_accuracy = float(os.getenv("MIN_ACCURACY") or 0.90)

    X, y, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }
    print(f"Test accuracy: {metrics['accuracy']:.4f}, macro F1: {metrics['f1_macro']:.4f}")

    if metrics["accuracy"] < min_accuracy:
        sys.exit(
            f"Accuracy {metrics['accuracy']:.4f} is below the required {min_accuracy:g}. "
            "Model not saved."
        )

    info = {
        "model_name": "wine-cultivar-classifier",
        "model_version": os.getenv("MODEL_VERSION", "local"),
        "algorithm": "StandardScaler + LogisticRegression",
        "dataset": "UCI Wine (sklearn load_wine), 178 samples",
        "features": features,
        "classes": CLASS_NAMES,
        "feature_stats": feature_stats(X, features),
        "metrics": metrics,
        "min_accuracy": min_accuracy,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved {model_dir / 'model.pkl'} and {model_dir / 'model_info.json'}")
    return info


if __name__ == "__main__":
    run_training()
