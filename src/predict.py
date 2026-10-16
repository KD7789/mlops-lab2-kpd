"""Load the trained model and its metadata, and turn inputs into predictions."""
import json
from pathlib import Path

import joblib
import numpy as np


class ModelService:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "model.pkl")
        self.info = json.loads((model_dir / "model_info.json").read_text())
        self.features = self.info["features"]
        self.classes = self.info["classes"]

    def predict(self, rows):
        """Predict a list of samples.

        Each row is a dict with a value for every feature (already validated).
        Returns one result per row: the predicted class, its confidence, and the probability of every class.
        """
        X = np.array([[row[name] for name in self.features] for row in rows], dtype=float)
        probabilities = self.model.predict_proba(X)

        results = []
        for probs in probabilities:
            best = int(np.argmax(probs))
            results.append({
                "prediction": self.classes[best],
                "confidence": round(float(probs[best]), 4),
                "probabilities": {
                    name: round(float(p), 4) for name, p in zip(self.classes, probs)
                },
            })
        return results
