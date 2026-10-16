"""Flask API for the wine cultivar classifier.

Endpoints:
    GET  /health       quick check that the service is up (used after each deploy)
    GET  /model-info   model metadata: features, classes, metrics, version
    POST /predict      one sample (JSON object) or a batch (JSON list, max 100)

Run locally:   python src/main.py
In Docker:     gunicorn serves create_app() (see Dockerfile)
"""
import logging
import math
import os

from flask import Flask, jsonify, request

from predict import ModelService

MAX_BATCH = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wine-api")


def error_response(status, message, details=None):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), status


def validate_sample(sample, features):
    """Check one sample. Returns (clean_sample, list_of_problems)."""
    if not isinstance(sample, dict):
        return None, ["each sample must be a JSON object"]

    problems = []
    missing = [name for name in features if name not in sample]
    unknown = [name for name in sample if name not in features]
    if missing:
        problems.append("missing fields: " + ", ".join(missing))
    if unknown:
        problems.append("unknown fields: " + ", ".join(unknown))

    clean = {}
    for name in features:
        if name not in sample:
            continue
        value = sample[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{name} must be a number")
        elif not math.isfinite(value):
            problems.append(f"{name} must be a finite number")
        elif value < 0:
            problems.append(f"{name} must not be negative")
        else:
            clean[name] = float(value)
    return clean, problems


def create_app(model_dir=None):
    app = Flask(__name__)
    service = ModelService(model_dir or os.getenv("MODEL_DIR", "model"))
    version = service.info["model_version"]
    logger.info("Loaded %s (version %s)", service.info["model_name"], version)

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_version=version)

    @app.get("/model-info")
    def model_info():
        return jsonify(service.info)

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return error_response(400, "request body must be JSON (Content-Type: application/json)")

        is_batch = isinstance(payload, list)
        samples = payload if is_batch else [payload]
        if not samples:
            return error_response(400, "batch is empty")
        if len(samples) > MAX_BATCH:
            return error_response(400, f"batch has {len(samples)} samples; the maximum is {MAX_BATCH}")

        clean_samples, problems = [], []
        for i, sample in enumerate(samples):
            clean, sample_problems = validate_sample(sample, service.features)
            prefix = f"sample {i}: " if is_batch else ""
            problems.extend(prefix + p for p in sample_problems)
            clean_samples.append(clean)
        if problems:
            logger.info("Rejected request: %s", "; ".join(problems))
            return error_response(400, "invalid input", problems)

        results = service.predict(clean_samples)
        logger.info("Predicted %d sample(s)", len(results))

        body = {"predictions": results} if is_batch else results[0]
        body["model_version"] = version
        return jsonify(body)

    @app.errorhandler(404)
    def not_found(_):
        return error_response(404, "not found; endpoints are /health, /model-info and /predict")

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error_response(405, "method not allowed")

    @app.errorhandler(500)
    def server_error(_):
        logger.exception("Unhandled error")
        return error_response(500, "internal server error")

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
