"""Automated API tests. They use Flask's test client, so no server has to be running.

A fresh model is trained into a temporary folder once per test session.
"""
import json
from pathlib import Path

import pytest

from main import MAX_BATCH, create_app
from train import run_training

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("model")
    run_training(model_dir=directory, min_accuracy=0.90)
    return directory


@pytest.fixture(scope="session")
def client(model_dir):
    return create_app(model_dir).test_client()


@pytest.fixture
def sample():
    return json.loads((SAMPLES / "sample.json").read_text())


@pytest.fixture
def batch():
    return json.loads((SAMPLES / "batch.json").read_text())


# health and model info 

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_model_info(client):
    info = client.get("/model-info").get_json()
    assert len(info["features"]) == 13
    assert info["classes"] == ["cultivar_1", "cultivar_2", "cultivar_3"]
    assert info["metrics"]["accuracy"] >= 0.90


# predictions

def test_predict_single_sample(client, sample):
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == "cultivar_1"
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.001)
    assert body["confidence"] == max(body["probabilities"].values())


def test_predict_batch(client, batch):
    response = client.post("/predict", json=batch)
    assert response.status_code == 200
    labels = [p["prediction"] for p in response.get_json()["predictions"]]
    assert labels == ["cultivar_1", "cultivar_2", "cultivar_3"]


# input validation

def test_missing_field_returns_400(client, sample):
    del sample["hue"]
    response = client.post("/predict", json=sample)
    assert response.status_code == 400
    assert "missing fields: hue" in response.get_json()["details"]


def test_non_numeric_value_returns_400(client, sample):
    sample["alcohol"] = "abc"
    response = client.post("/predict", json=sample)
    assert response.status_code == 400
    assert "alcohol must be a number" in response.get_json()["details"]


def test_negative_value_returns_400(client, sample):
    sample["proline"] = -5
    response = client.post("/predict", json=sample)
    assert response.status_code == 400
    assert "proline must not be negative" in response.get_json()["details"]


def test_unknown_field_returns_400(client, sample):
    sample["colour"] = 3.2
    response = client.post("/predict", json=sample)
    assert response.status_code == 400
    assert "unknown fields: colour" in response.get_json()["details"]


def test_non_json_body_returns_400(client):
    response = client.post("/predict", data="not json", content_type="text/plain")
    assert response.status_code == 400


def test_empty_batch_returns_400(client):
    response = client.post("/predict", json=[])
    assert response.status_code == 400


def test_batch_over_limit_returns_400(client, sample):
    response = client.post("/predict", json=[sample] * (MAX_BATCH + 1))
    assert response.status_code == 400


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert "error" in response.get_json()


# training

def test_training_rejects_low_accuracy(tmp_path):
    with pytest.raises(SystemExit):
        run_training(model_dir=tmp_path / "strict", min_accuracy=0.999)
    assert not (tmp_path / "strict" / "model.pkl").exists()
