"""Smoke test for a running API (local or on Cloud Run).

Usage:
    python scripts/smoke_test.py <base-url> [expected-model-version]

Examples:
    python scripts/smoke_test.py http://localhost:8080
    python scripts/smoke_test.py https://wine-api-xxxx.us-east1.run.app abc1234

Uses only the Python standard library, so it runs anywhere (including as the last step of Cloud Build). Exits with an error if any check fails.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def call(url, payload=None):
    """Return (status_code, json_body). HTTP errors are returned, not raised."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def check(name, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {name}  {detail}")
    return condition


def main(base_url, expected_version=None):
    base_url = base_url.rstrip("/")
    sample = json.loads((SAMPLES / "sample.json").read_text())
    batch = json.loads((SAMPLES / "batch.json").read_text())

    for attempt in range(5):
        try:
            status, health = call(f"{base_url}/health")
            break
        except (urllib.error.URLError, TimeoutError) as err:
            print(f"health check attempt {attempt + 1} failed: {err}")
            time.sleep(5)
    else:
        sys.exit("FAIL  service did not respond")

    results = [check("GET /health", status == 200 and health.get("status") == "ok", str(health))]
    if expected_version:
        results.append(check(
            "deployed version", health.get("model_version") == expected_version,
            f"expected {expected_version}, got {health.get('model_version')}",
        ))

    status, info = call(f"{base_url}/model-info")
    results.append(check("GET /model-info", status == 200 and len(info.get("features", [])) == 13,
                         f"accuracy={info.get('metrics', {}).get('accuracy')}"))

    status, body = call(f"{base_url}/predict", sample)
    results.append(check("POST /predict (single)", status == 200 and body.get("prediction") == "cultivar_1",
                         f"{body.get('prediction')} ({body.get('confidence')})"))

    status, body = call(f"{base_url}/predict", batch)
    labels = [p["prediction"] for p in body.get("predictions", [])]
    results.append(check("POST /predict (batch)", status == 200 and labels == ["cultivar_1", "cultivar_2", "cultivar_3"],
                         str(labels)))

    status, body = call(f"{base_url}/predict", {"alcohol": "abc"})
    results.append(check("POST /predict (invalid input -> 400)", status == 400, body.get("error", "")))

    if not all(results):
        sys.exit("Smoke test FAILED")
    print("Smoke test passed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
