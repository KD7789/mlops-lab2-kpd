"""Streamlit dashboard for the wine cultivar API.

Run:  streamlit run streamlit_app.py
By default it talks to the deployed Cloud Run service. To use a local API:
      API_URL=http://localhost:8080 streamlit run streamlit_app.py
"""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://wine-api-csexo5taia-ue.a.run.app").rstrip("/")
EXAMPLES_FILE = Path(__file__).resolve().parent / "samples" / "batch.json"
MAX_BATCH = 100

st.set_page_config(page_title="Wine Cultivar Classifier", page_icon="🍷", layout="wide")


# helpers

def pretty(name):
    """alcalinity_of_ash -> Alcalinity of ash"""
    if name == "od280_od315_of_diluted_wines":
        return "OD280/OD315 of diluted wines"
    return name.replace("_", " ").capitalize()


def pretty_class(label):
    """cultivar_1 -> Cultivar 1"""
    return label.replace("_", " ").capitalize()


def slider_step(stats):
    return 1.0 if stats["max"] - stats["min"] > 50 else 0.01


def api_get(path):
    response = requests.get(f"{API_URL}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def api_predict(payload):
    """Returns (status_code, json_body)."""
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    return response.status_code, response.json()


@st.cache_data(ttl=300)
def load_model_info(url):
    return api_get("/model-info")


def load_examples():
    if EXAMPLES_FILE.exists():
        return json.loads(EXAMPLES_FILE.read_text())
    return []


# sidebar: API status and model info 

with st.sidebar:
    st.title("Wine classifier")
    st.caption(f"API: {API_URL}")

    try:
        health = api_get("/health")
        st.success(f"API online · model version {health['model_version']}")
    except requests.RequestException as err:
        st.error(f"API not reachable: {err}")
        st.stop()

    info = load_model_info(API_URL)
    st.subheader("Model")
    st.write(f"**Algorithm:** {info['algorithm']}")
    st.write(f"**Dataset:** {info['dataset']}")
    col_a, col_b = st.columns(2)
    col_a.metric("Test accuracy", f"{info['metrics']['accuracy']:.1%}")
    col_b.metric("Macro F1", f"{info['metrics']['f1_macro']:.3f}")
    st.caption(f"Trained {info['trained_at_utc']} · scikit-learn {info['sklearn_version']}")

features = info["features"]
stats = info["feature_stats"]
examples = load_examples()

st.title("Wine cultivar classifier")
st.write(
    "Predict which of three grape cultivars a wine comes from, using 13 results "
    "of its chemical analysis. Predictions come from the Flask API running on Cloud Run."
)

single_tab, batch_tab = st.tabs(["Single wine", "Batch (CSV)"])

# single prediction

with single_tab:
    for name in features:
        key = f"feature_{name}"
        if key not in st.session_state:
            st.session_state[key] = float(stats[name]["mean"])

    def apply_preset():
        choice = st.session_state["preset"]
        if choice == "Average wine":
            values = {name: stats[name]["mean"] for name in features}
        else:
            values = examples[int(choice.split()[-1]) - 1]
        for name in features:
            low, high = stats[name]["min"], stats[name]["max"]
            st.session_state[f"feature_{name}"] = float(min(max(values[name], low), high))

    presets = ["Average wine"] + [f"Example wine {i + 1}" for i in range(len(examples))]
    st.selectbox("Start from", presets, key="preset", on_change=apply_preset)

    columns = st.columns(3)
    for i, name in enumerate(features):
        with columns[i % 3]:
            st.slider(
                pretty(name),
                min_value=float(stats[name]["min"]),
                max_value=float(stats[name]["max"]),
                step=slider_step(stats[name]),
                key=f"feature_{name}",
            )

    if st.button("Predict cultivar", type="primary"):
        sample = {name: st.session_state[f"feature_{name}"] for name in features}
        with st.spinner("Asking the API..."):
            try:
                status, body = api_predict(sample)
            except requests.RequestException as err:
                st.error(f"Could not reach the API: {err}")
                st.stop()

        if status != 200:
            st.error(f"API returned {status}: {body.get('error')}")
            for detail in body.get("details", []):
                st.write(f"- {detail}")
        else:
            left, right = st.columns([1, 2])
            with left:
                st.metric("Predicted cultivar", pretty_class(body["prediction"]))
                st.metric("Confidence", f"{body['confidence']:.1%}")
                st.caption(f"Model version {body['model_version']}")
            with right:
                chart = pd.DataFrame(
                    {"Probability": list(body["probabilities"].values())},
                    index=[pretty_class(c) for c in body["probabilities"]],
                )
                st.bar_chart(chart)

# batch prediction

with batch_tab:
    st.write(
        f"Upload a CSV with one row per wine and these {len(features)} columns "
        f"(up to {MAX_BATCH} rows): " + ", ".join(f"`{f}`" for f in features)
    )
    if examples:
        st.download_button(
            "Download an example CSV",
            data=pd.DataFrame(examples)[features].to_csv(index=False),
            file_name="wine_examples.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        table = pd.read_csv(uploaded)
        missing = [f for f in features if f not in table.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        elif len(table) > MAX_BATCH:
            st.error(f"The file has {len(table)} rows; the maximum is {MAX_BATCH}.")
        else:
            with st.spinner(f"Predicting {len(table)} wines..."):
                status, body = api_predict(table[features].to_dict(orient="records"))
            if status != 200:
                st.error(f"API returned {status}: {body.get('error')}")
                for detail in body.get("details", []):
                    st.write(f"- {detail}")
            else:
                results = table[features].copy()
                results.insert(0, "confidence", [p["confidence"] for p in body["predictions"]])
                results.insert(0, "prediction", [pretty_class(p["prediction"]) for p in body["predictions"]])
                st.dataframe(results)
                st.download_button(
                    "Download results",
                    data=results.to_csv(index=False),
                    file_name="wine_predictions.csv",
                    mime="text/csv",
                )
