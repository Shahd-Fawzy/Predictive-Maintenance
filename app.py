"""
Project 5 - Industrial Predictive Maintenance
Streamlit deployment app

Run with:
    streamlit run app/app.py

Expects to find, relative to the folder it's launched from:
    models/best_model.pkl
    models/best_model_meta.json
    models/feature_reference_stats.json
    src/features.py, src/modeling.py  (the shared feature/metric logic)
"""

import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import FEATURE_COLS as DEFAULT_FEATURE_COLS, build_feature_row  # noqa: E402
from src.modeling import risk_tier  # noqa: E402

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

MODEL_PATH = "models/best_model.pkl"
META_PATH = "models/best_model_meta.json"
STATS_PATH = "models/feature_reference_stats.json"
LOG_PATH = "app/prediction_history.csv"

LOG_COLUMNS = [
    "ObservationID", "PredictionTime", "ModelName", "ModelVersion",
    "FailureProbability", "PredictedFailure", "RiskTier",
    "RecommendedAction", "ActualFailure",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def load_reference_stats():
    if not os.path.exists(STATS_PATH):
        return None
    with open(STATS_PATH) as f:
        return json.load(f)


def load_history():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_to_history(rows: list):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    history = load_history()
    next_id = (int(history["ObservationID"].max()) + 1) if len(history) else 1

    for i, row in enumerate(rows):
        row["ObservationID"] = next_id + i
        row["PredictionTime"] = datetime.now().isoformat(timespec="seconds")

    updated = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    updated.to_csv(LOG_PATH, index=False)
    return updated


def check_out_of_range(feature_row: dict, stats: dict):
    """Very lightweight drift/sanity check: which inputs fall outside the
    1st-99th percentile band seen during training?"""
    if stats is None:
        return []
    flags = []
    for feature, value in feature_row.items():
        if feature not in stats:
            continue
        lo, hi = stats[feature]["p01"], stats[feature]["p99"]
        if value < lo or value > hi:
            flags.append(f"{feature} = {value:.2f} is outside the typical training range [{lo:.2f}, {hi:.2f}]")
    return flags


# ---------------------------------------------------------------------------
# Load everything up front
# ---------------------------------------------------------------------------
model, meta = load_model()
stats = load_reference_stats()

if model is None:
    st.error(
        "No trained model found at `models/best_model.pkl`. "
        "Run the full notebook first (Part 3, section 3.14 saves it)."
    )
    st.stop()

FEATURE_COLS = meta.get("features", DEFAULT_FEATURE_COLS)
MODEL_NAME = meta["model_name"]
DEFAULT_THRESHOLD = meta["metrics"].get("Threshold", 0.5)

st.title("Industrial Predictive Maintenance Dashboard")
st.caption(
    f"Model in use: **{MODEL_NAME}**  |  Decision threshold: **{DEFAULT_THRESHOLD:.3f}**  "
    "| Trained on the AI4I 2020 synthetic dataset - not a real industrial system."
)

tab_single, tab_batch, tab_monitor = st.tabs(
    ["Single Machine Check", "Batch Scoring", "Monitoring"]
)

# ---------------------------------------------------------------------------
# Tab 1 - single machine
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Check one machine")
    col1, col2, col3 = st.columns(3)

    with col1:
        machine_type = st.selectbox("Product quality type", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature [K]", value=298.0, step=0.1, format="%.1f")
        process_temp = st.number_input("Process temperature [K]", value=308.0, step=0.1, format="%.1f")

    with col2:
        rpm = st.number_input("Rotational speed [rpm]", value=1500, step=10)
        torque = st.number_input("Torque [Nm]", value=40.0, step=0.1, format="%.1f")

    with col3:
        tool_wear = st.number_input("Tool wear [min]", value=100, step=1)

    if st.button("Predict failure risk", type="primary"):
        X_input = build_feature_row(air_temp, process_temp, rpm, torque, tool_wear, machine_type)
        X_input = X_input[FEATURE_COLS]
        features_dict = X_input.iloc[0].to_dict()

        probability = float(model.predict_proba(X_input)[0, 1])
        predicted_failure = int(probability >= DEFAULT_THRESHOLD)
        tier, action = risk_tier(probability, DEFAULT_THRESHOLD)

        result_col, warn_col = st.columns([1, 1])

        with result_col:
            st.metric("Failure probability", f"{probability:.1%}")
            tier_color = {"Low": "green", "Medium": "orange", "High": "red"}[tier]
            st.markdown(f"**Risk tier:** :{tier_color}[{tier}]")
            st.write(f"**Recommended action:** {action}")

        with warn_col:
            flags = check_out_of_range(features_dict, stats)
            if flags:
                st.warning("Input values outside the typical training range:")
                for flag in flags:
                    st.write(f"- {flag}")
            else:
                st.success("All inputs are within the typical training range.")

        append_to_history([{
            "ModelName": MODEL_NAME,
            "ModelVersion": "v1.0",
            "FailureProbability": round(probability, 6),
            "PredictedFailure": predicted_failure,
            "RiskTier": tier,
            "RecommendedAction": action,
            "ActualFailure": None,
        }])
        st.caption("Logged to app/prediction_history.csv")

# ---------------------------------------------------------------------------
# Tab 2 - batch scoring
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Score a batch of machines from a CSV")
    st.write(
        "Expected columns: `Type`, `Air temperature [K]`, `Process temperature [K]`, "
        "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        required_cols = ["Type", "Air temperature [K]", "Process temperature [K]",
                          "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        missing = [c for c in required_cols if c not in raw_df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            try:
                engineered_rows = [
                    build_feature_row(
                        row["Air temperature [K]"], row["Process temperature [K]"],
                        row["Rotational speed [rpm]"], row["Torque [Nm]"],
                        row["Tool wear [min]"], row["Type"],
                    ).iloc[0]
                    for _, row in raw_df.iterrows()
                ]
            except ValueError as e:
                st.error(f"Some rows couldn't be processed: {e}")
                st.stop()

            X_batch = pd.DataFrame(engineered_rows)[FEATURE_COLS]

            probabilities = model.predict_proba(X_batch)[:, 1]
            predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
            tiers, actions = zip(*[risk_tier(p, DEFAULT_THRESHOLD) for p in probabilities])

            results = raw_df.copy()
            results["FailureProbability"] = np.round(probabilities, 4)
            results["PredictedFailure"] = predictions
            results["RiskTier"] = tiers
            results["RecommendedAction"] = actions

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download results as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="batch_predictions.csv",
                mime="text/csv",
            )

            log_rows = [
                {
                    "ModelName": MODEL_NAME, "ModelVersion": "v1.0",
                    "FailureProbability": round(float(p), 6), "PredictedFailure": int(pred),
                    "RiskTier": tier, "RecommendedAction": action, "ActualFailure": None,
                }
                for p, pred, tier, action in zip(probabilities, predictions, tiers, actions)
            ]
            append_to_history(log_rows)
            st.caption(f"Logged {len(log_rows)} predictions to app/prediction_history.csv")

# ---------------------------------------------------------------------------
# Tab 3 - monitoring
# ---------------------------------------------------------------------------
with tab_monitor:
    st.subheader("Prediction history & basic monitoring")
    history = load_history()

    if len(history) == 0:
        st.info("No predictions logged yet - use the Single Machine or Batch tabs first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total predictions logged", len(history))
        col2.metric("Predicted failures", int(history["PredictedFailure"].sum()))
        alert_rate = history["PredictedFailure"].mean()
        col3.metric("Alert rate", f"{alert_rate:.1%}")

        history["PredictionDate"] = pd.to_datetime(history["PredictionTime"]).dt.date
        daily = history.groupby("PredictionDate")["PredictedFailure"].agg(["count", "sum"])
        daily.columns = ["Total predictions", "Predicted failures"]

        st.write("**Predictions over time**")
        st.line_chart(daily)

        st.write("**Risk tier breakdown**")
        st.bar_chart(history["RiskTier"].value_counts())

        st.write("**Full log**")
        st.dataframe(history, use_container_width=True)

        st.caption(
            "This is a lightweight, self-contained monitor (a CSV log). The columns "
            "match the `PredictionHistory` SQL table 1-for-1, so this file can be bulk-"
            "loaded into SQL Server later without any reshaping."
        )
