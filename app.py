"""Streamlit dashboard for evaluating the five saved classification models."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Diagnostic Model Lab", page_icon="D", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }
    h1, h2, h3 { font-family: "DM Serif Display", serif; letter-spacing: 0; }
    .stApp { background: linear-gradient(135deg, #f8faf8 0%, #edf4f1 55%, #f8f3eb 100%); }
    [data-testid="stSidebar"] { background: #173d36; color: #f7fbf9; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #f7fbf9; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #f7fbf9; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #173d36; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] {
        background: #f7fbf9;
        border-color: #b8cbc5;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="group"] * { color: #173d36; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background: #f7fbf9; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #173d36; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background: #c24f32;
        border-color: #c24f32;
        color: #ffffff;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color: inherit; }
    [data-testid="stSidebar"] .stDownloadButton button {
        background: #f7fbf9;
        border-color: #b8cbc5;
        color: #173d36;
    }
    [data-testid="stSidebar"] .stDownloadButton button * { color: inherit; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid #dbe5e1; border-radius: 6px; padding: 14px; }
    .eyebrow { color: #c24f32; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .subtitle { color: #47615b; max-width: 760px; font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metadata():
    with (MODEL_DIR / "metadata.json").open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def calculate_metrics(y_true, y_pred, y_score):
    auc = roc_auc_score(y_true, y_score) if y_true.nunique() > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def evaluate(model, features, target):
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    else:
        scores = model.decision_function(features)
    return predictions, calculate_metrics(target, predictions, scores)


def confusion_figure(target, predictions):
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    image = axis.imshow(matrix, cmap="Greens")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=18)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        xlabel="Predicted label",
        ylabel="Actual label",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


metadata = load_metadata()
feature_names = metadata["feature_names"]
target_column = metadata["target_column"]
model_files = metadata["model_files"]

st.markdown('<div class="eyebrow">BITS AIML · ML Assignment 2</div>', unsafe_allow_html=True)
st.title("Diagnostic Model Lab")
st.markdown(
    '<p class="subtitle">Compare five classification algorithms trained on the UCI Breast Cancer Wisconsin Diagnostic dataset. Upload labeled test data to reproduce every metric.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Classification model", list(model_files))
    uploaded_file = st.file_uploader("Labeled test CSV", type="csv")
    st.caption("The CSV must include all 30 feature columns and the target column.")
    with (BASE_DIR / "test_data.csv").open("rb") as file:
        st.download_button(
            "Download sample test data",
            data=file,
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

if uploaded_file is None:
    st.info("Upload test data from the sidebar. The saved hold-out results are shown below.")
    saved_results = pd.DataFrame(metadata["metrics"]).T
    st.subheader("Saved hold-out results")
    st.dataframe(
        saved_results.style.format("{:.4f}").highlight_max(axis=0, color="#cce5d8"),
        width="stretch",
    )
    winner = saved_results["F1"].idxmax()
    st.success(f"Highest hold-out F1 score: {winner} ({saved_results.loc[winner, 'F1']:.4f})")
    st.stop()

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"The uploaded file could not be read: {error}")
    st.stop()

required_columns = feature_names + [target_column]
missing_columns = [column for column in required_columns if column not in uploaded_data.columns]
if missing_columns:
    st.error("Missing required columns: " + ", ".join(missing_columns))
    st.stop()

features = uploaded_data[feature_names]
target = uploaded_data[target_column]
if target.isna().any() or not set(target.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 (malignant) and 1 (benign), with no missing values.")
    st.stop()
if features.isna().any().any():
    st.error("Feature columns contain missing values. Provide a complete test dataset.")
    st.stop()

st.caption(f"Loaded {len(uploaded_data):,} rows and {len(feature_names)} features")
selected_pipeline = load_model(model_files[selected_model])
predictions, selected_metrics = evaluate(selected_pipeline, features, target)

st.subheader(selected_model)
metric_columns = st.columns(6)
for column, (metric_name, metric_value) in zip(metric_columns, selected_metrics.items()):
    column.metric(metric_name, "N/A" if pd.isna(metric_value) else f"{metric_value:.4f}")

left_column, right_column = st.columns([1, 1])
with left_column:
    st.subheader("Confusion matrix")
    st.pyplot(confusion_figure(target, predictions), width="stretch")
with right_column:
    st.subheader("Classification report")
    report = classification_report(
        target,
        predictions,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        zero_division=0,
    )
    st.code(report, language="text")

st.subheader("All models on uploaded data")
all_results = {}
for model_name, filename in model_files.items():
    _, all_results[model_name] = evaluate(load_model(filename), features, target)
results_frame = pd.DataFrame(all_results).T
st.dataframe(
    results_frame.style.format("{:.4f}").highlight_max(axis=0, color="#cce5d8"),
    width="stretch",
)
st.success(f"Highest F1 score on this upload: {results_frame['F1'].idxmax()}")

with st.expander("Preview uploaded rows"):
    st.dataframe(uploaded_data.head(20), width="stretch")
