import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import matplotlib.pyplot as plt


# Must match train.py
TARGET_COL = "Survived"
NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked"]


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror train.py preprocessing (simple imputation + required columns check)."""
    df = df.copy()

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    required = set(NUM_COLS + CAT_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    return df


def df_to_model_inputs(df: pd.DataFrame) -> dict:
    """Build model inputs dict like train.py expects (numeric float32, cat string)."""
    X = {}
    for c in NUM_COLS:
        X[c] = df[c].astype("float32").to_numpy().reshape(-1, 1)
    for c in CAT_COLS:
        # Keras StringLookup expects strings
        X[c] = df[c].astype(str).to_numpy().reshape(-1, 1)
    return X


@st.cache_resource
def load_model_cached(model_path: str):
    return tf.keras.models.load_model(model_path)


@st.cache_data
def read_csv_cached(csv_path: str):
    return pd.read_csv(csv_path)


def eval_and_plot(y_true: np.ndarray, prob: np.ndarray, threshold: float):
    y_pred = (prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # ROC-AUC needs both classes present
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, prob)
    else:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred)

    # Confusion matrix plot
    fig_cm, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig_cm.tight_layout()

    # ROC curve plot
    fig_roc = None
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, prob)
        fig_roc, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_title("ROC Curve")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        fig_roc.tight_layout()

    # Prob histogram plot
    fig_hist, ax3 = plt.subplots()
    ax3.hist(prob[y_true == 0], bins=25, alpha=0.7, label="True 0")
    ax3.hist(prob[y_true == 1], bins=25, alpha=0.7, label="True 1")
    ax3.axvline(threshold, linestyle="--")
    ax3.set_title("Predicted Probability Histogram")
    ax3.set_xlabel("P(Survived=1)")
    ax3.set_ylabel("Count")
    ax3.legend()
    fig_hist.tight_layout()

    return metrics, fig_cm, fig_roc, fig_hist


def main():
    st.set_page_config(page_title="Titanic Inference & Evaluation", layout="wide")
    st.title("Titanic — Inference + Held-out Test Evaluation")

    st.markdown(
        """
This app loads:
- a **trained model** (default: `artifacts/model.keras`)
- a **test CSV** (must include `Survived`)

Then it runs inference and shows metrics + plots.
"""
    )

    with st.sidebar:
        st.header("Inputs")

        model_path = st.text_input("Model path", value="artifacts/model.keras")
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.01)

        st.divider()
        st.subheader("Test dataset (CSV)")

        csv_path = st.text_input("CSV path (on disk)", value="")
        uploaded = st.file_uploader("Or upload CSV", type=["csv"])

    # Load CSV
    df = None
    load_error = None

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            load_error = f"Failed reading uploaded CSV: {e}"
    elif csv_path.strip():
        try:
            if not os.path.exists(csv_path.strip()):
                load_error = f"CSV path not found: {csv_path}"
            else:
                df = read_csv_cached(csv_path.strip())
        except Exception as e:
            load_error = f"Failed reading CSV path: {e}"
    else:
        st.info("Provide a CSV path or upload a CSV to evaluate.")
        return

    if load_error:
        st.error(load_error)
        return

    # Load model
    try:
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            return
        model = load_model_cached(model_path)
    except Exception as e:
        st.error(f"Failed loading model: {e}")
        return

    # Preprocess like train.py
    try:
        df = make_feature_frame(df)
    except Exception as e:
        st.error(f"Dataset validation/preprocessing failed: {e}")
        st.write("Columns found:", list(df.columns))
        return

    # Build inputs + predict
    X = df_to_model_inputs(df)
    y_true = df[TARGET_COL].astype(int).to_numpy()

    prob = model.predict(X, verbose=0).reshape(-1)
    metrics, fig_cm, fig_roc, fig_hist = eval_and_plot(y_true, prob, threshold)

    # Show results
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1", f"{metrics['f1']:.4f}")
    c5.metric("ROC-AUC", "N/A" if np.isnan(metrics["roc_auc"]) else f"{metrics['roc_auc']:.4f}")

    st.subheader("Plots")
    left, right = st.columns(2)
    with left:
        st.pyplot(fig_cm)
        st.pyplot(fig_hist)
    with right:
        if fig_roc is not None:
            st.pyplot(fig_roc)
        else:
            st.info("ROC curve requires both classes to exist in y_true (0 and 1).")

    st.subheader("Prediction preview")
    preview = df.copy()
    preview["pred_proba"] = prob
    preview["pred_label"] = (prob >= threshold).astype(int)
    st.dataframe(preview.head(25))

    st.subheader("Biggest mistakes (by confidence)")
    mistakes = preview[preview["pred_label"] != preview[TARGET_COL]].copy()
    if len(mistakes) == 0:
        st.success("No mistakes at this threshold on this dataset.")
    else:
        mistakes["confidence"] = np.where(
            mistakes["pred_label"] == 1, mistakes["pred_proba"], 1 - mistakes["pred_proba"]
        )
        mistakes = mistakes.sort_values("confidence", ascending=False)
        st.dataframe(mistakes.head(25))


if __name__ == "__main__":
    main()
