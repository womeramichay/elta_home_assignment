# ds_app.py
# Streamlit app for Titanic (PyTorch) - loads artifacts/model.pt + artifacts/preprocess.pkl
# Runs inference and (if Survived exists) shows performance metrics + plots.

import io
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


# -----------------------------
# PyTorch model (must match train.py)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units=(128, 64), dropout=0.3, batch_norm=True, activation="gelu"):
        super().__init__()
        activation = str(activation).lower()
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "leakyrelu":
            act = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        prev = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# -----------------------------
# Helpers
# -----------------------------
def load_preprocess(preprocess_path: str) -> Dict:
    with open(preprocess_path, "rb") as f:
        payload = pickle.load(f)
    # expected keys: preprocessor, num_cols, cat_cols, feature_names, target_col
    return payload


def load_model(model_path: str, device: torch.device) -> nn.Module:
    payload = torch.load(model_path, map_location=device)

    model = MLP(
        input_dim=int(payload["input_dim"]),
        hidden_units=tuple(payload["hidden_units"]),
        dropout=float(payload["dropout"]),
        batch_norm=bool(payload["batch_norm"]),
        activation=str(payload["activation"]),
    ).to(device)

    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X).float()
    probs = []
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size].to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs).reshape(-1)


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred": y_pred,
    }


def ensure_columns(df: pd.DataFrame, needed_cols) -> Tuple[bool, str]:
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Titanic Survival - PyTorch", layout="wide")
st.title("Titanic Survival Prediction (PyTorch)")

with st.sidebar:
    st.header("Artifacts")
    default_pre = "artifacts/preprocess.pkl"
    default_model = "artifacts/model.pt"
    preprocess_path = st.text_input("Preprocess path", value=default_pre)
    model_path = st.text_input("Model path", value=default_model)

    st.header("Inference")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    st.caption("Tip: If you trained in Colab, download artifacts/ and place them next to this app when running locally.")

# Device selection
if device_choice == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif device_choice == "cuda":
    if not torch.cuda.is_available():
        st.warning("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.write(f"**Using device:** `{device}`")

# Load artifacts
art_ok = True
if not Path(preprocess_path).exists():
    st.error(f"Preprocess file not found: {preprocess_path}")
    art_ok = False
if not Path(model_path).exists():
    st.error(f"Model file not found: {model_path}")
    art_ok = False

if not art_ok:
    st.stop()

try:
    pre_payload = load_preprocess(preprocess_path)
    preprocessor = pre_payload["preprocessor"]
    num_cols = pre_payload.get("num_cols", [])
    cat_cols = pre_payload.get("cat_cols", [])
    target_col = pre_payload.get("target_col", "Survived")
except Exception as e:
    st.exception(e)
    st.stop()

try:
    model = load_model(model_path, device=device)
except Exception as e:
    st.exception(e)
    st.stop()

needed_feature_cols = list(num_cols) + list(cat_cols)

st.header("Upload data")
uploaded = st.file_uploader("Upload a CSV (Titanic format)", type=["csv"])
sample_path = st.text_input("...or provide a CSV path (optional)", value="")

df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(io.BytesIO(uploaded.read()))
elif sample_path.strip():
    if Path(sample_path).exists():
        df = pd.read_csv(sample_path)
    else:
        st.warning(f"Path not found: {sample_path}")

if df is None:
    st.info("Upload a CSV to run inference. If it includes `Survived`, the app will show evaluation metrics.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

ok, msg = ensure_columns(df, needed_feature_cols)
if not ok:
    st.error(msg)
    st.stop()

# Prepare X
X_raw = df[needed_feature_cols].copy()

# Make sure categoricals are strings (robust)
for c in cat_cols:
    X_raw[c] = X_raw[c].astype(str)

try:
    X = preprocessor.transform(X_raw).astype(np.float32)
except Exception as e:
    st.error("Preprocessing transform failed. This usually means the columns/types don't match training.")
    st.exception(e)
    st.stop()

# Predict
prob = predict_proba(model, X, device=device)
pred = (prob >= threshold).astype(int)

out = df.copy()
out["pred_proba"] = prob
out["pred_label"] = pred

st.header("Predictions")
st.dataframe(out.head(50), use_container_width=True)

# If ground truth exists -> show evaluation
has_target = target_col in df.columns
if has_target:
    # Clean y
    try:
        y_true = df[target_col].astype(int).to_numpy()
    except Exception:
        st.error(f"Column `{target_col}` exists but couldn't be converted to int (0/1).")
        st.stop()

    m = compute_metrics(y_true, prob, threshold=threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.4f}")
    c2.metric("Precision", f"{m['precision']:.4f}")
    c3.metric("Recall", f"{m['recall']:.4f}")
    c4.metric("F1", f"{m['f1']:.4f}")

    st.subheader("Confusion matrix")
    cm = m["confusion_matrix"]

    import matplotlib.pyplot as plt

    fig_cm = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig_cm)

    st.subheader("ROC curve")
    try:
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)
    except Exception as e:
        st.warning("ROC curve could not be computed (needs both classes present in y).")
        st.caption(str(e))

    st.subheader("Probability histogram")
    fig_hist = plt.figure()
    plt.hist(prob, bins=30)
    plt.title("Predicted Probability Histogram")
    plt.xlabel("P(Survived)")
    plt.ylabel("Count")
    st.pyplot(fig_hist)

else:
    st.info(f"No `{target_col}` column found. Showing predictions only (no evaluation).")

# Download predictions
st.subheader("Download")
csv_bytes = out.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
