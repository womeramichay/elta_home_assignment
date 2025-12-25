#!/usr/bin/env python3
"""
train.py  (PyTorch)

What this script does (assignment-friendly):
- Downloads Titanic data from Kaggle *in code* (default) using Kaggle CLI.
- Builds a simple, configurable MLP classifier in **PyTorch**.
- Splits into train/val/test, trains with early stopping, evaluates on held-out test.
- Saves:
  - artifacts/model.pt  (state_dict + architecture params)
  - artifacts/preprocess.json (feature columns + imputation values)
  - artifacts/metrics.json
  - artifacts/confusion_matrix.png
  - artifacts/roc_curve.png  (if possible)

Requirements:
- Kaggle API configured in the environment (kaggle.json present) OR pass --data to a local CSV.
  In Colab: upload kaggle.json to ~/.kaggle/kaggle.json and chmod 600 it.

Example:
  python train.py --out_dir artifacts

Or with a local CSV:
  python train.py --data /path/to/train.csv --out_dir artifacts
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Kaggle download
# ----------------------------
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stdout}")


def fetch_titanic_from_kaggle(work_dir: Path) -> Path:
    """
    Download Titanic competition files using Kaggle CLI and return path to train.csv.
    Requires kaggle CLI + kaggle.json configured.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    zip_path = work_dir / "titanic.zip"
    # Download competition data
    _run(["kaggle", "competitions", "download", "-c", "titanic", "-p", str(work_dir)])

    # Kaggle typically downloads titanic.zip
    # Find the newest zip in folder just in case filename differs.
    zips = sorted(work_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip found in {work_dir}. Kaggle download may have failed.")
    zip_path = zips[0]

    # Unzip
    unzip_dir = work_dir / "titanic_data"
    if unzip_dir.exists():
        shutil.rmtree(unzip_dir)
    unzip_dir.mkdir(parents=True, exist_ok=True)

    shutil.unpack_archive(str(zip_path), str(unzip_dir))

    train_csv = unzip_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected {train_csv} not found after unzip. Contents: {list(unzip_dir.iterdir())}")
    return train_csv


# ----------------------------
# Preprocessing
# ----------------------------
@dataclass
class PreprocessArtifacts:
    feature_cols: List[str]
    age_median: float
    embarked_mode: str


def preprocess_fit_transform(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, PreprocessArtifacts]:
    """
    Fit preprocessing on the provided dataframe and return (X, y, artifacts).
    - Drops: PassengerId, Name, Ticket, Cabin
    - Imputes: Age (median), Embarked (mode)
    - One-hot: Sex, Embarked
    - Keeps: Pclass, Age, SibSp, Parch, Fare, Sex_*, Embarked_*
    """
    df = df.copy()

    # Target
    if "Survived" not in df.columns:
        raise ValueError("Expected 'Survived' column in training data.")
    y = df["Survived"].astype(int)

    # Drop columns (high-cardinality / missing-heavy / non-learnable)
    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Impute
    age_median = float(df["Age"].median()) if "Age" in df.columns else 0.0
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(age_median)

    embarked_mode = "S"
    if "Embarked" in df.columns:
        embarked_mode = str(df["Embarked"].mode(dropna=True)[0]) if df["Embarked"].notna().any() else "S"
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    # Basic fill for Fare if needed
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(float(df["Fare"].median()))

    # One-hot encode categoricals
    cat_cols = [c for c in ["Sex", "Embarked"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Build feature list
    feature_cols = []
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        if col in df.columns:
            feature_cols.append(col)

    # Add dummies
    for col in df.columns:
        if col.startswith("Sex_") or col.startswith("Embarked_"):
            feature_cols.append(col)

    # Remove target from features if present
    if "Survived" in feature_cols:
        feature_cols.remove("Survived")

    X = df[feature_cols].astype(np.float32)

    artifacts = PreprocessArtifacts(
        feature_cols=feature_cols,
        age_median=age_median,
        embarked_mode=embarked_mode,
    )
    return X, y, artifacts


def preprocess_transform(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """
    Apply preprocessing using saved artifacts (for inference).
    Produces columns exactly as artifacts.feature_cols.
    """
    df = df.copy()

    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(artifacts.age_median)
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(float(df["Fare"].median()))
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(artifacts.embarked_mode)

    cat_cols = [c for c in ["Sex", "Embarked"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure all expected columns exist
    for col in artifacts.feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[artifacts.feature_cols].astype(np.float32)
    return X


# ----------------------------
# Model
# ----------------------------
def make_activation(name: str, negative_slope: float = 0.01) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(negative_slope=negative_slope)
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_units: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim

        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        # Binary classification head (logit)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [B]


# ----------------------------
# Training utilities
# ----------------------------
def get_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def train_one_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    hidden_units: List[int],
    activation: str,
    dropout: float,
    batch_norm: bool,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
) -> Tuple[MLP, Dict[str, float]]:
    set_seed(seed)

    model = MLP(
        in_dim=X_train.shape[1],
        hidden_units=hidden_units,
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(device)

    opt = get_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))

        # val
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_losses.append(float(loss.item()))
        val_loss = float(np.mean(va_losses))

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # quick val F1
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X_val).to(device))).cpu().numpy()
    y_hat = (p >= 0.5).astype(int)
    val_f1 = float(f1_score(y_val.astype(int), y_hat))

    return model, {"best_val_loss": float(best_val), "val_f1": val_f1}


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X).to(device))).cpu().numpy()
    preds = (probs >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
    }

    # ROC AUC (only if both classes present)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    except Exception:
        metrics["roc_auc"] = float("nan")

    return metrics


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    # requires both classes
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Optional path to Titanic train.csv. If omitted, downloads from Kaggle.")
    p.add_argument("--out_dir", type=str, default="artifacts", help="Where to save model and metrics.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--test_size", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "leakyrelu"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden_units", type=int, nargs="+", default=[64], help="Example: --hidden_units 256 128 64")
    p.add_argument("--no_batch_norm", action="store_true", help="Disable batch norm")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    if args.data is not None:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"--data not found: {data_path}")
    else:
        # Fetch from Kaggle in code (default)
        cache_dir = out_dir / "_kaggle_cache"
        data_path = fetch_titanic_from_kaggle(cache_dir)

    df = pd.read_csv(data_path)

    # 2) Preprocess
    X_df, y_s, prep = preprocess_fit_transform(df)
    X = X_df.values.astype(np.float32)
    y = y_s.values.astype(np.float32)

    # 3) Split: train / val / test (stratified)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y.astype(int)
    )
    val_relative = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, random_state=args.seed, stratify=y_tmp.astype(int)
    )

    # 4) Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_info = train_one_run(
        X_train, y_train,
        X_val, y_val,
        seed=args.seed,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_units=list(args.hidden_units),
        activation=args.activation,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=device,
    )

    # 5) Evaluate on held-out test
    model.eval()
    with torch.no_grad():
        probs_test = torch.sigmoid(model(torch.from_numpy(X_test).to(device))).cpu().numpy()
    preds_test = (probs_test >= args.threshold).astype(int)

    test_metrics = evaluate(model, X_test, y_test.astype(int), threshold=args.threshold, device=device)

    # 6) Save artifacts (model + preprocess + metrics + plots)
    model_payload = {
        "state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
        "hidden_units": list(args.hidden_units),
        "activation": args.activation,
        "dropout": float(args.dropout),
        "batch_norm": bool(not args.no_batch_norm),
    }
    torch.save(model_payload, out_dir / "model.pt")

    with open(out_dir / "preprocess.json", "w", encoding="utf-8") as f:
        json.dump(asdict(prep), f, indent=2)

    metrics_payload = {
        "device": str(device),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "split": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "train_info": train_info,
        "test_metrics": test_metrics,
        "params": {
            "seed": args.seed,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "activation": args.activation,
            "dropout": args.dropout,
            "hidden_units": list(args.hidden_units),
            "batch_norm": bool(not args.no_batch_norm),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "threshold": args.threshold,
            "val_size": args.val_size,
            "test_size": args.test_size,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    plot_confusion(y_test.astype(int), preds_test, out_dir / "confusion_matrix.png")
    plot_roc(y_test.astype(int), probs_test, out_dir / "roc_curve.png")

    # Print summary (nice for logs)
    print("✅ Training complete")
    print(f"Saved model: {out_dir / 'model.pt'}")
    print(f"Saved preprocess: {out_dir / 'preprocess.json'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
