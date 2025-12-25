#!/usr/bin/env python3
# train.py
"""
Fetches Titanic dataset directly from Kaggle in code
Loads + preprocesses data (fit on train split only)
Trains a PyTorch classifier with a SMALL grid search + early stopping
Evaluates on a held-out test split
Saves:
    artifacts/model.pt          (PyTorch state_dict + model metadata)
    artifacts/preprocess.pkl    (fitted sklearn preprocessing pipeline + feature names)
    artifacts/metrics.json      (best params + test metrics + split sizes)
"""

import argparse
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Feature config 
TARGET_COL = "Survived"

# Keep it simple and strong:
# - Treat Pclass as categorical (often works better than numeric)
NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic may reduce speed slightly but helps reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Kaggle download
# -----------------------------
def download_kaggle_titanic(data_dir: Path) -> Tuple[Path, Optional[Path]]:
    """
    Downloads Titanic competition files into data_dir using Kaggle API.

    Requirements:
    - kaggle package installed (pip install kaggle)
    - credentials set via:
        ~/.kaggle/kaggle.json  OR  env vars KAGGLE_USERNAME / KAGGLE_KEY
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    # If already present, reuse (reproducible, faster)
    if train_path.exists():
        return train_path, test_path if test_path.exists() else None

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "Kaggle API not available. Install it with: pip install kaggle\n"
            "and ensure you have credentials (~/.kaggle/kaggle.json or env vars)."
        ) from e

    api = KaggleApi()
    api.authenticate()

    # Download competition files (zip) then unzip
    api.competition_download_files("titanic", path=str(data_dir), quiet=False)
    zip_path = data_dir / "titanic.zip"
    if not zip_path.exists():
        # Kaggle sometimes names it differently; try to find any zip
        zips = list(data_dir.glob("*.zip"))
        if not zips:
            raise RuntimeError("Failed to download titanic zip from Kaggle.")
        zip_path = zips[0]

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=str(data_dir))

    # Clean up zip (optional)
    try:
        zip_path.unlink()
    except Exception:
        pass

    if not train_path.exists():
        raise RuntimeError(f"Expected {train_path} after download, but it was not found.")

    return train_path, test_path if test_path.exists() else None


# -----------------------------
# Data loading + light cleaning
# -----------------------------
def load_raw_train_csv(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    # Minimal sanity checks
    required = set(NUM_COLS + CAT_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing)}")

    # Keep only what we need (less chance of leakage / mismatch)
    df = df[NUM_COLS + CAT_COLS + [TARGET_COL]].copy()

    # Ensure dtypes are safe
    for c in CAT_COLS:
        df[c] = df[c].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Fit this ONLY on train split.
    Then use the fitted object to transform val/test and later inference in ds_app.py.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    # Works for sklearn >= 1.0; if not available, we fall back to generic
    try:
        names = preprocessor.get_feature_names_out()
        return [str(x) for x in names]
    except Exception:
        # fallback
        return [f"f{i}" for i in range(preprocessor.transform(pd.DataFrame(columns=NUM_COLS + CAT_COLS)).shape[1])]


# -----------------------------
# PyTorch model
# -----------------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()

        act_layer: nn.Module
        activation = activation.lower()
        if activation == "relu":
            act_layer = nn.ReLU()
        elif activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "leakyrelu":
            act_layer = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(hidden_units):
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # (N,)


@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    hidden_units: Tuple[int, ...]
    dropout: float
    batch_norm: bool
    activation: str
    batch_size: int
    max_epochs: int
    patience: int


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xb = torch.from_numpy(X).float().to(device)
    logits = model(xb)
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    return prob.reshape(-1)


@torch.no_grad()
def eval_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_one_config(
    cfg: TrainConfig,
    X_train: np.ndarray,
