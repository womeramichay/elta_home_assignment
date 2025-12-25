#!/usr/bin/env python3
# train.py
"""
Titanic (Kaggle) end-to-end training script (PyTorch)

What this script does (per assignment):
- Fetches Titanic dataset directly from Kaggle in code
- Loads + preprocesses data (fit on train split only)
- Trains a PyTorch classifier with a SMALL grid search + early stopping
- Evaluates on a held-out test split
- Saves:
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


# -----------------------------
# Feature config (keep consistent with ds_app.py later)
# -----------------------------
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
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    device: torch.device,
    threshold: float,
    seed: int,
) -> Tuple[nn.Module, Dict]:
    set_seed(seed)

    model = MLP(
        input_dim=input_dim,
        hidden_units=cfg.hidden_units,
        dropout=cfg.dropout,
        batch_norm=cfg.batch_norm,
        activation=cfg.activation,
    ).to(device)

    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, cfg.batch_size)

    # Class imbalance: mild; optional pos_weight. We'll keep it simple.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0

    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        val_prob = predict_proba(model, X_val, device=device)
        val_stats = eval_metrics(y_val, val_prob, threshold=threshold)
        val_f1 = val_stats["f1"]

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_f1": float(val_f1),
                "val_accuracy": val_stats["accuracy"],
            }
        )

        # Early stopping on val F1
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(len(history)),
        "history_tail": history[-5:],  # keep it short
    }
    return model, summary


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--out_dir", default="artifacts", help="Where to save model + preprocessing + metrics.")
    p.add_argument("--data_dir", default="data", help="Where to download Kaggle files (train.csv/test.csv).")
    p.add_argument("--no_kaggle_download", action="store_true", help="Skip Kaggle download and expect train.csv in data_dir.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--test_size", type=float, default=0.2)

    p.add_argument("--max_epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1) Get data from Kaggle (required)
    if args.no_kaggle_download:
        train_csv = data_dir / "train.csv"
        if not train_csv.exists():
            raise FileNotFoundError(
                f"--no_kaggle_download was set but {train_csv} not found. "
                "Either place train.csv there or remove --no_kaggle_download."
            )
        test_csv = data_dir / "test.csv" if (data_dir / "test.csv").exists() else None
    else:
        train_csv, test_csv = download_kaggle_titanic(data_dir)

    # 2) Load + basic clean
    df = load_raw_train_csv(train_csv)
    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    # 3) Split train/val/test (held-out test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=args.val_size + args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    relative_test = args.test_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        random_state=args.seed,
        stratify=y_tmp,
    )

    # 4) Fit preprocessing ONLY on train split
    pre = build_preprocessor()
    pre.fit(X_train)

    X_train_t = pre.transform(X_train).astype(np.float32)
    X_val_t = pre.transform(X_val).astype(np.float32)
    X_test_t = pre.transform(X_test).astype(np.float32)

    feature_names = get_feature_names(pre)
    input_dim = X_train_t.shape[1]

    # 5) Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device cpu or --device auto.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 6) Small grid search (keep it small and sensible)
    #    You can tweak these sets based on your EDA notebook findings.
    grid_hidden = [(128, 64), (64, 32), (128, 128)]
    grid_lr = [1e-3, 3e-4]
    grid_dropout = [0.1, 0.3]
    grid_wd = [0.0, 1e-4]

    configs: List[TrainConfig] = []
    for hu in grid_hidden:
        for lr in grid_lr:
            for dr in grid_dropout:
                for wd in grid_wd:
                    configs.append(
                        TrainConfig(
                            lr=lr,
                            weight_decay=wd,
                            hidden_units=hu,
                            dropout=dr,
                            batch_norm=True,
                            activation="gelu",
                            batch_size=args.batch_size,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                        )
                    )

    print(f"Grid size: {len(configs)} configs")
    print(f"Device: {device}")
    print(f"Input dim after preprocessing: {input_dim}")

    # 7) Train all configs with early stopping; select best by val F1
    best = {
        "val_f1": -1.0,
        "cfg": None,
        "model": None,
        "train_summary": None,
    }

    for i, cfg in enumerate(configs, start=1):
        print(
            f"\n[{i}/{len(configs)}] "
            f"hu={cfg.hidden_units} lr={cfg.lr} wd={cfg.weight_decay} drop={cfg.dropout}"
        )

        model, summary = train_one_config(
            cfg=cfg,
            X_train=X_train_t, y_train=y_train,
            X_val=X_val_t, y_val=y_val,
            input_dim=input_dim,
            device=device,
            threshold=args.threshold,
            seed=args.seed,
        )

        val_f1 = summary["best_val_f1"]
        print(f"  best_val_f1={val_f1:.4f} @epoch={summary['best_epoch']} (ran {summary['epochs_ran']})")

        if val_f1 > best["val_f1"] + 1e-6:
            best.update({"val_f1": val_f1, "cfg": cfg, "model": model, "train_summary": summary})

    if best["model"] is None:
        raise RuntimeError("No model was trained successfully.")

    best_model: nn.Module = best["model"]
    best_cfg: TrainConfig = best["cfg"]

    # 8) Evaluate on held-out test
    test_prob = predict_proba(best_model, X_test_t, device=device)
    test_metrics = eval_metrics(y_test, test_prob, threshold=args.threshold)

    print("\n=== BEST CONFIG ===")
    print(json.dumps({**asdict(best_cfg), "best_val_f1": best["val_f1"]}, indent=2))
    print("\n=== HELD-OUT TEST METRICS ===")
    print(json.dumps(test_metrics, indent=2))

    # 9) Save artifacts
    preprocess_path = out_dir / "preprocess.pkl"
    model_path = out_dir / "model.pt"
    metrics_path = out_dir / "metrics.json"

    # Save preprocessing + feature names
    preprocess_payload = {
        "preprocessor": pre,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "feature_names": feature_names,
        "target_col": TARGET_COL,
    }
    with open(preprocess_path, "wb") as f:
        pickle.dump(preprocess_payload, f)

    # Save PyTorch state_dict + minimal metadata to reconstruct
    torch_payload = {
        "state_dict": best_model.state_dict(),
        "input_dim": int(input_dim),
        "hidden_units": tuple(best_cfg.hidden_units),
        "dropout": float(best_cfg.dropout),
        "batch_norm": bool(best_cfg.batch_norm),
        "activation": str(best_cfg.activation),
    }
    torch.save(torch_payload, model_path)

    # Save training metadata + metrics
    full_metrics = {
        "data": {
            "train_csv": str(train_csv),
            "test_csv": str(test_csv) if test_csv else None,
            "rows_total": int(len(df)),
            "splits": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
            "features": {"num": NUM_COLS, "cat": CAT_COLS},
        },
        "best_config": {**asdict(best_cfg), "best_val_f1": float(best["val_f1"])},
        "train_summary": best["train_summary"],
        "threshold": float(args.threshold),
        "heldout_test_metrics": test_metrics,
    }
    metrics_path.write_text(json.dumps(full_metrics, indent=2))

    print(f"\nSaved preprocessing to: {preprocess_path}")
    print(f"Saved model to:        {model_path}")
    print(f"Saved metrics to:      {metrics_path}")


if __name__ == "__main__":
    main()
