#!/usr/bin/env python3
# train.py

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Columns / feature choices
# -----------------------------
TARGET_COL = "Survived"

NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked", "Pclass"]  # keep Pclass as categorical signal


# -----------------------------
# Data loading + cleaning
# -----------------------------
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop columns you decided to remove (safe even if missing)
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Basic imputations (deterministic)
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if "Embarked" in df.columns:
        mode = df["Embarked"].mode()
        df["Embarked"] = df["Embarked"].fillna(mode.iloc[0] if len(mode) else "S")

    # Ensure types
    df["Pclass"] = df["Pclass"].astype(str)   # treat as categorical
    df["Sex"] = df["Sex"].astype(str)
    df["Embarked"] = df["Embarked"].astype(str)

    # Keep only needed columns + target (if exists)
    keep = [c for c in (NUM_COLS + CAT_COLS + [TARGET_COL]) if c in df.columns]
    df = df[keep].copy()

    return df


def df_to_model_inputs(df: pd.DataFrame) -> dict:
    """Convert a feature dataframe into Keras input dict."""
    inputs = {}
    for c in NUM_COLS:
        inputs[c] = df[c].astype("float32").to_numpy()
    for c in CAT_COLS:
        inputs[c] = df[c].astype(str).to_numpy()
    return inputs


# -----------------------------
# Model building (preprocessing inside model)
# -----------------------------
def make_activation(name: str):
    name = name.lower()
    if name == "relu":
        return layers.Activation("relu")
    if name == "gelu":
        return layers.Activation(tf.nn.gelu)
    if name == "leakyrelu":
        return layers.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unknown activation: {name}. Choose from: relu, gelu, leakyrelu")


def build_model(
    X_train_df: pd.DataFrame,
    hidden_units=(128, 64),
    dropout=0.3,
    batch_norm=True,
    activation="gelu",
    lr=1e-3,
    seed=42,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # ---- Inputs
    inputs = {}
    for c in NUM_COLS:
        inputs[c] = keras.Input(shape=(), name=c, dtype=tf.float32)
    for c in CAT_COLS:
        inputs[c] = keras.Input(shape=(), name=c, dtype=tf.string)

    # ---- Numeric preprocessing
    num_stack = layers.Concatenate(name="num_concat")([inputs[c] for c in NUM_COLS])
    normalizer = layers.Normalization(name="num_norm")
    normalizer.adapt(X_train_df[NUM_COLS].astype("float32").to_numpy())
    num_encoded = normalizer(num_stack)

    # ---- Categorical preprocessing
    cat_encoded_parts = []
    for c in CAT_COLS:
        lookup = layers.StringLookup(output_mode="int", name=f"{c}_lookup")
        lookup.adapt(X_train_df[c].astype(str).to_numpy())

        ids = lookup(inputs[c])
        onehot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size(), output_mode="one_hot", name=f"{c}_onehot")(ids)
        cat_encoded_parts.append(onehot)

    cat_encoded = layers.Concatenate(name="cat_concat")(cat_encoded_parts)

    # ---- Combine
    x = layers.Concatenate(name="features")([num_encoded, cat_encoded])

    # ---- MLP
    act = make_activation(activation)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = act(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"drop_{i}")(x)

    out = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=inputs, outputs=out, name="titanic_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    return model


# -----------------------------
# Train + evaluate
# -----------------------------
def eval_on_split(model, X_df, y, threshold=0.5):
    probs = model.predict(df_to_model_inputs(X_df), verbose=0).ravel()
    y_hat = (probs >= threshold).astype(int)

    f1 = float(f1_score(y, y_hat))
    acc = float(accuracy_score(y, y_hat))
    prec = float(precision_score(y, y_hat, zero_division=0))
    rec = float(recall_score(y, y_hat, zero_division=0))
    cm = confusion_matrix(y, y_hat).tolist()

    return {"f1": f1, "accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV (train dataset with Survived column).")
    p.add_argument("--out_dir", default="artifacts", help="Directory to save artifacts.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--test_size", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=10)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--activation", choices=["relu", "gelu", "leakyrelu"], default="gelu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_units", type=int, nargs="+", default=[128, 64])
    p.add_argument("--no_batch_norm", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load/clean
    df = load_and_clean(data_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV must contain target column '{TARGET_COL}' for training/evaluation.")

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    # 2) Train/val/test split (held-out test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=args.test_size + args.val_size,
        random_state=args.seed,
        stratify=y,
    )
    relative_test = args.test_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=relative_test,
        random_state=args.seed,
        stratify=y_tmp,
    )

    # 3) Build model (preprocessing adapts ONLY on train split)
    model = build_model(
        X_train_df=X_train,
        hidden_units=tuple(args.hidden_units),
        dropout=args.dropout,
        batch_norm=(not args.no_batch_norm),
        activation=args.activation,
        lr=args.lr,
        seed=args.seed,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]

    # 4) Train
    model.fit(
        df_to_model_inputs(X_train),
        y_train,
        validation_data=(df_to_model_inputs(X_val), y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # 5) Evaluate on held-out test
    test_metrics = eval_on_split(model, X_test, y_test, threshold=args.threshold)

    # 6) Save artifacts
    model_path = out_dir / "model.keras"
    metrics_path = out_dir / "metrics.json"

    model.save(model_path)

    payload = {
        "data_path": str(data_path),
        "n_rows": int(df.shape[0]),
        "splits": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "features": {"num": NUM_COLS, "cat": CAT_COLS},
        "hyperparams": {
            "seed": args.seed,
            "lr": args.lr,
            "activation": args.activation,
            "dropout": args.dropout,
            "hidden_units": args.hidden_units,
            "batch_norm": (not args.no_batch_norm),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "threshold": args.threshold,
        },
        "test_metrics": test_metrics,
    }

    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved model to:   {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print("\nHeld-out TEST metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
