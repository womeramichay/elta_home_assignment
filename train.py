import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


# Data / preprocessing helpers
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]
TARGET_COL = "Survived"

NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked"]


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Drop columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=c)

    # Basic missing handling
    # (keep it explicit; preprocessing layers don't impute by themselves)
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].fillna(df["Sex"].mode()[0])

    # Ensure required columns exist
    required = set(NUM_COLS + CAT_COLS + [TARGET_COL])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    return df


def df_to_model_inputs(df_features: pd.DataFrame) -> dict:
    """Convert a dataframe of features to Keras inputs dict."""
    out = {}
    for c in NUM_COLS:
        out[c] = df_features[c].astype("float32").to_numpy().reshape(-1, 1)
    for c in CAT_COLS:
        out[c] = df_features[c].astype(str).to_numpy().reshape(-1, 1)
    return out


# -----------------------------
# Model
# -----------------------------
def build_model(
    *,
    normalizer: layers.Normalization,
    sex_lookup: layers.StringLookup,
    emb_lookup: layers.StringLookup,
    lr: float = 0.003,
    hidden_units=(64,),
    activation: str = "gelu",
    dropout: float = 0.0,
    batch_norm: bool = True,
) -> keras.Model:
    # Inputs
    inputs = {}
    for c in NUM_COLS:
        inputs[c] = keras.Input(shape=(1,), name=c, dtype=tf.float32)
    for c in CAT_COLS:
        inputs[c] = keras.Input(shape=(1,), name=c, dtype=tf.string)

    # Numeric branch
    num_stack = layers.Concatenate(name="num_concat")([inputs[c] for c in NUM_COLS])
    num_norm = normalizer(num_stack)

    # Categorical branch (one-hot via StringLookup -> CategoryEncoding)
    sex_ids = sex_lookup(inputs["Sex"])
    emb_ids = emb_lookup(inputs["Embarked"])

    sex_oh = layers.CategoryEncoding(
        num_tokens=sex_lookup.vocabulary_size(), output_mode="one_hot", name="sex_onehot"
    )(sex_ids)
    emb_oh = layers.CategoryEncoding(
        num_tokens=emb_lookup.vocabulary_size(), output_mode="one_hot", name="emb_onehot"
    )(emb_ids)

    # Combine
    x = layers.Concatenate(name="features_concat")([num_norm, sex_oh, emb_oh])

    # MLP
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        if activation.lower() == "gelu":
            x = layers.Activation(tf.nn.gelu, name=f"act_{i}")(x)
        elif activation.lower() == "relu":
            x = layers.Activation("relu", name=f"act_{i}")(x)
        elif activation.lower() in ("leakyrelu", "leaky_relu"):
            x = layers.LeakyReLU(negative_slope=0.1, name=f"act_{i}")(x)
        else:
            raise ValueError("activation must be one of: relu, gelu, leakyrelu")

        if dropout and dropout > 0:
            x = layers.Dropout(dropout, name=f"dropout_{i}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="survival_prob")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="titanic_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[],
    )
    return model


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC AUC only if both classes exist in y_true
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


# -----------------------------
# Main
# -----------------------------
def main(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(data_path)

    X = df[NUM_COLS + CAT_COLS].copy()
    y = df[TARGET_COL].astype(int).to_numpy()

    # Train/val/test split (held-out test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y
    )
    relative_test = args.test_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, random_state=args.seed, stratify=y_tmp
    )

    # Build & adapt preprocessing layers on TRAIN only (no leakage)
    normalizer = layers.Normalization(name="num_normalization")
    normalizer.adapt(X_train[NUM_COLS].astype("float32").to_numpy())

    sex_lookup = layers.StringLookup(output_mode="int", name="sex_lookup")
    sex_lookup.adapt(X_train["Sex"].astype(str).to_numpy())

    emb_lookup = layers.StringLookup(output_mode="int", name="emb_lookup")
    emb_lookup.adapt(X_train["Embarked"].astype(str).to_numpy())

    model = build_model(
        normalizer=normalizer,
        sex_lookup=sex_lookup,
        emb_lookup=emb_lookup,
        lr=args.lr,
        hidden_units=tuple(args.hidden_units),
        activation=args.activation,
        dropout=args.dropout,
        batch_norm=not args.no_batch_norm,
    )

    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        )
    ]

    train_inputs = df_to_model_inputs(X_train)
    val_inputs = df_to_model_inputs(X_val)

    model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    # Evaluate on held-out TEST
    test_inputs = df_to_model_inputs(X_test)
    y_prob_test = model.predict(test_inputs, verbose=0).ravel()
    metrics = evaluate_binary(y_test, y_prob_test, threshold=args.threshold)

    print("\n=== Held-out TEST metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save model
    model.save(out_path)
    print(f"\nSaved model to: {out_path}")

    # Save metrics (json)
    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": str(data_path),
                "model_path": str(out_path),
                "seed": args.seed,
                "splits": {"val_size": args.val_size, "test_size": args.test_size},
                "hyperparams": {
                    "optimizer": "adam",
                    "lr": args.lr,
                    "hidden_units": list(args.hidden_units),
                    "activation": args.activation,
                    "dropout": args.dropout,
                    "batch_norm": not args.no_batch_norm,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "threshold": args.threshold,
                },
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training CSV (must include Survived)")
    ap.add_argument("--out", default="artifacts/model.keras", help="Output path for saved model")
    ap.add_argument("--seed", type=int, default=42)

    # splits
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--test_size", type=float, default=0.2)

    # training
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.5)

    # best-found hyperparams (from your coarse grid)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu", "leakyrelu"])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden_units", type=int, nargs="+", default=[64])
    ap.add_argument("--no_batch_norm", action="store_true")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
