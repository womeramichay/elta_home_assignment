# titanic_model.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn


# ----------------------------
# Model
# ----------------------------
class Titanic(nn.Module):
    """
    Simple MLP for binary classification (Titanic survival).
    We will feed it a numeric feature matrix after preprocessing.
    Output: logits (not probabilities).
    """
    def __init__(self, in_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns logits shape: (batch,)
        return self.net(x).squeeze(1)


# ----------------------------
# Training config (we’ll use later)
# ----------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Save / Load (we’ll wire into train.py and ds_app.py later)
# ----------------------------
def save_model(model: nn.Module, artifacts_dir: str | Path, input_dim: int) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": int(input_dim),
    }
    out_path = artifacts_dir / "model.pt"
    torch.save(ckpt, out_path)
    return out_path


def load_model(artifacts_dir: str | Path, device: Optional[str] = None) -> nn.Module:
    artifacts_dir = Path(artifacts_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt: Dict[str, Any] = torch.load(artifacts_dir / "model.pt", map_location="cpu")
    model = TitanicMLP(in_dim=int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
