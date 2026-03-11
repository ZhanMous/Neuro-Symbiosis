from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class EEGArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 3:
            raise ValueError("Expected x shape [N, C, T]")
        if y.ndim != 1:
            raise ValueError("Expected y shape [N]")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def load_bci_iv_2a_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"BCI-IV 2a npz not found: {npz_path}")

    data = np.load(npz_path)
    if "x" not in data or "y" not in data:
        raise ValueError("NPZ must contain keys 'x' and 'y'")

    x = data["x"]
    y = data["y"]

    if x.ndim != 3:
        raise ValueError("x must be [N, C, T]")
    if y.ndim != 1:
        y = y.reshape(-1)

    y = y.astype(np.int64)
    if y.min() == 1:
        y = y - 1
    return x.astype(np.float32), y


def split_bci_dataset(
    x: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    seed: int,
) -> tuple[EEGArrayDataset, EEGArrayDataset]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        train_size=train_ratio,
        random_state=seed,
        stratify=y,
    )
    return EEGArrayDataset(x_train, y_train), EEGArrayDataset(x_val, y_val)
