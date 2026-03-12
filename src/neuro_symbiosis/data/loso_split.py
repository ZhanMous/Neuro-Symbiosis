"""Leave-One-Subject-Out (LOSO) cross-validation data split utilities.

Usage
-----
>>> from neuro_symbiosis.data.loso_split import loso_splits, loso_summary
>>> for fold_idx, (train_ds, val_ds, held_out_subject) in enumerate(loso_splits(x, y, subjects)):
...     # train on train_ds, evaluate on val_ds
...     pass
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


def loso_splits(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> Iterator[tuple[Dataset, Dataset, int]]:
    """Yield (train_dataset, val_dataset, held_out_subject) for each unique subject.

    Parameters
    ----------
    x : np.ndarray
        EEG data of shape ``(N, C, T)``.
    y : np.ndarray
        Integer class labels of shape ``(N,)``.
    subjects : np.ndarray
        Integer subject IDs of shape ``(N,)``.  The subject held out in each
        fold is used as the validation set; the remaining subjects form the
        training set.

    Yields
    ------
    train_dataset : TensorDataset
    val_dataset   : TensorDataset
    held_out      : int
        Subject ID that was withheld in this fold.
    """
    unique_subjects = np.unique(subjects)
    for held_out in unique_subjects:
        train_mask = subjects != held_out
        val_mask = subjects == held_out

        x_train = torch.from_numpy(x[train_mask]).float()
        y_train = torch.from_numpy(y[train_mask]).long()
        x_val = torch.from_numpy(x[val_mask]).float()
        y_val = torch.from_numpy(y[val_mask]).long()

        yield TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), int(held_out)


def loso_summary(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
) -> dict:
    """Return a plain-dict summary of the LOSO structure (no data copied).

    Returns a dict with keys:
    - ``n_subjects``     : total number of unique subjects
    - ``n_classes``      : number of unique class labels
    - ``total_samples``  : N
    - ``per_subject``    : list of {subject, n_samples, class_counts} dicts
    """
    unique_subjects = np.unique(subjects)
    unique_classes = np.unique(y)

    per_subject = []
    for sid in unique_subjects:
        mask = subjects == sid
        class_counts = {int(c): int((y[mask] == c).sum()) for c in unique_classes}
        per_subject.append(
            {
                "subject": int(sid),
                "n_samples": int(mask.sum()),
                "class_counts": class_counts,
            }
        )

    return {
        "n_subjects": int(len(unique_subjects)),
        "n_classes": int(len(unique_classes)),
        "total_samples": int(len(y)),
        "per_subject": per_subject,
    }


def synthetic_loso_data(
    n_subjects: int = 9,
    n_trials_per_subject: int = 288,
    n_channels: int = 22,
    n_time: int = 256,
    n_classes: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic BCI-style data with subject labels for LOSO testing.

    Returns
    -------
    x : (N, C, T) float32
    y : (N,) int64
    subjects : (N,) int64
    """
    rng = np.random.default_rng(seed)
    n_total = n_subjects * n_trials_per_subject

    x = rng.standard_normal((n_total, n_channels, n_time)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n_total)
    subjects = np.repeat(np.arange(1, n_subjects + 1), n_trials_per_subject)
    return x, y, subjects


if __name__ == "__main__":
    import json

    x, y, subjects = synthetic_loso_data()
    summary = loso_summary(x, y, subjects)
    print("LOSO Summary:")
    print(json.dumps(summary, indent=2))

    print(f"\nRunning {summary['n_subjects']} LOSO folds...")
    for fold_idx, (train_ds, val_ds, held_out) in enumerate(loso_splits(x, y, subjects)):
        print(
            f"  Fold {fold_idx + 1}: held-out subject={held_out:2d} | "
            f"train={len(train_ds):4d} | val={len(val_ds):4d}"
        )
    print("Done.")
