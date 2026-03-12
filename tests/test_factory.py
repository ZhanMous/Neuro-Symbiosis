from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuro_symbiosis.data.bci_iv_2a import load_bci_iv_2a_npz
from neuro_symbiosis.models.factory import build_model


def _base_cfg(model_type: str) -> dict:
    return {
        "seed": 1,
        "device": "cpu",
        "data": {
            "dataset": "synthetic_motor_imagery",
            "num_samples": 32,
            "num_classes": 4,
            "channels": 22,
            "time_steps": 64,
            "train_ratio": 0.8,
        },
        "model": {
            "type": model_type,
            "embed_dim": 32,
            "snn_hidden": 32,
            "lif_beta": 0.9,
            "threshold": 1.0,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 8,
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "dp_enabled": False,
        },
        "output_dir": "outputs/test",
    }


def test_model_factory_forward_all_types():
    for t in ["snn", "transformer", "hybrid"]:
        cfg = _base_cfg(t)
        model = build_model(cfg, device=torch.device("cpu"))
        x = torch.randn(4, 22, 64)
        logits, spike = model(x)
        assert logits.shape == (4, 4)
        assert spike.ndim == 0


def test_load_bci_npz_label_shift(tmp_path: Path):
    x = np.random.randn(10, 22, 256).astype(np.float32)
    y = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.int64)
    f = tmp_path / "bci.npz"
    np.savez_compressed(f, x=x, y=y)

    lx, ly = load_bci_iv_2a_npz(f)
    assert lx.shape == x.shape
    assert ly.min() == 0
    assert ly.max() == 3


def test_load_bci_npz_with_subjects(tmp_path: Path):
    x = np.random.randn(12, 22, 128).astype(np.float32)
    y = np.array([1, 2, 3, 4] * 3, dtype=np.int64)
    subjects = np.repeat(np.array([1, 2, 3]), 4)
    f = tmp_path / "bci_subjects.npz"
    np.savez_compressed(f, x=x, y=y, subjects=subjects)

    lx, ly, ls = load_bci_iv_2a_npz(f, return_subjects=True)
    assert lx.shape == x.shape
    assert ly.min() == 0
    assert ly.max() == 3
    assert ls.shape == (12,)
    assert set(ls.tolist()) == {1, 2, 3}
