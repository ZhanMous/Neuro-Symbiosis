from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuro_symbiosis.models.hybrid_decoder import NeuroSymbiosisNet


def test_hybrid_forward_shape():
    model = NeuroSymbiosisNet(
        in_channels=22,
        num_classes=4,
        snn_hidden=64,
        embed_dim=64,
        lif_beta=0.9,
        threshold=1.0,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.1,
    )
    x = torch.randn(8, 22, 128)
    logits, spike_rate = model(x)

    assert logits.shape == (8, 4)
    assert spike_rate.ndim == 0
    assert 0.0 <= float(spike_rate.item()) <= 1.0
