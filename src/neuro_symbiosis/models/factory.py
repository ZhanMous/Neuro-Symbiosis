from __future__ import annotations

import torch

from neuro_symbiosis.models.hybrid_decoder import NeuroSymbiosisNet
from neuro_symbiosis.models.snn_baseline import SNNBaselineNet
from neuro_symbiosis.models.transformer_baseline import TransformerBaselineNet


def build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    model_type = str(mcfg.get("type", "hybrid")).lower()

    if model_type == "hybrid":
        model = NeuroSymbiosisNet(
            in_channels=int(dcfg["channels"]),
            num_classes=int(dcfg["num_classes"]),
            snn_hidden=int(mcfg["snn_hidden"]),
            embed_dim=int(mcfg["embed_dim"]),
            lif_beta=float(mcfg["lif_beta"]),
            threshold=float(mcfg["threshold"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
        )
    elif model_type == "snn":
        model = SNNBaselineNet(
            in_channels=int(dcfg["channels"]),
            num_classes=int(dcfg["num_classes"]),
            snn_hidden=int(mcfg["snn_hidden"]),
            embed_dim=int(mcfg["embed_dim"]),
            lif_beta=float(mcfg["lif_beta"]),
            threshold=float(mcfg["threshold"]),
            dropout=float(mcfg["dropout"]),
        )
    elif model_type == "transformer":
        model = TransformerBaselineNet(
            in_channels=int(dcfg["channels"]),
            num_classes=int(dcfg["num_classes"]),
            embed_dim=int(mcfg["embed_dim"]),
            transformer_layers=int(mcfg["transformer_layers"]),
            transformer_heads=int(mcfg["transformer_heads"]),
            dropout=float(mcfg["dropout"]),
        )
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")

    return model.to(device)
