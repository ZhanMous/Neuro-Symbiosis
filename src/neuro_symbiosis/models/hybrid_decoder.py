from __future__ import annotations

import torch
import torch.nn as nn

from .snn_encoder import SNNTemporalEncoder


class NeuroSymbiosisNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        snn_hidden: int,
        embed_dim: int,
        lif_beta: float,
        threshold: float,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SNNTemporalEncoder(
            in_channels=in_channels,
            hidden_dim=snn_hidden,
            embed_dim=embed_dim,
            beta=lif_beta,
            threshold=threshold,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, spike_rate = self.encoder(x)
        z = self.transformer(tokens)
        pooled = z.mean(dim=1)
        logits = self.head(pooled)
        return logits, spike_rate
