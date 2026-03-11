from __future__ import annotations

import torch
import torch.nn as nn

from .snn_encoder import SNNTemporalEncoder


class SNNBaselineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        snn_hidden: int,
        embed_dim: int,
        lif_beta: float,
        threshold: float,
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
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, spike_rate = self.encoder(x)
        pooled = tokens.mean(dim=1)
        logits = self.head(pooled)
        return logits, spike_rate
