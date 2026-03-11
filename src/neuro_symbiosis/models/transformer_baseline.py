from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBaselineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, embed_dim, kernel_size=5, padding=2)
        self.norm = nn.BatchNorm1d(embed_dim)

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
        # x: [B, C, T] -> [B, T, D]
        h = self.norm(self.input_proj(x)).transpose(1, 2)
        z = self.transformer(h)
        logits = self.head(z.mean(dim=1))
        spike_rate = torch.zeros((), device=x.device, dtype=x.dtype)
        return logits, spike_rate
