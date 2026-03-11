from __future__ import annotations

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).to(membrane.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (membrane,) = ctx.saved_tensors
        threshold = ctx.threshold
        # Fast-sigmoid style surrogate gradient.
        scale = 10.0
        s = torch.sigmoid(scale * (membrane - threshold))
        surrogate_grad = scale * s * (1.0 - s)
        grad_membrane = grad_output * surrogate_grad
        return grad_membrane, None


class LIFBlock(nn.Module):
    def __init__(self, beta: float = 0.9, threshold: float = 1.0) -> None:
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x_t: torch.Tensor, mem_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem = self.beta * mem_prev + x_t
        spk = SurrogateSpike.apply(mem, self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class SNNTemporalEncoder(nn.Module):
    """Encode EEG [B, C, T] into token sequence [B, T, D]."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        embed_dim: int,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.lif = LIFBlock(beta=beta, threshold=threshold)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T]
        h = self.norm(self.input_proj(x))  # [B, H, T]
        h = h.transpose(1, 2)  # [B, T, H]

        batch_size, time_steps, hidden_dim = h.shape
        mem = torch.zeros(batch_size, hidden_dim, device=h.device, dtype=h.dtype)

        spikes = []
        for t in range(time_steps):
            spk_t, mem = self.lif(h[:, t, :], mem)
            spikes.append(spk_t)

        spk_seq = torch.stack(spikes, dim=1)  # [B, T, H]
        tokens = self.out_proj(spk_seq)  # [B, T, D]
        spike_rate = spk_seq.mean()
        return tokens, spike_rate
