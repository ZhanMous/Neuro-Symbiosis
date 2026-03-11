from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


@dataclass
class EEGShape:
    channels: int
    time_steps: int
    classes: int


class SyntheticMotorImageryDataset(Dataset):
    """Generate EEG-like class-conditioned sinusoids with noise.

    Shape per sample: [channels, time_steps].
    """

    def __init__(
        self,
        num_samples: int,
        channels: int,
        time_steps: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)

        self.x = np.zeros((num_samples, channels, time_steps), dtype=np.float32)
        self.y = np.zeros((num_samples,), dtype=np.int64)

        base_t = np.linspace(0.0, 2.0 * math.pi, time_steps, endpoint=False)
        class_freqs = np.linspace(6.0, 18.0, num_classes)

        for i in range(num_samples):
            label = int(rng.integers(0, num_classes))
            self.y[i] = label

            freq = class_freqs[label]
            phase = rng.uniform(0.0, 2.0 * math.pi)

            for ch in range(channels):
                ch_scale = 1.0 + 0.05 * rng.standard_normal()
                ch_phase = phase + 0.1 * ch
                signal = ch_scale * np.sin(freq * base_t + ch_phase)

                # Add class-dependent burst around center as a weak discriminative cue.
                center = int(time_steps * (0.4 + 0.1 * (label % 2)))
                width = max(4, time_steps // 20)
                window = np.exp(-0.5 * ((np.arange(time_steps) - center) / width) ** 2)
                burst = (0.2 + 0.1 * label) * window

                noise = 0.3 * rng.standard_normal(time_steps)
                self.x[i, ch] = signal + burst + noise

        self.x = self.x.astype(np.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def build_train_val_split(dataset: Dataset, train_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)
