from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = fs * 0.5
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    # x: [N, C, T]
    return filtfilt(b, a, x, axis=2).astype(np.float32)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    # Normalize each trial/channel along time axis.
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def main(input_npz: str, output_npz: str, fs: float, low_hz: float, high_hz: float, t_start: int, t_len: int) -> None:
    in_path = Path(input_npz)
    out_path = Path(output_npz)
    data = np.load(in_path)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64).reshape(-1)

    if x.ndim != 3:
        raise ValueError("Expected x shape [N, C, T]")

    if t_start >= 0 and t_len > 0:
        x = x[:, :, t_start : t_start + t_len]

    x = bandpass_filter(x, fs=fs, low_hz=low_hz, high_hz=high_hz)
    x = zscore_per_channel(x)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, x=x, y=y)
    print({"input": str(in_path), "output": str(out_path), "x_shape": list(x.shape)})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess BCI-IV 2a NPZ data with standard EEG steps")
    p.add_argument("--input", required=True, help="Input NPZ path containing x[N,C,T] and y[N]")
    p.add_argument("--output", required=True, help="Output preprocessed NPZ path")
    p.add_argument("--fs", type=float, default=250.0)
    p.add_argument("--low-hz", type=float, default=8.0)
    p.add_argument("--high-hz", type=float, default=30.0)
    p.add_argument("--t-start", type=int, default=0)
    p.add_argument("--t-len", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output, args.fs, args.low_hz, args.high_hz, args.t_start, args.t_len)
