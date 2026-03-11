from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.data.eeg_synthetic import SyntheticMotorImageryDataset
from neuro_symbiosis.models.factory import build_model


def query_gpu_power_watt() -> float | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        first = out.splitlines()[0]
        return float(first)
    except Exception:
        return None


def evaluate_energy_from_config(cfg: dict, num_batches: int) -> dict:
    cfg = copy.deepcopy(cfg)
    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(str(cfg.get("output_dir", "outputs/default")))
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    dataset = SyntheticMotorImageryDataset(
        num_samples=max(1024, int(dcfg["num_samples"]) // 2),
        channels=int(dcfg["channels"]),
        time_steps=int(dcfg["time_steps"]),
        num_classes=int(dcfg["num_classes"]),
        seed=int(cfg["seed"]) + 1,
    )
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    model = build_model(cfg, device)
    if bool(cfg.get("train", {}).get("dp_enabled", False)):
        model = ModuleValidator.fix(model)
    ckpt = out_dir / "best_model.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    batch_lat_ms = []
    sample_lat_ms = []
    power_samples = []

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            x = x.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _logits, _spk = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt_ms = (t1 - t0) * 1000.0
            batch_lat_ms.append(dt_ms)
            sample_lat_ms.append(dt_ms / x.shape[0])

            power = query_gpu_power_watt()
            if power is not None:
                power_samples.append(power)

    mean_batch_ms = float(np.mean(batch_lat_ms)) if batch_lat_ms else 0.0
    mean_sample_ms = float(np.mean(sample_lat_ms)) if sample_lat_ms else 0.0
    mean_power_w = float(np.mean(power_samples)) if power_samples else None

    energy_per_sample_mj = None
    if mean_power_w is not None and mean_sample_ms > 0.0:
        energy_per_sample_mj = mean_power_w * mean_sample_ms

    report = {
        "num_batches": int(min(num_batches, len(loader))),
        "mean_batch_latency_ms": mean_batch_ms,
        "mean_sample_latency_ms": mean_sample_ms,
        "mean_gpu_power_w": mean_power_w,
        "energy_per_sample_mj": energy_per_sample_mj,
    }

    csv_path = out_dir / "energy_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_latency_ms", "sample_latency_ms"])
        for a, b in zip(batch_lat_ms, sample_lat_ms):
            writer.writerow([a, b])

    save_json(report, out_dir / "energy_report.json")
    return report


def main(config_path: str, num_batches: int) -> None:
    cfg = load_config(config_path)
    report = evaluate_energy_from_config(cfg, num_batches)
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference latency and energy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num-batches", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config, args.num_batches)
