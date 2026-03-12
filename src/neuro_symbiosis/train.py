from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.data.factory import build_datasets
from neuro_symbiosis.models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    spike_rates = []
    eff_token_lengths = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, spike_rate = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            total_correct += int((pred == y).sum().item())
            total_count += int(y.numel())
            total_loss += float(loss.item()) * y.size(0)
            spike_rates.append(float(spike_rate.item()))
            inner = model._module if hasattr(model, "_module") else model
            if hasattr(inner, "last_metrics"):
                eff_token_lengths.append(inner.last_metrics.get("effective_token_length", 0.0))

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)
    avg_spike = float(np.mean(spike_rates)) if spike_rates else 0.0
    avg_eff_tokens = float(np.mean(eff_token_lengths)) if eff_token_lengths else 0.0
    return avg_loss, acc, avg_spike, avg_eff_tokens


def train_from_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    set_seed(int(cfg["seed"]))

    device = resolve_device(str(cfg.get("device", "cuda")))
    out_dir = Path(str(cfg.get("output_dir", "outputs/default")))
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    tcfg = cfg["train"]

    train_set, val_set = build_datasets(cfg)

    train_loader = DataLoader(train_set, batch_size=int(tcfg["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(tcfg["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
    )

    privacy_spent = None
    if bool(tcfg.get("dp_enabled", False)):
        model = ModuleValidator.fix(model)
        ModuleValidator.validate(model, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(tcfg["lr"]),
            weight_decay=float(tcfg["weight_decay"]),
        )
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=float(tcfg["dp_noise_multiplier"]),
            max_grad_norm=float(tcfg["dp_max_grad_norm"]),
        )
    else:
        privacy_engine = None

    best_val_acc = -1.0
    history = []

    for epoch in range(int(tcfg["epochs"])):
        model.train()
        running_loss = 0.0
        running_count = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{tcfg['epochs']}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, spike_rate = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            optimizer.step()

            running_loss += float(loss.item()) * y.size(0)
            running_count += y.size(0)
            pbar.set_postfix(loss=float(loss.item()), spike_rate=float(spike_rate.item()))

        train_loss = running_loss / max(running_count, 1)
        val_loss, val_acc, val_spike, val_eff_tokens = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_spike_rate": val_spike,
            "val_effective_token_length": val_eff_tokens,
        }

        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            row["epsilon_delta1e-5"] = float(epsilon)
            privacy_spent = {"epsilon": float(epsilon), "delta": 1e-5}

        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state_dict = model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
            torch.save(state_dict, out_dir / "best_model.pt")

    report = {
        "config": cfg,
        "best_val_acc": best_val_acc,
        "privacy_spent": privacy_spent,
        "history": history,
    }
    save_json(report, out_dir / "train_metrics.json")
    return report


def train(cfg_path: str) -> dict:
    cfg = load_config(cfg_path)
    return train_from_config(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Neuro-Symbiosis hybrid model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
