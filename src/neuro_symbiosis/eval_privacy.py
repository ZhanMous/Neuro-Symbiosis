from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from opacus.validators import ModuleValidator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.data.factory import build_datasets
from neuro_symbiosis.models.factory import build_model


def collect_confidences(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    model.eval()
    conf = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits, _ = model(x)
            probs = F.softmax(logits, dim=1)
            max_conf = probs.max(dim=1).values
            conf.extend(max_conf.detach().cpu().numpy().tolist())
    return conf


def evaluate_privacy_from_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    device = torch.device("cuda" if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(str(cfg.get("output_dir", "outputs/default")))

    train_set, val_set = build_datasets(cfg)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    model = build_model(cfg, device)
    if bool(cfg.get("train", {}).get("dp_enabled", False)):
        model = ModuleValidator.fix(model)
    ckpt = out_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))

    member_conf = collect_confidences(model, train_loader, device)
    nonmember_conf = collect_confidences(model, val_loader, device)

    y_true = np.array([1] * len(member_conf) + [0] * len(nonmember_conf))
    y_score = np.array(member_conf + nonmember_conf)

    threshold = float(np.median(y_score))
    y_pred = (y_score >= threshold).astype(np.int64)
    mia_acc = float((y_pred == y_true).mean())
    mia_auc = float(roc_auc_score(y_true, y_score))

    report = {
        "attack": "confidence_threshold",
        "mia_acc": mia_acc,
        "mia_auc": mia_auc,
        "threshold": threshold,
        "member_mean_conf": float(np.mean(member_conf)),
        "nonmember_mean_conf": float(np.mean(nonmember_conf)),
    }
    save_json(report, out_dir / "privacy_report.json")
    return report


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    report = evaluate_privacy_from_config(cfg)
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate membership inference risk")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
