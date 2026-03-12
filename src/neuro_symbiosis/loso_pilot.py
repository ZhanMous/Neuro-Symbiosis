"""LOSO (Leave-One-Subject-Out) Pilot Runner.

Trains each model type on BCI-style data using LOSO cross-validation.
Records per-fold: val_acc, spike_rate, effective_token_length, energy estimate.
Aggregates over folds to produce mean ± std summary.

Outputs (under --out-dir):
  loso_fold_results.csv   — per-fold raw rows
  loso_summary.csv        — per-model mean ± std across folds
  loso_coupling.png       — spike_rate vs effective_token_length scatter per model

Run
---
    PYTHONPATH=src python src/neuro_symbiosis/loso_pilot.py
    PYTHONPATH=src python src/neuro_symbiosis/loso_pilot.py \\
        --model-types hybrid snn transformer \\
        --n-subjects 9 --n-trials 288 --epochs 5 --out-dir outputs/loso_pilot
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuro_symbiosis.config import save_json
from neuro_symbiosis.data.loso_split import loso_splits, synthetic_loso_data
from neuro_symbiosis.models.factory import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_cfg(
    model_type: str,
    n_channels: int,
    n_time: int,
    n_classes: int,
    epochs: int,
    seed: int,
    embed_dim: int = 64,
    snn_hidden: int = 64,
) -> dict:
    return {
        "seed": seed,
        "device": "cuda",
        "data": {
            "channels": n_channels,
            "time_steps": n_time,
            "num_classes": n_classes,
        },
        "model": {
            "type": model_type,
            "embed_dim": embed_dim,
            "snn_hidden": snn_hidden,
            "lif_beta": 0.9,
            "threshold": 1.0,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 32,
            "epochs": epochs,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "dp_enabled": False,
        },
    }


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    grad_clip: float = 1.0,
) -> dict:
    """Train for `epochs` and return best-fold metrics dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = -1.0
    best_metrics: dict = {}

    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _spike = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # ---- val ----
        model.eval()
        correct = total = 0
        spike_rates, eff_tokens = [], []
        latencies = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                t0 = time.perf_counter()
                logits, spike_rate = model(x)
                latencies.append((time.perf_counter() - t0) * 1000.0 / x.size(0))

                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum())
                total += int(y.numel())
                spike_rates.append(float(spike_rate.item()))

                inner = model._module if hasattr(model, "_module") else model
                if hasattr(inner, "last_metrics"):
                    eff_tokens.append(inner.last_metrics.get("effective_token_length", 0.0))

        acc = correct / max(total, 1)
        avg_spike = float(np.mean(spike_rates))
        avg_eff_tok = float(np.mean(eff_tokens)) if eff_tokens else 0.0
        avg_latency = float(np.mean(latencies))

        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                "val_acc": round(acc, 6),
                "spike_rate": round(avg_spike, 6),
                "effective_token_length": round(avg_eff_tok, 4),
                "latency_ms": round(avg_latency, 4),
            }

    return best_metrics


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_loso_pilot(
    model_types: list[str],
    n_subjects: int,
    n_trials: int,
    n_channels: int,
    n_time: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    device_name: str,
    seed: int,
    out_dir: Path,
) -> list[dict]:
    """Run full LOSO for each model type, return list of per-fold rows."""
    device = resolve_device(device_name)
    x, y, subjects = synthetic_loso_data(
        n_subjects=n_subjects,
        n_trials_per_subject=n_trials,
        n_channels=n_channels,
        n_time=n_time,
        n_classes=n_classes,
        seed=seed,
    )

    all_rows: list[dict] = []

    for model_type in model_types:
        print(f"\n=== Model: {model_type.upper()} ===")
        cfg = build_cfg(model_type, n_channels, n_time, n_classes, epochs, seed)

        for fold_idx, (train_ds, val_ds, held_out) in enumerate(loso_splits(x, y, subjects)):
            torch.manual_seed(seed)
            model = build_model(cfg, device)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

            metrics = train_one_fold(model, train_loader, val_loader, device, epochs)

            row = {
                "model": model_type,
                "fold": fold_idx + 1,
                "held_out_subject": held_out,
                **metrics,
            }
            all_rows.append(row)
            print(
                f"  Fold {fold_idx + 1:2d} (subj={held_out:2d}) | "
                f"val_acc={metrics['val_acc']:.4f}  "
                f"spike_rate={metrics['spike_rate']:.4f}  "
                f"eff_tok={metrics['effective_token_length']:.1f}  "
                f"lat={metrics['latency_ms']:.3f}ms"
            )

    return all_rows


def summarise(rows: list[dict], model_types: list[str]) -> list[dict]:
    """Aggregate per-fold rows to per-model mean ± std."""
    summary = []
    for mt in model_types:
        subset = [r for r in rows if r["model"] == mt]
        if not subset:
            continue
        def stat(key: str) -> tuple[float, float]:
            vals = [r[key] for r in subset]
            return float(np.mean(vals)), float(np.std(vals))

        acc_m, acc_s = stat("val_acc")
        sr_m, sr_s = stat("spike_rate")
        el_m, el_s = stat("effective_token_length")
        lat_m, lat_s = stat("latency_ms")
        n_folds = len(subset)
        ci95_acc = 1.96 * acc_s / (n_folds ** 0.5)

        summary.append({
            "model": mt,
            "n_folds": n_folds,
            "val_acc_mean": round(acc_m, 4),
            "val_acc_std": round(acc_s, 4),
            "val_acc_ci95": round(ci95_acc, 4),
            "spike_rate_mean": round(sr_m, 6),
            "spike_rate_std": round(sr_s, 6),
            "eff_token_len_mean": round(el_m, 2),
            "eff_token_len_std": round(el_s, 2),
            "latency_ms_mean": round(lat_m, 4),
            "latency_ms_std": round(lat_s, 4),
        })
    return summary


def make_coupling_plot(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"hybrid": "tab:blue", "snn": "tab:orange", "transformer": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_type, color in colors.items():
        subset = [r for r in rows if r["model"] == model_type]
        if not subset:
            continue
        sr = [r["spike_rate"] for r in subset]
        el = [r["effective_token_length"] for r in subset]
        ax.scatter(sr, el, c=color, label=model_type, s=50, alpha=0.75)

    ax.set_xlabel("Spike Rate $r$")
    ax.set_ylabel("Effective Token Length $L$")
    ax.set_title("LOSO Coupling: $r$ vs $L$ per Fold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot saved] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOSO pilot runner")
    p.add_argument("--model-types", nargs="+", default=["hybrid", "snn", "transformer"])
    p.add_argument("--n-subjects", type=int, default=9)
    p.add_argument("--n-trials", type=int, default=288)
    p.add_argument("--n-channels", type=int, default=22)
    p.add_argument("--n-time", type=int, default=128)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="outputs/loso_pilot")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_loso_pilot(
        model_types=args.model_types,
        n_subjects=args.n_subjects,
        n_trials=args.n_trials,
        n_channels=args.n_channels,
        n_time=args.n_time,
        n_classes=args.n_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        seed=args.seed,
        out_dir=out_dir,
    )

    # Save fold-level CSV
    fold_csv = out_dir / "loso_fold_results.csv"
    fold_fields = ["model", "fold", "held_out_subject", "val_acc", "spike_rate",
                   "effective_token_length", "latency_ms"]
    with fold_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[csv saved] {fold_csv}")

    # Summary
    model_types = args.model_types
    summary = summarise(rows, model_types)
    sum_csv = out_dir / "loso_summary.csv"
    sum_fields = ["model", "n_folds", "val_acc_mean", "val_acc_std", "val_acc_ci95",
                  "spike_rate_mean", "spike_rate_std",
                  "eff_token_len_mean", "eff_token_len_std",
                  "latency_ms_mean", "latency_ms_std"]
    with sum_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[csv saved] {sum_csv}")

    # Coupling plot
    make_coupling_plot(rows, out_dir / "loso_coupling.png")

    # Print summary table
    print("\n=== LOSO Summary ===")
    print(f"{'model':<14} {'folds':>6} {'acc_mean±std':>18} {'acc_ci95':>10} {'eff_tok_mean':>14}")
    for s in summary:
        print(
            f"{s['model']:<14} {s['n_folds']:>6} "
            f"{s['val_acc_mean']:.4f}±{s['val_acc_std']:.4f}"
            f"{'':>4}{s['val_acc_ci95']:.4f}{'':>6}"
            f"{s['eff_token_len_mean']:.1f}±{s['eff_token_len_std']:.1f}"
        )
