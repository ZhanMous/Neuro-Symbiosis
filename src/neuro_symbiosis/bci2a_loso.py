from __future__ import annotations

import argparse
import csv
from pathlib import Path

from torch.utils.data import DataLoader

from neuro_symbiosis.data.bci_iv_2a import load_bci_iv_2a_npz
from neuro_symbiosis.data.loso_split import loso_splits, loso_summary
from neuro_symbiosis.loso_pilot import (
    build_cfg,
    make_coupling_plot,
    resolve_device,
    summarise,
    train_one_fold,
)
from neuro_symbiosis.models.factory import build_model


def run_bci2a_loso(
    data_path: str,
    model_types: list[str],
    epochs: int,
    batch_size: int,
    device_name: str,
    seed: int,
) -> list[dict]:
    x, y, subjects = load_bci_iv_2a_npz(data_path, return_subjects=True)
    device = resolve_device(device_name)
    all_rows: list[dict] = []

    n_channels = int(x.shape[1])
    n_time = int(x.shape[2])
    n_classes = int(y.max() + 1)

    print("=== BCI-IV 2a LOSO ===")
    print(loso_summary(x, y, subjects))

    for model_type in model_types:
        print(f"\n=== Model: {model_type.upper()} ===")
        cfg = build_cfg(model_type, n_channels, n_time, n_classes, epochs, seed)
        cfg["device"] = device_name
        cfg["train"]["batch_size"] = batch_size

        for fold_idx, (train_ds, val_ds, held_out) in enumerate(loso_splits(x, y, subjects)):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCI-IV 2a LOSO runner")
    parser.add_argument("--data-path", type=str, default="data/processed/bci2a_preprocessed.npz")
    parser.add_argument("--model-types", nargs="+", default=["hybrid", "snn", "transformer"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/bci2a_loso")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_bci2a_loso(
        data_path=args.data_path,
        model_types=args.model_types,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
        seed=args.seed,
    )

    fold_csv = out_dir / "loso_fold_results.csv"
    fold_fields = [
        "model",
        "fold",
        "held_out_subject",
        "val_acc",
        "spike_rate",
        "effective_token_length",
        "latency_ms",
    ]
    with fold_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fold_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv saved] {fold_csv}")

    summary = summarise(rows, args.model_types)
    summary_csv = out_dir / "loso_summary.csv"
    summary_fields = [
        "model",
        "n_folds",
        "val_acc_mean",
        "val_acc_std",
        "val_acc_ci95",
        "spike_rate_mean",
        "spike_rate_std",
        "eff_token_len_mean",
        "eff_token_len_std",
        "latency_ms_mean",
        "latency_ms_std",
    ]
    with summary_csv.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"[csv saved] {summary_csv}")

    make_coupling_plot(rows, out_dir / "loso_coupling.png")
