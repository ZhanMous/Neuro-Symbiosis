from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.eval_energy import evaluate_energy_from_config
from neuro_symbiosis.eval_privacy import evaluate_privacy_from_config
from neuro_symbiosis.train import train_from_config


def run_baselines(config_path: str, num_batches: int) -> dict:
    base_cfg = load_config(config_path)
    base_out = Path(str(base_cfg.get("output_dir", "outputs/default")))
    summary_dir = base_out / "benchmark"
    summary_dir.mkdir(parents=True, exist_ok=True)

    model_types = ["snn", "transformer", "hybrid"]
    rows = []

    for model_type in model_types:
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["type"] = model_type
        cfg["output_dir"] = str(base_out / model_type)

        train_report = train_from_config(cfg)
        privacy_report = evaluate_privacy_from_config(cfg)
        energy_report = evaluate_energy_from_config(cfg, num_batches=num_batches)

        row = {
            "model": model_type,
            "val_acc": float(train_report["best_val_acc"]),
            "mia_auc": float(privacy_report["mia_auc"]),
            "mia_acc": float(privacy_report["mia_acc"]),
            "sample_latency_ms": float(energy_report["mean_sample_latency_ms"]),
            "energy_mj": energy_report["energy_per_sample_mj"],
            "mean_spike_rate": float(train_report["history"][-1]["val_spike_rate"]),
        }
        rows.append(row)

    # Pareto scatter: x=energy(mJ), y=accuracy, color=MIA AUC.
    x = []
    y = []
    c = []
    labels = []
    for r in rows:
        energy_val = r["energy_mj"] if r["energy_mj"] is not None else r["sample_latency_ms"]
        x.append(float(energy_val))
        y.append(float(r["val_acc"]))
        c.append(float(r["mia_auc"]))
        labels.append(r["model"])

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=140)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Energy per sample (mJ) or latency proxy")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Neuro-Symbiosis Baseline Pareto")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("MIA AUC (lower is better)")
    fig.tight_layout()
    fig.savefig(summary_dir / "pareto_baselines.png", dpi=180)
    plt.close(fig)

    # Save table.
    csv_lines = ["model,val_acc,mia_auc,mia_acc,sample_latency_ms,energy_mj,mean_spike_rate"]
    for r in rows:
        csv_lines.append(
            ",".join(
                [
                    r["model"],
                    f"{r['val_acc']:.6f}",
                    f"{r['mia_auc']:.6f}",
                    f"{r['mia_acc']:.6f}",
                    f"{r['sample_latency_ms']:.6f}",
                    "" if r["energy_mj"] is None else f"{float(r['energy_mj']):.6f}",
                    f"{r['mean_spike_rate']:.6f}",
                ]
            )
        )
    (summary_dir / "baseline_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    summary = {"rows": rows, "summary_dir": str(summary_dir)}
    save_json(summary, summary_dir / "baseline_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SNN/Transformer/Hybrid baselines and draw Pareto plot")
    p.add_argument("--config", type=str, default="configs/quick.yaml")
    p.add_argument("--num-batches", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_baselines(args.config, num_batches=args.num_batches)
    print(result)
