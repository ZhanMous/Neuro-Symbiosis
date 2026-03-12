from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.eval_energy import evaluate_energy_from_config
from neuro_symbiosis.eval_privacy import evaluate_privacy_from_config
from neuro_symbiosis.train import train_from_config


def run_multiseed(config_path: str, seeds: list[int], num_batches: int) -> dict:
    base_cfg = load_config(config_path)
    base_out = Path(str(base_cfg.get("output_dir", "outputs/default")))
    out_dir = base_out / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_types = ["snn", "transformer", "hybrid"]
    raw_rows: list[dict] = []

    for model_type in model_types:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = int(seed)
            cfg["model"]["type"] = model_type
            cfg["output_dir"] = str(out_dir / f"{model_type}_seed{seed}")

            train_report = train_from_config(cfg)
            privacy_report = evaluate_privacy_from_config(cfg)
            energy_report = evaluate_energy_from_config(cfg, num_batches=num_batches)

            raw_rows.append(
                {
                    "model": model_type,
                    "seed": seed,
                    "val_acc": float(train_report["best_val_acc"]),
                    "mia_auc": float(privacy_report["mia_auc"]),
                    "energy_mj": None if energy_report["energy_per_sample_mj"] is None else float(energy_report["energy_per_sample_mj"]),
                    "lat_ms": float(energy_report["mean_sample_latency_ms"]),
                }
            )

    summary_rows = []
    for model_type in model_types:
        rows = [r for r in raw_rows if r["model"] == model_type]
        acc = np.array([r["val_acc"] for r in rows], dtype=np.float64)
        auc = np.array([r["mia_auc"] for r in rows], dtype=np.float64)
        lat = np.array([r["lat_ms"] for r in rows], dtype=np.float64)
        e_vals = [r["energy_mj"] for r in rows if r["energy_mj"] is not None]
        ene = np.array(e_vals, dtype=np.float64) if e_vals else None

        summary_rows.append(
            {
                "model": model_type,
                "val_acc_mean": float(acc.mean()),
                "val_acc_std": float(acc.std(ddof=0)),
                "mia_auc_mean": float(auc.mean()),
                "mia_auc_std": float(auc.std(ddof=0)),
                "lat_ms_mean": float(lat.mean()),
                "lat_ms_std": float(lat.std(ddof=0)),
                "energy_mj_mean": None if ene is None else float(ene.mean()),
                "energy_mj_std": None if ene is None else float(ene.std(ddof=0)),
            }
        )

    csv_lines = ["model,val_acc_mean,val_acc_std,mia_auc_mean,mia_auc_std,lat_ms_mean,lat_ms_std,energy_mj_mean,energy_mj_std"]
    for r in summary_rows:
        em = "" if r["energy_mj_mean"] is None else f"{r['energy_mj_mean']:.6f}"
        es = "" if r["energy_mj_std"] is None else f"{r['energy_mj_std']:.6f}"
        csv_lines.append(
            f"{r['model']},{r['val_acc_mean']:.6f},{r['val_acc_std']:.6f},{r['mia_auc_mean']:.6f},{r['mia_auc_std']:.6f},{r['lat_ms_mean']:.6f},{r['lat_ms_std']:.6f},{em},{es}"
        )

    (out_dir / "multiseed_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    save_json({"seeds": seeds, "raw_rows": raw_rows, "summary_rows": summary_rows}, out_dir / "multiseed_summary.json")
    return {"out_dir": str(out_dir), "summary_rows": summary_rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-seed benchmark for three model types")
    p.add_argument("--config", type=str, default="configs/quick.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--num-batches", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_multiseed(args.config, seeds=args.seeds, num_batches=args.num_batches)
    print(result)
