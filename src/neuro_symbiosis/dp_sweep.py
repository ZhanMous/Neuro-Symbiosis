from __future__ import annotations

import argparse
import copy
from pathlib import Path

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.eval_energy import evaluate_energy_from_config
from neuro_symbiosis.train import train_from_config


def run_dp_sweep(config_path: str, num_batches: int) -> dict:
    base_cfg = load_config(config_path)
    base_out = Path(str(base_cfg.get("output_dir", "outputs/default")))
    sweep_dir = base_out / "dp_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Lightweight grid for single-GPU laptop runs.
    noise_grid = [0.8, 1.0, 1.2]
    clip_grid = [0.8, 1.0]

    rows = []
    run_id = 0
    for noise in noise_grid:
        for clip in clip_grid:
            run_id += 1
            cfg = copy.deepcopy(base_cfg)
            cfg["model"]["type"] = "hybrid"
            cfg["train"]["dp_enabled"] = True
            cfg["train"]["dp_noise_multiplier"] = float(noise)
            cfg["train"]["dp_max_grad_norm"] = float(clip)
            cfg["output_dir"] = str(sweep_dir / f"run_{run_id:02d}_n{noise}_c{clip}")

            train_report = train_from_config(cfg)
            energy_report = evaluate_energy_from_config(cfg, num_batches=num_batches)

            eps = None
            if train_report.get("privacy_spent"):
                eps = train_report["privacy_spent"].get("epsilon")

            rows.append(
                {
                    "run_id": run_id,
                    "noise_multiplier": noise,
                    "max_grad_norm": clip,
                    "epsilon": eps,
                    "val_acc": float(train_report["best_val_acc"]),
                    "energy_mj": energy_report["energy_per_sample_mj"],
                    "sample_latency_ms": float(energy_report["mean_sample_latency_ms"]),
                }
            )

    csv_lines = ["run_id,noise_multiplier,max_grad_norm,epsilon,val_acc,energy_mj,sample_latency_ms"]
    md_lines = ["|run_id|noise|clip|epsilon|val_acc|energy_mj|sample_latency_ms|", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        eps_str = "" if r["epsilon"] is None else f"{float(r['epsilon']):.4f}"
        energy_str = "" if r["energy_mj"] is None else f"{float(r['energy_mj']):.4f}"
        csv_lines.append(
            f"{r['run_id']},{r['noise_multiplier']},{r['max_grad_norm']},{eps_str},{r['val_acc']:.6f},{energy_str},{r['sample_latency_ms']:.6f}"
        )
        md_lines.append(
            f"|{r['run_id']}|{r['noise_multiplier']:.2f}|{r['max_grad_norm']:.2f}|{eps_str}|{r['val_acc']:.4f}|{energy_str}|{r['sample_latency_ms']:.4f}|"
        )

    (sweep_dir / "dp_sweep_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    (sweep_dir / "dp_sweep_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary = {"rows": rows, "sweep_dir": str(sweep_dir)}
    save_json(summary, sweep_dir / "dp_sweep_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DP-SGD grid search and export epsilon-accuracy-energy table")
    p.add_argument("--config", type=str, default="configs/quick.yaml")
    p.add_argument("--num-batches", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_dp_sweep(args.config, num_batches=args.num_batches)
    print(result)
