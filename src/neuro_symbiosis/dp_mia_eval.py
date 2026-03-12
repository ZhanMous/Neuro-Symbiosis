from __future__ import annotations

import argparse
import copy
from pathlib import Path

from neuro_symbiosis.config import load_config, save_json
from neuro_symbiosis.eval_privacy import evaluate_privacy_from_config


def run_dp_mia(config_path: str) -> dict:
    base_cfg = load_config(config_path)
    base_out = Path(str(base_cfg.get("output_dir", "outputs/default"))) / "dp_sweep"
    if not base_out.exists():
        raise FileNotFoundError(f"DP sweep directory not found: {base_out}")

    rows = []
    for run_dir in sorted([p for p in base_out.iterdir() if p.is_dir() and p.name.startswith("run_")]):
        name = run_dir.name
        # run_01_n0.8_c0.8
        parts = name.split("_")
        noise = float(parts[2][1:])
        clip = float(parts[3][1:])

        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["dp_enabled"] = True
        cfg["train"]["dp_noise_multiplier"] = noise
        cfg["train"]["dp_max_grad_norm"] = clip
        cfg["output_dir"] = str(run_dir)

        report = evaluate_privacy_from_config(cfg)
        rows.append(
            {
                "run": name,
                "noise": noise,
                "clip": clip,
                "mia_acc": float(report["mia_acc"]),
                "mia_auc": float(report["mia_auc"]),
            }
        )

    csv_lines = ["run,noise,clip,mia_acc,mia_auc"]
    for r in rows:
        csv_lines.append(f"{r['run']},{r['noise']:.2f},{r['clip']:.2f},{r['mia_acc']:.6f},{r['mia_auc']:.6f}")

    out_path = base_out / "dp_mia_summary.csv"
    out_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    save_json({"rows": rows}, base_out / "dp_mia_summary.json")
    return {"out_path": str(out_path), "rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MIA on all DP sweep runs")
    p.add_argument("--config", type=str, default="configs/quick.yaml")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_dp_mia(args.config)
    print(result)
