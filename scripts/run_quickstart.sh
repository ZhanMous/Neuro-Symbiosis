#!/usr/bin/env bash
set -euo pipefail

python -m neuro_symbiosis.train --config configs/default.yaml
python -m neuro_symbiosis.eval_privacy --config configs/default.yaml
python -m neuro_symbiosis.eval_energy --config configs/default.yaml --num-batches 50

echo "Neuro-Symbiosis quickstart done. Reports are in outputs/default"
