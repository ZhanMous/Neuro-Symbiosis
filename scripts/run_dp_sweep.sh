#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m neuro_symbiosis.dp_sweep --config configs/quick.yaml --num-batches 20

echo "DP sweep finished. See outputs/quick/dp_sweep"
