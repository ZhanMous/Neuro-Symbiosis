#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m neuro_symbiosis.benchmark_baselines --config configs/quick.yaml --num-batches 20

echo "Baseline benchmark finished. See outputs/quick/benchmark"
