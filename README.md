# Neuro-Symbiosis

Neuro-Symbiosis is a single-GPU research project for BCI decoding with a hybrid SNN-Transformer model, focusing on the joint optimization of:

- utility (classification performance)
- privacy (membership inference resistance, optional DP-SGD)
- energy efficiency (runtime and optional GPU power logging)

## Why this project

This repo is designed for laptop-scale research (RTX 4070 class). It provides:

- a lightweight SNN front-end with surrogate gradients
- a Transformer temporal-context decoder
- reproducible scripts for training, privacy attack evaluation, and energy profiling

## Quickstart

```bash
cd /home/yanshi/Neuro-Symbiosis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m neuro_symbiosis.train --config configs/default.yaml
python -m neuro_symbiosis.eval_privacy --config configs/default.yaml
python -m neuro_symbiosis.eval_energy --config configs/default.yaml --num-batches 100
```

## Execute the 3 core workflows

1) BCI-IV 2a data preprocessing and training

```bash
cd /home/yanshi/Neuro-Symbiosis
python scripts/preprocess_bci2a_npz.py \
	--input data/raw/bci2a_raw.npz \
	--output data/processed/bci2a_preprocessed.npz \
	--fs 250 --low-hz 8 --high-hz 30 --t-start 0 --t-len 256
PYTHONPATH=src python -m neuro_symbiosis.train --config configs/bci2a_template.yaml
```

2) Baselines and Pareto figure (SNN / Transformer / Hybrid)

```bash
cd /home/yanshi/Neuro-Symbiosis
bash scripts/run_baselines.sh
```

3) DP-SGD grid search and epsilon-accuracy-energy table

```bash
cd /home/yanshi/Neuro-Symbiosis
bash scripts/run_dp_sweep.sh
```

## Project structure

- `src/neuro_symbiosis/models`: SNN, Transformer, and hybrid model
- `src/neuro_symbiosis/data`: synthetic EEG-like dataset and split helpers
- `src/neuro_symbiosis/train.py`: training and validation loop
- `src/neuro_symbiosis/eval_privacy.py`: confidence-threshold MIA baseline
- `src/neuro_symbiosis/eval_energy.py`: latency and optional GPU power profile
- `src/neuro_symbiosis/benchmark_baselines.py`: baseline benchmark + Pareto plot
- `src/neuro_symbiosis/dp_sweep.py`: DP-SGD sweep + summary table
- `docs/literature_notes.md`: public-source notes for BCI and tooling
- `docs/research_plan.md`: 8-week execution plan for this topic
- `paper/Neuro-Symbiosis_Draft_v1_zh.md`: Chinese first draft manuscript

## Datasets suggested for next step

- BCI Competition IV 2a and 2b
- PhysioNet EEG Motor Movement/Imagery (eegmmidb)

Current default uses synthetic EEG-like signals so the full pipeline can be tested without dataset download.

## Expected outputs

Training creates artifacts under `outputs/default`:

- `best_model.pt`
- `train_metrics.json`
- `privacy_report.json`
- `energy_report.json`

## Reproducibility tips for 4070

- start with batch size 32 to 64
- keep transformer layers <= 2 in early ablations
- run 3 seeds for preliminary report, 5 seeds for final tables

## GitHub publish

```bash
cd /home/yanshi/Neuro-Symbiosis
chmod +x scripts/publish_github.sh
bash scripts/publish_github.sh Neuro-Symbiosis public
```

If `gh auth status` fails, run `gh auth login` once and rerun the script.
