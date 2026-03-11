# Neuro-Symbiosis 8-Week Plan

## Goal

Develop and evaluate a hybrid SNN-Transformer model for BCI decoding with co-optimization over:
- utility
- privacy
- energy

## Research questions

1. Does SNN-Transformer improve utility-energy Pareto tradeoff against pure Transformer baselines?
2. How does DP-SGD impact utility and MIA resistance in this hybrid architecture?
3. Which SNN time-step and Transformer depth settings are optimal on a single RTX 4070?

## Milestones

Week 1
- finalize task definition and metrics
- validate synthetic pipeline end to end

Week 2
- add BCI IV 2a data loader and baseline preprocessing
- run CNN and Transformer baseline training

Week 3
- tune SNN encoder hyperparameters: beta, threshold, hidden width, time window

Week 4
- hybrid ablations: transformer depth/heads and token pooling strategy

Week 5
- privacy experiments: MIA baseline and DP-SGD sweeps

Week 6
- energy profiling and throughput analysis on fixed hardware settings

Week 7
- statistical repeats across seeds and confidence intervals

Week 8
- finalize figures and paper draft

## Metrics

Utility
- accuracy, macro-F1

Privacy
- MIA accuracy, MIA AUC
- epsilon under delta=1e-5 when DP is enabled

Energy and efficiency
- per-sample latency
- optional measured GPU power and estimated mJ/sample
- spike rate and effective sparse compute ratio

## Risks and mitigations

1. Dataset preprocessing complexity
- Mitigation: stage rollout with synthetic data first, then one real dataset at a time.

2. DP training instability
- Mitigation: use clipping and lower LR; start with smaller noise multiplier.

3. GPU memory limits
- Mitigation: gradient accumulation and smaller batch/token lengths.
