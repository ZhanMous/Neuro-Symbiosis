# Energy Mechanism Hypothesis (Neuro-Symbiosis)

## Core falsifiable hypothesis

In SNN-Transformer hybrid decoding for BCI, SNN firing rate and Transformer token length are nonlinearly coupled, and this coupling dominates end-to-end decoding energy cost.

Formally, for sample-level cost:

E_total = E_snn(r, T_s) + E_attn(L, d) + E_ffn(L, d) + E_io

where:
- r: average firing rate in SNN encoder
- T_s: SNN simulation steps / temporal window
- L: effective token length entering Transformer
- d: embedding dimension

Coupling assumption:

L = g(r, T_s, tau)

with tau as tokenization/threshold controls. We expect:
- low r can reduce redundant tokens and attention cost;
- overly high r introduces noisy tokens and increases attention and FFN cost;
- there exists an operating region r* that minimizes energy under bounded utility degradation.

## What makes this not just engineering

1. The paper must explain why this coupling exists (signal sparsification and token redundancy dynamics), not only report benchmarks.
2. The cost model must be validated against measured latency/power trends and spike statistics.
3. The model must be predictive: selecting (r, L) by the model should improve energy while preserving accuracy within a target margin.

## Testable predictions

1. Under fixed accuracy band, energy vs firing-rate is U-shaped or piecewise-convex.
2. Model-guided hyperparameter search outperforms blind grid search on energy-accuracy Pareto.
3. Coupling trend is consistent across LOSO splits on BCI-IV 2a and eegmmidb.

## Required outputs for paper

1. 3D plot: firing rate vs token length vs energy.
2. Cross-subject heatmap: utility/energy stability under LOSO.
3. Ablation radar: utility/privacy/energy shifts for key modules.
4. Table with effect sizes (Cohen's d) and confidence intervals.
