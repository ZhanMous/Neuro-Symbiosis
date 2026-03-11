# NeurIPS Rebuild Plan (Energy-First Track)

## Positioning statement (for advisor/coauthors)

Current version is a reproducible prototype, not NeurIPS-ready evidence. We now pivot to one central scientific claim: energy mechanism in SNN-Transformer coupling.

## Important dataset note

PhysioNet eegmmidb is the EEG Motor Movement/Imagery dataset itself. If you want a third dataset, choose a different corpus instead of counting this twice.

## Scope and milestones

### Phase A (4-6 weeks): workshop-ready mechanism evidence

Goals:
1. Implement LOSO protocol on BCI-IV 2a and eegmmidb.
2. Add mechanism-oriented metrics: firing rate, effective token length, per-sample energy proxy.
3. Produce multi-seed statistics with CI and effect size.

Deliverables:
1. Mechanism section draft with equations and assumptions.
2. Core figures: 3D coupling, heatmap, radar.
3. Workshop manuscript and reproducibility appendix.

### Phase B (8-12 weeks): NeurIPS-level strengthening

Goals:
1. Validate coupling model across datasets and subject splits.
2. Add stronger baselines and deeper ablations.
3. Add robust statistical testing (ANOVA + effect sizes + multiple comparisons control).

Deliverables:
1. Full paper with mechanism + predictive validation.
2. Public code release with one-command experiment reproduction.

## Experimental matrix (minimum)

1. Datasets:
- BCI-IV 2a (LOSO)
- eegmmidb (LOSO)
- one additional distinct EEG/BCI corpus

2. Model families:
- SNN
- Transformer
- CNN-Transformer
- Hybrid (Neuro-Symbiosis)

3. Evaluation:
- Utility: accuracy, macro-F1
- Energy: latency, power, mJ/sample, spike-driven sparse proxy
- Privacy (secondary axis): MIA (shadow + threshold), report trade-off only

4. Statistics:
- >=5 random seeds
- 95% CI
- ANOVA / paired tests
- Cohen's d

## Resource-aware training strategy (RTX 4070 laptop)

1. Start with narrow model widths and short sequences for pilot effect estimation.
2. Use early stopping and gradient accumulation for full LOSO sweeps.
3. Prioritize statistical power on fewer key settings over broad weak grids.

## Weekly execution checklist

Week 1:
1. Lock hypothesis and notation.
2. Implement LOSO loaders and split verification.
3. Build metrics logging for r, L, latency/power.

Week 2:
1. Run pilot LOSO on two datasets.
2. Generate first coupling curves and sanity checks.

Week 3:
1. Multi-seed full baseline runs.
2. Compute CI and effect sizes.

Week 4:
1. Ablation runs and trade-off plots.
2. Workshop draft submission package.

Week 5+:
1. Scale to third dataset and stronger baselines.
2. Finalize NeurIPS-oriented manuscript.

## What to cut

1. Do not spend major time on framework polishing unless it affects reproducibility.
2. Keep implementation text compact; move details to appendix.
3. Focus writing on falsifiable claim, evidence quality, and limits.
