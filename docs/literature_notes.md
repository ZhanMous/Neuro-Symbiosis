# Literature and Source Notes

## BCI data sources

1. BCI Competition IV
- Includes classic motor imagery datasets 2a and 2b.
- Dataset 2a summary: 22 EEG + 3 EOG, 250 Hz, 4 classes, 9 subjects.
- Dataset 2b summary: 3 bipolar EEG + 3 EOG, 250 Hz, 2 classes, 9 subjects.
- Action item: read dataset-specific PDF and follow citation requirements for publication.

2. PhysioNet EEG Motor Movement/Imagery (eegmmidb)
- 109 subjects, 64 channels, 160 Hz.
- More than 1500 recordings.
- Annotation labels include rest and movement/imagery markers (T0, T1, T2).
- Access policy is open with dataset license terms.
- Download command provided by PhysioNet:
  - wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

## SNN tooling

1. snnTorch
- PyTorch-integrated SNN library with surrogate-gradient support.
- Supports spike generation and utility functions.
- Can run on GPU with standard PyTorch flow.

## Differential privacy tooling

1. Opacus
- Differential privacy for PyTorch.
- Core API: PrivacyEngine.make_private(module, optimizer, data_loader, ...)
- Practical for DP-SGD baselines in this project.

## Design implications for Neuro-Symbiosis

- Keep first phase lightweight using synthetic data to validate full pipeline.
- Add BCI-IV 2a and eegmmidb adapters as phase 2.
- Report both utility and privacy metrics together.
- For energy reporting, include latency and optional measured GPU power.
