from __future__ import annotations

from torch.utils.data import Dataset

from neuro_symbiosis.data.bci_iv_2a import load_bci_iv_2a_npz, split_bci_dataset
from neuro_symbiosis.data.eeg_synthetic import SyntheticMotorImageryDataset, build_train_val_split


def build_datasets(cfg: dict) -> tuple[Dataset, Dataset]:
    dcfg = cfg["data"]
    dataset_name = str(dcfg.get("dataset", "synthetic_motor_imagery")).lower()
    seed = int(cfg["seed"])

    if dataset_name == "synthetic_motor_imagery":
        dataset = SyntheticMotorImageryDataset(
            num_samples=int(dcfg["num_samples"]),
            channels=int(dcfg["channels"]),
            time_steps=int(dcfg["time_steps"]),
            num_classes=int(dcfg["num_classes"]),
            seed=seed,
        )
        return build_train_val_split(dataset, float(dcfg["train_ratio"]), seed)

    if dataset_name == "bci_iv_2a_npz":
        data_path = str(dcfg["data_path"])
        x, y = load_bci_iv_2a_npz(data_path)
        return split_bci_dataset(x, y, float(dcfg["train_ratio"]), seed)

    raise ValueError(f"Unsupported data.dataset: {dataset_name}")
