from __future__ import annotations

from pathlib import Path
from typing import Iterable


class EEGMMIDBAdapter:
    """Placeholder adapter for PhysioNet eegmmidb.

    This class documents the expected interface for the next stage where EDF files
    are parsed into tensors [channels, time_steps] and labels.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def check_exists(self) -> bool:
        return self.root.exists() and any(self.root.rglob("*.edf"))

    def subject_dirs(self) -> list[Path]:
        return sorted([p for p in self.root.glob("**/S*") if p.is_dir()])

    def iter_records(self) -> Iterable[Path]:
        for p in sorted(self.root.rglob("*.edf")):
            yield p
