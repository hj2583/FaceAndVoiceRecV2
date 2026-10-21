from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeTree:
    project_root: Path

    @property
    def data_root(self) -> Path:
        return self.project_root / "data"

    def legacy_path(self, relative_path: str) -> Path:
        return self.project_root / relative_path

    def data_path(self, relative_path: str) -> Path:
        return self.data_root / relative_path

    def write_legacy(self, relative_path: str, content: bytes) -> Path:
        target = self.legacy_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target

    def write_data(self, relative_path: str, content: bytes) -> Path:
        target = self.data_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target
