import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VersionEntry:
    """A single code version with its test results."""

    code: str
    test_passed: int
    test_total: int
    timestamp: float = field(default_factory=time.time)
    label: str = ""  # "initial", "repair_1", etc.

    @property
    def score(self) -> float:
        if self.test_total == 0:
            return 0.0
        return self.test_passed / self.test_total

    def __gt__(self, other: "VersionEntry") -> bool:
        return self.score > other.score


class GenerationMemory:
    """Tracks code versions and test results with rollback support.

    Persists to .sisyphus/memory.json so state survives restarts.
    """

    def __init__(self, memory_dir: str = ".sisyphus"):
        self._path = Path(memory_dir) / "memory.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.versions: list[VersionEntry] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self.versions = [VersionEntry(**v) for v in data]
            except (json.JSONDecodeError, TypeError):
                self.versions = []

    def _save(self):
        data = [asdict(v) for v in self.versions]
        self._path.write_text(json.dumps(data, indent=2))

    def add(self, code: str, passed: int, total: int, label: str = ""):
        entry = VersionEntry(
            code=code, test_passed=passed, test_total=total, label=label
        )
        self.versions.append(entry)
        self._save()

    @property
    def best(self) -> Optional[VersionEntry]:
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.score)

    @property
    def last(self) -> Optional[VersionEntry]:
        return self.versions[-1] if self.versions else None

    def should_rollback(self, new_passed: int, new_total: int) -> bool:
        """Return True if the new version is worse than the best so far."""
        best_entry = self.best
        if best_entry is None:
            return False
        best_score = best_entry.score
        if new_total == 0:
            return True
        return (new_passed / new_total) < best_score

    def clear(self):
        self.versions = []
        if self._path.exists():
            self._path.unlink()
