from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from optimization.types import TrafficCase


class TrafficGenerator(ABC):
    """Base interface for traffic generation."""

    @abstractmethod
    def build_rows(self, case: TrafficCase) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def write_jsonl(self, case: TrafficCase, output_path: Path) -> Path:
        raise NotImplementedError

