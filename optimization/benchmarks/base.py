from __future__ import annotations

from abc import ABC, abstractmethod

from optimization.types import BenchmarkCase, BenchmarkResult


class BenchmarkRunner(ABC):
    """Base interface for all benchmark runners."""

    @abstractmethod
    def run(self, case: BenchmarkCase) -> BenchmarkResult:
        raise NotImplementedError

