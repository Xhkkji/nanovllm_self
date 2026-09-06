from __future__ import annotations

from abc import ABC, abstractmethod

from optimization.types import BenchmarkCase, CandidateSpec


class OptimizationCandidate(ABC):
    """Base interface for a single optimization idea."""

    @abstractmethod
    def spec(self) -> CandidateSpec:
        raise NotImplementedError

    @abstractmethod
    def benchmark_cases(self) -> list[BenchmarkCase]:
        raise NotImplementedError

    @abstractmethod
    def change_plan(self) -> dict:
        """Return a human-readable plan of what would change."""
        raise NotImplementedError

