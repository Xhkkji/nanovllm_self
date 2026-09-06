from __future__ import annotations

from dataclasses import dataclass

from optimization.benchmarks.base import BenchmarkRunner
from optimization.candidates.base import OptimizationCandidate
from optimization.traffic.generator import TrafficGenerator
from optimization.types import BenchmarkResult, TrafficCase


@dataclass
class OptimizationLoop:
    """Thin orchestration shell for future candidate search."""

    candidates: list[OptimizationCandidate]
    traffic_generator: TrafficGenerator | None
    benchmark_runner: BenchmarkRunner | None

    def list_candidates(self) -> list[str]:
        return [candidate.spec().name for candidate in self.candidates]

    def run_one(self, candidate_name: str, traffic_case: TrafficCase | None = None) -> BenchmarkResult:
        raise NotImplementedError(
            "OptimizationLoop is a scaffold. Implement candidate execution here."
        )

