from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateSpec:
    """Description of one optimization candidate."""

    name: str
    target: str
    baseline: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    """A benchmark input used by operator or serving experiments."""

    name: str
    model_path: str = ""
    batch_sizes: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    dtypes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrafficCase:
    """A synthetic or recorded traffic recipe."""

    name: str
    dataset_path: str = ""
    profile: str = ""
    concurrency: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """A single benchmark run result."""

    candidate: str
    case: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_dir: Path | None = None

