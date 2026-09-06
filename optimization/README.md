# Optimization Scaffold

This package is the outer orchestration layer for:

- operator-level experiments
- micro benchmarks
- traffic generation
- agent-driven optimization loops

The runtime, PD serving, and agent demo stay in the existing packages:

- `nanovllm/`
- `pd_self/`
- `claude_self/`

## Suggested flow

1. Define a candidate.
2. Build benchmark cases.
3. Generate traffic if needed.
4. Run the benchmark.
5. Compare results.
6. Record artifacts.

## Directory roles

- `candidates/`: candidate metadata and interfaces
- `benchmarks/`: micro benchmark adapters
- `traffic/`: request/trace generators
- `loop/`: optimization orchestration
- `reports/`: summaries and comparisons
- `artifacts/`: experiment outputs

This package intentionally starts as a skeleton.
