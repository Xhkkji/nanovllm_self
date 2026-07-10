# PD Evaluation

## Recommended Run Order

Use the `pytorch` environment and run the modules separately to reduce GPU peak memory:

```bash
cd /home/xhk/nanovllm_self

PYTHONPATH=/home/xhk/nanovllm_self /home/xhk/miniconda3/envs/pytorch/bin/python -m unittest pd_self.evaluation.test_pd_vs_monolithic
PYTHONPATH=/home/xhk/nanovllm_self /home/xhk/miniconda3/envs/pytorch/bin/python -m unittest pd_self.evaluation.test_handoff_payload
PYTHONPATH=/home/xhk/nanovllm_self /home/xhk/miniconda3/envs/pytorch/bin/python -m unittest pd_self.evaluation.test_pd_batch_edges
PYTHONPATH=/home/xhk/nanovllm_self /home/xhk/miniconda3/envs/pytorch/bin/python -m unittest pd_self.evaluation.test_pd_per_seq_sampling_eval_only
```

## What Each File Covers

- `test_pd_vs_monolithic.py`
  - single prompt greedy correctness
  - equal-length two-prompt batch correctness

- `test_handoff_payload.py`
  - payload fields after prefill handoff
  - `max_tokens=1` finished-in-prefill behavior
  - multi-payload order and id stability

- `test_pd_batch_edges.py`
  - `max_tokens=1` batch edge case
  - shorter prompt should hand off no later than longer prompt
  - mixed-length batch correctness regression

- `test_pd_per_seq_sampling_eval_only.py`
  - evaluation-only workaround for per-sequence sampling params inside one batch

- `MIXED_LENGTH_PD_DRIFT_REPORT_20260710.md`
  - formal diagnosis note for the current mixed-length batch PD divergence
  - records where the drift first appears and why it currently looks numerical rather than scheduling-related

## Known Failure

- `test_pd_batch_edges.PDBatchEdgeCasesTest.test_mixed_prompt_lengths_match_monolithic`
  - marked as `expectedFailure`
  - this currently exposes a real mixed-length batch PD divergence on the longer sequence
