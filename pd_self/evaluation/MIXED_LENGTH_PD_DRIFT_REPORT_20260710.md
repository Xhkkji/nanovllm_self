# Mixed-Length PD Drift Report

## Scope

This note records the current diagnosis for the mixed-length batch PD divergence:

- short prompt + long prompt
- PD path matches monolithic for:
  - single prompt
  - equal-length batch
  - handoff payload structure
- PD path diverges only for:
  - mixed-length batch decode on the longer sequence

## Reproduction Shape

- short prompt: `What is a large language model?`
- long prompt: transformer explanation prompt
- `max_tokens=32`
- greedy decoding
- compare:
  - PD batched decode: short + long together
  - serial monolithic baseline
  - PD long-only decode after the same pair prefill

## Main Conclusion

The issue is not caused by:

- prefill handoff boundary
- payload serialization
- prompt-history KV corruption

The divergence appears to be a batch-dependent numerical drift during decode.

More specifically:

- the longer request is correct when decoded alone
- the longer request diverges only when decoded together with the short request
- the first visible token divergence happens later, but the KV drift starts earlier

## Key Findings

### 1. Prefill is not the root cause

For the long request, these matched between:

- pair prefill
- long-alone prefill

Matched fields:

- `token_ids`
- `num_cached_tokens`
- first generated token after handoff
- block table length

### 2. Handoff payload is not the root cause

The payload fields are structurally correct and the handoff tests pass.

### 3. Long request alone decode is correct

Both cases match the serial monolithic baseline:

- pair prefill -> long alone decode
- long-alone prefill -> long-alone decode

### 4. Only batched decode causes divergence

In batched decode:

- short request stays correct
- long request diverges

### 5. First token divergence

- first decode-step divergence: `step 7`
- long sequence position at that step: `42`
- batched decode emits: `17646`
- long-alone decode emits: `1614`

This is the first greedy flip. After that, autoregressive drift expands quickly.

### 6. The drift starts before the first token flip

For the long request, layer-0 KV already differs at the first batched decode step:

- position: `35`
- context length: `36`
- emitted token is still the same on both paths: `11`

But KV already differs:

- `k_cache max diff = 1.0`
- `v_cache max diff = 0.001953`

This means:

- token divergence is a later symptom
- numerical drift is injected earlier and written into KV cache first

### 7. First visible hidden-state drift appears in layer 0

At the first token-divergence step, layer-by-layer comparison showed:

- step `6`: long-seq layer outputs still match
- step `7`: first nonzero difference appears at `layer 0`
- later layers amplify it

### 8. The earliest operator-level drift is before attention aggregation

At long-seq decode position `35`, layer-0 comparison gave:

- `x_in`: `0.0`
- `ln1`: `0.0`
- `q`: `0.001953`
- `k`: `0.015625`
- `v`: `0.001953`
- `qn`: `0.03125`
- `kn`: `1.0`
- `qr`: `0.03125`
- `kr`: `1.0`
- `attn_output`: `0.003906`

Interpretation:

- input hidden state is identical
- drift first appears at `q/k/v` projection
- `k_norm` amplifies it noticeably
- then the slightly different values are written into KV cache
- later decode steps consume the already drifted suffix KV

## Likely Cause

Current evidence points to batch-shape-dependent low-precision numerical drift, not a scheduler or handoff logic bug.

The most suspicious zone is the decode forward path:

- `QwenDecoderLayer.forward()` in [qwen3.py](/home/xhk/nanovllm_self/nanovllm/models/qwen3.py:53)
- `q_proj/k_proj/v_proj` in [qwen3.py](/home/xhk/nanovllm_self/nanovllm/models/qwen3.py:65)
- `q_norm/k_norm` in [qwen3.py](/home/xhk/nanovllm_self/nanovllm/models/qwen3.py:69)
- unified attention entry in [attention.py](/home/xhk/nanovllm_self/nanovllm/layers/attention.py:372)
- KV write in [attention.py](/home/xhk/nanovllm_self/nanovllm/layers/attention.py:388)
- flash path dispatch in [attention.py](/home/xhk/nanovllm_self/nanovllm/layers/attention.py:411)
- packed mixed-shape input assembly in [model_runner.py](/home/xhk/nanovllm_self/nanovllm/engine/model_runner.py:413)

This is consistent with:

- `bf16/fp16` accumulation differences
- different GEMM tiling / reduction order under `batch=1` vs `batch=2`
- near-tied logits flipping under greedy decode

## Practical Interpretation

This should currently be treated as:

- a real mixed-length batch decode numerical drift
- not yet proven to be a hard logic bug

For infra work, this means:

- strict token-by-token equality across different batch shapes may be too strong as the only criterion
- we should track:
  - first divergence step
  - first divergence token
  - top-logit margin at divergence
  - whether drift starts before or after KV write

## Recommended Next Checks

### P0

Run one controlled experiment without changing scheduler logic:

- raise decode-side `q/k/v projection` precision
- raise `q_norm/k_norm` precision
- rerun mixed-length PD comparison

If the first divergence step moves much later, or disappears, that strongly supports the numerical-drift diagnosis.

### P1

Add a structured drift benchmark that records:

- prompt lengths
- batch shape
- first diff step
- first diff token
- top-2 logit margin at first diff

### P2

Separate correctness checks into:

- same-shape exact token parity
- cross-shape drift characterization

## Current Status

- mixed-length PD divergence: reproduced
- root cause narrowed to decode-time numerical drift
- first visible token divergence localized
- first KV drift localized
- first operator-level drift localized to layer-0 pre-attention projections / norms

This is enough to justify moving from "is the logic broken?" to "how much decode precision do we need to stabilize cross-batch behavior?"
