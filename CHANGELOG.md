# Changelog

## Unreleased

- Replace the v0.1.0 per-request query-chunk workaround with one native
  batched operator call. Accept Q across the existing 262144-token context
  envelope; CUDA work is already tiled in 64-query blocks.
- Fix packed LSE addressing for fully masked rows in ragged unsplit batches
  and widen split-workspace address arithmetic before multiplication.
- Export the compiled query limit in artifact manifests so consumers can
  retire their 2048 scheduler clamp while retaining compatibility with old
  images. Add long-query, graph-replay, large-workspace and performance checks.

## 0.1.0 — 2026-10-04

- Split long full-causal prefill in mixed vLLM 0.29 batches into per-request
  paged calls of at most 2048 query tokens. Preserve causal offsets and exact
  GPU context lengths, including concurrent speculative decode. Add CPU
  routing regressions and a CUDA numerical/graph check. All nine CUDA
  regressions pass on both RTX 3090s with vLLM 0.29; vLLM 0.30 also passes
  the adapter checks. The previous artifact fails six cases with the
  sequence-envelope error. See BENCHMARKS.md for validation scope.

## 2026-09-12

- Use model-neutral `fa2_fp8kv` library and operator names, and
  `FA2_FP8KV_*` build/runtime settings.
- Clear V rows outside the visible sliding-window union before matrix
  multiplication. Discarded pages containing NaN could otherwise poison
  output even after their attention scores were masked.
- Add `check_window.py`: 12 noncausal CUDA Graph replay cases across
  lengths 2055, 32771 and 262143, splits 1/32/128, and discarded pages.
  The regression fails before the fix and passes after it. The 15 existing
  numerical cases and 18 target replay cases also passed after the fix.
- Extend the geometry-scoped vLLM adapter to two KV heads at D256 and
  four KV heads at D128, including noncausal window 2048 draft attention
  and full draft CUDA Graphs. Record the Qwen3.8-27B FP8/DFlash2 model
  comparison, context check, medium quality and large-image check.
- Add opt-in `FA2_FP8KV_PREFILL=1`: bounded FP8 KV unpacking, native FA2
  BF16 attention, FP32 merging and a paged-query fallback on workspace OOM.
  Build its operator with `build.py --prefill`. FP8 storage and FA2 decode
  remain unchanged. Record successful numerical checks and model prefill
  measurements; reject the 4096-token batch variants that failed with OOM.
