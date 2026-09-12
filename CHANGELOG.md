# Changelog

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
