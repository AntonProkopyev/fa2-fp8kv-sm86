# FlashAttention plugin for vLLM 0.29

This experimental integration registers through `vllm.general_plugins` and
`register_backend(AttentionBackendEnum.FLASH_ATTN, ...)`. It uses the native
FlashAttention metadata and KV-write contract. It does not edit vLLM source
files or inherit its FlashInfer implementation.

Select `--attention-backend FLASH_ATTN` for the target and
`"attention_backend":"FLASH_ATTN"` in the DFlash configuration. With SM86,
SM89 or SM120
and FP8 E4M3 KV, the plugin selects the custom CUDA operator; other native
FlashAttention configurations delegate to the stock implementation. This
does not establish runtime validation on SM89 or SM120. The image recipe
builds native code for all three targets; only SM86 has been exercised on a
physical GPU. SM89/SM120 execution and performance remain unverified.

The plugin wheel contains the Python adapter and prefill modules. CUDA
libraries are supplied separately through `FA2_FP8KV_LIBRARY` and
`FA2_FP8KV_PREFILL_LIBRARY`. The source build recipe for their artifact image
is `containers/Dockerfile`; the published digest, ABI and extraction command
are in [`containers/README.md`](../../containers/README.md).
The build records the CUDA/PyTorch/Python ABI and payload checksums.

On September 13, the new implementation passed native KV-write byte checks,
FP32 attention references and CUDA Graph replay in `check_backend.py`.
The eager model run passed verify-full 10/10 including vision, and a controlled
medium quality run scored 64/75, matching the stock-FI layout-control arm.
The graph-enabled run passed verify-full 10/10 and scored 63/75, matching the
compiled stock control's total. Matched whole-phase counters did not reproduce
a doubled draft acceptance or flat tail. The prebuilt artifact also passed
261K text and 260K combined vision requests with cached follow-ups on SM86.
Continuous soak passed 25/25 turns without errors or VRAM growth. The stress
response probes passed through 240K, but its 1024 MiB free-memory gate failed
at 1021 MiB on the tested full-context profile. See the detailed report in
[club-3090 PR #1274](https://github.com/noonghunna/club-3090/pull/1274).
These are experimental results, not a production guarantee or a claim of
bit-identical model outputs.

`integration/vllm/flashinfer.py` provides the source overlay for vLLM 0.27.1.
Do not combine that overlay with this plugin.

## Long prefill with concurrent requests

The paged FP8 operator accepts at most 2048 query tokens per request per
call. For full causal attention, the adapter splits longer mixed batches
into per-request chunks within that limit. It keeps the original request
order, block-table rows, and causal positions. The scheduler's batch budget
and prefill threshold do not need to be lowered for this path.

Query boundaries come from vLLM's CPU metadata. Context lengths stay on the
GPU: the CPU upper bound can include rejected speculative tokens on decode
rows. Each chunk subtracts only the query tokens after it from the exact
GPU context length. Empty rows are skipped. Batches within the kernel limit
and the existing single-request native FA2 prefill route retain their paths.
This change does not extend long-query support to sliding-window,
noncausal, or full CUDA Graph prefill execution.

Run the routing regression without GPU access in the pinned plugin image:

```bash
docker run --rm --runtime runc --network none \
  -e NVIDIA_VISIBLE_DEVICES=void -e VLLM_PLUGINS= \
  -e PYTHONUTF8=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/work:/work/integration/vllm_plugin \
  -v "$PWD:/work:ro" -w /work --entrypoint python3 \
  vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1 \
  check_mixed_prefill.py -v
```

The CPU run substitutes an FP32 reference for the CUDA operator and enforces
its 2048-token query limit. It tests adapter routing and causal chunk offsets;
it does not validate CUDA numerics or graph execution. Five regression cases
fail on the original adapter and all eight CPU checks pass with the fix.

In a matching vLLM 0.29 CUDA environment with the existing libraries built
and `FA2_FP8KV_LIBRARY` / `FA2_FP8KV_PREFILL_LIBRARY` set, run:

```bash
export PYTHONPATH="$PWD:$PWD/integration/vllm_plugin${PYTHONPATH:+:$PYTHONPATH}"
python3 check_mixed_prefill.py --device cuda -v
python3 check_backend.py
```

The CUDA mode uses the real operators and adds a 262143-token context case
and decode graph replay with changed GPU sequence lengths. Both modes check
mixed request ordering, two long prefills, an empty row, the 2048/2049
boundary, missing CPU context bounds, OOM fallback, non-unit KV scales, and
strided outputs with padding guards. On October 4, 2026, all nine CUDA checks
passed on each RTX 3090 in the pinned vLLM 0.29 image. They also passed in
vLLM 0.30.0, together with `check_backend.py`. The previous artifact failed
six of the same CUDA cases with the sequence-envelope error. See
`BENCHMARKS.md` for the validation scope. No throughput improvement is claimed.
