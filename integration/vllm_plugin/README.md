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

The paged FP8 operator accepts queries through its 262144-token context
envelope. The former 2048 bound was a validation restriction in the C++
entrypoint, not a kernel tile dimension: each CUDA block processes 64 query
rows and the grid covers the longest query. Long prefill uses one KV split,
so it does not allocate the padded split-KV accumulation buffers.

Mixed batches use one operator call with the original GPU query boundaries,
sequence lengths and block table. The Python query-chunk loop introduced in
v0.1.0 is removed. Exact GPU lengths avoid optimistic CPU context bounds for
speculative decode. The single-request bounded BF16 prefill optimization
remains; on workspace OOM it falls through to the same batched paged operator.

Long queries also work in the operator's sliding-window/noncausal paths and
CUDA Graph replay. vLLM retains its own graph eligibility policy; supporting
the operator in a graph does not change which model steps vLLM captures.
The artifact manifest records `max_query_tokens` by querying the compiled
operator. Consumers can remove their old 2048 scheduler clamp for that
verified artifact and retain it for older images without the capability.

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

The CPU run substitutes an FP32 reference for the CUDA operator and asserts
that every request reaches it in one batched call. It does not validate CUDA
numerics or graph execution. GPU regressions exercise the actual operator.

In a matching vLLM 0.29 CUDA environment with the existing libraries built
and `FA2_FP8KV_LIBRARY` / `FA2_FP8KV_PREFILL_LIBRARY` set, run:

```bash
export PYTHONPATH="$PWD:$PWD/integration/vllm_plugin${PYTHONPATH:+:$PYTHONPATH}"
python3 check_mixed_prefill.py --device cuda -v
python3 check_backend.py
python3 check_full.py --library build-pipeline/fa2_fp8kv.so --full --long-query
python3 check_replay.py build-pipeline/fa2_fp8kv.so --long-query
```

Coverage includes mixed request order, two long prefills, empty rows,
2048/2049 boundaries, queries up to 262144, non-unit scales, strided buffers,
OOM fallback, and graph replay with changing GPU query boundaries and lengths.
`check_full.py --library build-pipeline/fa2_fp8kv.so --large-workspace`
separately tests 64-bit offsets with more than 2^31 FP32 scratch elements;
it needs about 9 GiB of free VRAM. `bench_mixed.py` compares native batching
with the v0.1.0 workaround using three warmups and five measured runs.
These operation timings are not model throughput. See `BENCHMARKS.md` for
measured results and remaining validation limits.
