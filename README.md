# FA2 FP8 KV for SM 8.6

An experimental FlashAttention-2 derivative for paged FP8 KV attention on
NVIDIA Ampere SM 8.6. This repository contains a CUDA extension and a narrow
vLLM adapter tested on two RTX 3090 GPUs. Model-specific measurements are
recorded in BENCHMARKS.md. It is not the complete FlashAttention library or a
drop-in replacement for `flash-attn`.

Ampere has no native FP8 compute. K/V use FP8 storage; the kernel converts
them to BF16 for attention math. The extension combines grouped-query
packing with asynchronous FP8 tile loads and deferred BF16 conversion.

## Scope and results

The kernel includes head dimensions 128 and 256. The model adapter has a
narrower contract: BF16 queries, FP8 E4M3 KV, NHD layout, and the tested
(head dimension, KV heads per rank) pairs (256, 1), (256, 2), and (128, 4).
It supports full causal attention and noncausal attention with a 2048-token
left window. Decode context parallelism and softcap are unsupported. Shared Qwen3.5 ancestry
does not establish compatibility with every Qwen3.5 model or configuration.

See [BENCHMARKS.md](BENCHMARKS.md) for measured model throughput, isolated
kernel timing, VRAM, correctness coverage, and long-context behavior.
The complete-model improvement does not imply a faster isolated kernel.
Quality OFF is measured; complete quality ON and soak validation are
not available. This is an experimental serving path.

## Upstream comparison

The original FA2 revision is pinned as a submodule in
[`upstream/flash-attention`](upstream/flash-attention). The
[CUDA source patch](patches/fa2-fp8kv.patch) can be checked with
`git submodule update --init upstream/flash-attention` followed by
`python3 check_upstream.py`. See [the mapping and verification steps](upstream/README.md)
for the four modified headers, twelve unchanged headers and two new CUDA files.

## Build

The reproducible environment is the pinned vLLM 0.27.1 image below.
It provides CUDA, PyTorch, and FlashInfer headers. Fetch the pinned CUTLASS
dependency on the host first: the container does not provide Git.
Compilation does not need GPU access.
The packaged source was successfully built and loaded with this image
and `TORCH_CUDA_ARCH_LIST=8.6` on September 12, 2026.

```bash
git clone https://github.com/AntonProkopyev/fa2-fp8kv-sm86.git
cd fa2-fp8kv-sm86
bash setup.sh --fetch-only
docker run --rm --runtime runc \
  -e NVIDIA_VISIBLE_DEVICES=void -e MAX_JOBS=2 -e TORCH_CUDA_ARCH_LIST=8.6 \
  -v "$PWD:/work" -w /work --entrypoint python3 \
  vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967 \
  build.py --pipeline --prefill
```

The output is `build-pipeline/fa2_fp8kv.so`; the operator namespace is
`torch.ops.fa2_fp8kv`.
`--prefill` also builds `build-prefill/fa2_fp8kv_prefill.so` for the optional
bounded KV unpacking path. It keeps persistent KV in FP8 and runs attention
through native FA2 with temporary BF16 inputs. Enable it explicitly using
the integration instructions below.
For a compatible local CUDA/PyTorch environment, `bash setup.sh` fetches
the dependency and builds the same extension. `build.py --help` lists
header-path overrides. Set `TORCH_CUDA_ARCH_LIST=8.6` explicitly in Docker:
the image's broader architecture list overrides the script's default.

## GPU checks

Run these when the GPU is free. They allocate GPU memory and exercise
long-context cases. Use the same image that built the extension.

```bash
docker run --rm --gpus all --ipc host \
  -v "$PWD:/work" -w /work --entrypoint bash \
  vllm/vllm-openai:v0.27.1@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967 \
  -lc 'set -e
    python3 bench.py --library build-pipeline/fa2_fp8kv.so --check-only
    python3 check_replay.py build-pipeline/fa2_fp8kv.so
    python3 check_window.py build-pipeline/fa2_fp8kv.so
    python3 check_full.py --library build-pipeline/fa2_fp8kv.so --full
    python3 check_prefill.py build-prefill/fa2_fp8kv_prefill.so --paged-library build-pipeline/fa2_fp8kv.so'
```

For kernel timings, run `bench.py` with the same `--library` argument and
omit `--check-only`. These timings are not model tokens per second.
`bench_prefill.py build-pipeline/fa2_fp8kv.so build-prefill/fa2_fp8kv_prefill.so`
compares complete prefill operations, including unpacking and FP32 merging.
Development results are recorded in [BENCHMARKS.md](BENCHMARKS.md);
publication does not imply a fresh GPU run of the packaged source.

## Serving integration

The [vLLM 0.29 FlashAttention plugin](integration/vllm_plugin/README.md) is under
validation. It selects `FLASH_ATTN` through vLLM's plugin registry and leaves
FlashInfer source files untouched. The instructions below describe the source
overlay for vLLM 0.27.1.

Follow [the vLLM adapter instructions](integration/vllm/README.md).
The adapter overlays an engine source file and must stay paired with its
pinned engine version. No model weights or compiled binaries are included.

## Contributing and licenses

Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing kernels or reporting
performance. File reproducible failures in
[Issues](https://github.com/AntonProkopyev/fa2-fp8kv-sm86/issues).

See [NOTICE](NOTICE), [LICENSE](LICENSE), and [licenses/](licenses/) for
upstream attribution and component licenses. The adapter and project code
use Apache-2.0; copied FlashAttention headers retain their BSD license.
