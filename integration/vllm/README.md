# vLLM adapter

This source overlay targets vLLM 0.27.1 at image digest
`sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`.
Do not apply it to an arbitrary vLLM version: it replaces the entire
`vllm.v1.attention.backends.flashinfer` module.

The tested model is `ornith-ai/Ornith-1.5-35B-A3B-FP8` with two RTX 3090
GPUs, TP=2, MTP=3, vision enabled, and a 262144-token context limit.
Supported decoder inputs are BF16 queries, FP8 E4M3 KV, head dimension 256,
one KV head per rank, and NHD cache layout. Decode context parallelism,
sliding-window attention, and softcap are not supported by this adapter.
The kernel's broader head-dimension support does not widen this contract.

## Mounts and settings

Build the library using the root README. Add these read-only mounts to
your pinned vLLM container, resolving source paths from the repository root:

```yaml
volumes:
  - ./integration/vllm/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:ro
  - ./build-pipeline/ornith_fa2_fp8.so:/opt/ornith-fa2/ornith_fa2_fp8.so:ro
environment:
  FA2_FP8KV_LIBRARY: /opt/ornith-fa2/ornith_fa2_fp8.so
  ORNITH_FA2_SPLITS: "128"
```

Use these serving arguments with your model path and existing service
settings. This is an argument fragment, not a complete Compose service:

```text
--tensor-parallel-size 2
--dtype bfloat16
--kv-cache-dtype fp8_e4m3
--attention-backend FLASHINFER
--max-model-len 262144
--max-num-seqs 1
--max-num-batched-tokens 2048
--gpu-memory-utilization 0.95
--limit-mm-per-prompt '{"image":1,"video":0}'
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
--no-enable-prefix-caching
```

On the tested PCIe-only two-RTX-3090 topology, also pass
`--disable-custom-all-reduce`. Evaluate that setting for your actual
interconnect instead of copying it to an NVLink system.

`FLASHINFER` remains the registry enum used to select the cache layout.
The overlay routes normal decoder attention, including MTP attention,
through the FA2 extension. It does not replace the model's Gated DeltaNet
or vision encoder kernels. Confirm the startup log contains
`ORNITH FA2 native GQA/async backend enabled`.

The 262144-token setting was checked with a 261000-token input; it does
not guarantee that every multimodal workload fits. See the root
[BENCHMARKS.md](../../BENCHMARKS.md) for the measured configuration and
pending quality/soak validation. Keep a copy of your previous serving
configuration so the overlay can be removed and the service restarted
with its original backend and KV settings.
