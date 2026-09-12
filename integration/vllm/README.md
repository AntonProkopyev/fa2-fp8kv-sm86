# vLLM adapter

This source overlay targets vLLM 0.27.1 at image digest
`sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`.
Do not apply it to an arbitrary vLLM version: it replaces the entire
`vllm.v1.attention.backends.flashinfer` module.

Measured configurations use two RTX 3090 GPUs, TP=2, vision and a
262144-token context limit: Ornith-1.5-35B-A3B-FP8 with MTP=3, and
Qwen3.8-27B FP8 with DFlash2 W4A16 n=7.
Supported decoder inputs are BF16 queries, FP8 E4M3 KV, NHD layout, and
(head dimension, KV heads per rank) pairs (256, 1), (256, 2), and (128, 4).
The adapter supports full causal attention and a noncausal 2048-token left
window. Decode context parallelism, other window sizes and softcap are
unsupported. These checks do not establish general model compatibility.

## Mounts and settings

Build the library using the root README. Add these read-only mounts to
your pinned vLLM container, resolving source paths from the repository root:

```yaml
volumes:
  - ./integration/vllm/flashinfer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:ro
  - ./build-pipeline/fa2_fp8kv.so:/opt/fa2-fp8kv/fa2_fp8kv.so:ro
environment:
  FA2_FP8KV_LIBRARY: /opt/fa2-fp8kv/fa2_fp8kv.so
  FA2_FP8KV_SPLITS: "128"
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

For DFlash2, replace the MTP speculative configuration with:

```json
{"method":"dflash","model":"/draft","num_speculative_tokens":7,"attention_backend":"FLASHINFER","kv_cache_dtype":"fp8_e4m3"}
```

Mount the compatible draft checkpoint at `/draft`. DFlash2 also requires
compatible engine support: the measured vLLM 0.27.1 run used a local
club-3090 DFlash2 backport, which this repository does not install.
The adapter captures noncausal draft attention in a full CUDA Graph;
look for `Capturing dflash2 CUDA graphs (FULL)` in the startup log.
Switching the draft to stock FA2 with BF16 KV exceeded the measured rig's
KV memory budget at 262144 context. The custom path retains FP8 KV.

`FLASHINFER` remains the registry enum used to select the cache layout.
The overlay routes normal decoder attention, including MTP attention,
and supported DFlash2 attention, through the FA2 extension. It does not replace the model's Gated DeltaNet
or vision encoder kernels. Confirm the startup log contains
`FA2 FP8 KV native GQA/async backend enabled`.

The 262144-token setting was checked with a 261000-token input; it does
not guarantee that every multimodal workload fits. See the root
[BENCHMARKS.md](../../BENCHMARKS.md) for the measured configuration and
pending quality/soak validation. Keep a copy of your previous serving
configuration so the overlay can be removed and the service restarted
with its original backend and KV settings.
