# FlashAttention plugin for vLLM 0.29

This experimental integration registers through `vllm.general_plugins` and
`register_backend(AttentionBackendEnum.FLASH_ATTN, ...)`. It uses the native
FlashAttention metadata and KV-write contract. It does not edit vLLM source
files or inherit its FlashInfer implementation.

Select `--attention-backend FLASH_ATTN` for the target and
`"attention_backend":"FLASH_ATTN"` in the DFlash configuration. With SM86
and FP8 E4M3 KV, the plugin selects the custom CUDA operator; other native
FlashAttention configurations delegate to the stock implementation. This
does not establish FP8 support on SM89 or SM120: those runtime ports remain
unvalidated and the current CUDA operator is SM86-gated.

The plugin wheel contains the Python adapter and prefill modules. CUDA
libraries are supplied separately through `FA2_FP8KV_LIBRARY` and
`FA2_FP8KV_PREFILL_LIBRARY`. The source build recipe for their artifact image
is `containers/Dockerfile`; publication and delivery integration are in progress.
The build records the CUDA/PyTorch/Python ABI and payload checksums.

On September 13, the new implementation passed native KV-write byte checks,
FP32 attention references and CUDA Graph replay in `check_backend.py`.
The eager model run passed verify-full 10/10 including vision, and a controlled
medium quality run scored 64/75, matching the stock-FI layout-control arm.
Complete serving-graph validation is still in progress. These are experimental
results, not a production guarantee or a claim of bit-identical model outputs.

`integration/vllm/flashinfer.py` provides the source overlay for vLLM 0.27.1.
Do not combine that overlay with this plugin.
