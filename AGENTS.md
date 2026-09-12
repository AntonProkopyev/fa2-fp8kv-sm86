# Agent guide

Read README.md, BENCHMARKS.md, NOTICE, and integration/vllm/README.md before
changing the kernel or adapter. This is an experimental FA2 derivative,
not the complete FlashAttention library.

Keep changes scoped. Preserve upstream notices and dependency pins. Do not
claim native FP8 math on SM 8.6: FP8 is KV storage and attention uses BF16.
Do not generalize Ornith validation to all Qwen3.5 models.

Use the documented pinned image for builds. Fetch CUTLASS on the host;
the image has no Git. Do not run GPU checks against an occupied serving
GPU without authorization to share or stop that workload.

Run relevant checks from README.md after behavioral changes. Report what
passed, what failed, and what was not run. Model throughput claims require
model benchmarks; kernel timing alone is insufficient. Preserve numerical
reference and CUDA Graph replay coverage.

Do not commit model weights, binaries, downloaded dependencies, private
paths, credentials, or raw private prompts. CLAUDE.md is a symlink to this
file; keep one canonical guide.
