---
name: Bug report
about: Report a reproducible build, kernel, or serving failure
title: ''
labels: bug
assignees: ''
---

## Expected and actual behavior

Describe the failure and the smallest input that triggers it.

## Environment

- Repository commit:
- GPU model, count, and compute capability:
- Driver and CUDA versions:
- PyTorch version and container image digest:
- CUTLASS revision:
- Model and revision, if using the adapter:
- TP, KV format, context limit, and MTP configuration:

## Reproduction

Include exact build and run commands, effective launch arguments, and
relevant error output. Remove credentials, private paths, and private
prompts before posting.

## Checks

State which README GPU checks passed, failed, or were not run. For a
numerical failure, include shapes, strides, scales, and reference error.
For a performance regression, include the comparison configuration,
warmup/run counts, GPU background load, and per-card peak VRAM.
