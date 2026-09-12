# Contributing

Keep changes focused on the documented SM 8.6 FP8 KV path. Open an issue
before broadening the model adapter or replacing dependency pins.

## Validation

Build with the image and commands in [README.md](README.md). Run the three
GPU checks there after kernel or adapter changes when an SM 8.6 GPU is
available. Report checks you could not run. Preserve reference comparisons
for non-unit scales, graph replay, ragged sequences, empty rows, strided
buffers, and long KV lengths. Passing a kernel check does not validate
model-level decoding or speculative acceptance.

For adapter changes, report the model revision, engine image digest,
effective launch arguments, cache format, context limit, and speculative
configuration. Check text generation, streaming, tools, vision, MTP,
long-context recall, and rejection of oversized requests for any claimed
serving configuration. Do not silently widen the supported contract.

## Performance reports

Compare the same model weights, prompts, sampler, context length, and
concurrency. Use three warmups and five measured requests per prompt.
Report wall-time throughput and decode throughput separately, with spread,
TTFT, and per-card peak VRAM. Include GPU model, power limit, interconnect,
driver, and background load. State the GPU sampling interval.

For comparison with the published model A/B, use an 800-word essay with
max_tokens=1000 and quicksort code with max_tokens=800. Send temperature=0.6,
top_p=0.95, top_k=20, and min_p=0 explicitly, with thinking OFF. Preserve
the exact prompt text in your report. Do not translate isolated kernel
microseconds into model tokens per second.

## Pull requests

Explain the failing case or intended behavior, the change, and validation.
Include compact reproduction commands and relevant logs. Remove credentials,
private prompts, host identifiers, and private paths before attaching logs.
Do not commit weights, generated binaries, build directories, downloaded
dependencies, or benchmark output containing private data.

Preserve upstream copyright and license notices. Identify upstream source
revisions when importing code. Update documentation when a supported
constraint or measured behavior changes; distinguish measurements from
estimates and pending checks.
