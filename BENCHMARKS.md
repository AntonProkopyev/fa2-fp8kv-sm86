# Measurements

Measured September 12, 2026 on two RTX 3090 GPUs (SM 8.6), PCIe P2P,
no NVLink, 250 W power limits. Model: ornith-ai/Ornith-1.5-35B-A3B-FP8,
revision fab11c26e2325a42f4b32da0249c819a0bade1b1.

## Complete model A/B

vLLM 0.27.1, TP=2, MTP=3, vision enabled, context limit 262144,
FP8 E4M3 KV, full CUDA Graph decode, one request at a time, 95% memory
budget. Both sides use the same FP8 model weights.

Three warmups and five measured runs per prompt, thinking OFF.
Narrative: an 800-word essay, max_tokens=1000. Code: quicksort,
max_tokens=800. Sampler: temperature=0.6, top_p=0.95, top_k=20, min_p=0.
The engine warns that min_p is unsupported with speculative decoding;
the requested value is zero on both sides.

| Metric | FlashInfer E4M3 | FA2 GQA + async E4M3 |
|---|---:|---:|
| Narrative decode tokens/s | 133.28 ± 2.53 | 174.20 ± 5.16 |
| Code decode tokens/s | 176.49 ± 6.76 | 222.34 ± 7.37 |
| Narrative wall tokens/s | 131.28 ± 2.44 | 170.86 ± 4.93 |
| Code wall tokens/s | 171.98 ± 6.42 | 215.54 ± 7.00 |
| TTFT narrative / code | 114 / 119 ms | 112 / 112 ms |
| Peak VRAM GPU0 / GPU1 after ready | 22558 / 22558 MiB | 22156 / 22156 MiB |

Values after ± are standard deviations across five requests.
This is one complete A/B, not a statistical significance claim.
GPU memory was sampled every 500 ms; shorter peaks may be missed.
The FA2 integration changes metadata handling as well as the attention
kernel. Their separate contributions have not been measured.

A subsequent FA2-only run on the main service measured 172.18 ± 5.63
narrative and 223.89 ± 6.76 code decode tokens/s; wall rates were
168.86 ± 5.53 and 216.94 ± 6.62. Peak VRAM was 22158 / 22156 MiB.

## Kernel microbenchmark

One rank: eight query heads, one KV head, head dimension 256, BF16 Q,
FP8 E4M3 K/V, page size 16, unit scales. CUDA events, three warmups and
five measurements of 100 graph replays. Split count is selected from
KV length, capped at 128. These are microseconds, not model tokens/s.

| Query tokens / KV tokens | Earlier FA2 port | GQA + async | FlashInfer |
|---|---:|---:|---:|
| 4 / 32768 | 1331 | 194 | 71 |
| 4 / 262144 | 8013 | 789 | 412 |
| 2048 / 4096 | 2523 | 1997 | 1324 |

At 262K, GQA alone measured 1259 us and async alone 5141 us.
An independent repeat measured 780 us for the combination and 405 us
for FlashInfer. FlashInfer is faster in this isolated kernel test.

## Correctness and long context

The development build passed six numerical shapes against an FP32
reference, 18 graph replay cases per build, and 15 legacy kernel cases.
Coverage includes non-unit scales, ragged sequences, empty rows,
strided KV/output, splits 1/4/128, finite FP8 byte values, head dimensions
128/256, sliding-window attention, and KV length 262143.

The model passed nine applicable functional checks, including tool use,
streaming, reasoning, MTP and vision (4/4). A request with 261000 input
tokens recovered a control string in 146 seconds; exceeding the context
limit returned HTTP 400. The earlier FlashInfer E5M2 profile took 100.7
seconds on that check, so this FA2 path does not improve long prefill.
The cache formats differ in that long-context comparison.

Full quality OFF completed all 150 scenarios: 117/150 (78%) at pass@1.
ToolCall 14/15, InstructFollow 13/15, StructOutput 15/15, DataExtract
10/15, ReasonMath 13/15, BugFind 13/15, Hermes 12/20, CLI 27/40.
The harness was benchlocal-cli 0.9.9 with explicit thinking OFF.
Quality ON and soak were still running when these results were prepared.
No production stability claim is made from the speed test.
