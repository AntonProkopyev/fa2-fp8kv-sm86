"""Compare one batched launch with the v0.1.0 per-request chunk workaround."""
import argparse
import json
import math
import os
import statistics
import time

import torch


def measure(mode, queries, context, dim, heads, kv_heads):
    torch.manual_seed(1358)
    page = 16
    blocks = math.ceil(context / page)
    count = sum(queries)
    query = torch.randn(count, heads, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(len(queries) * blocks, page, kv_heads, 2 * dim,
                        device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    keys, values = cache.split(dim, dim=-1)
    table = torch.randperm(len(queries) * blocks, device="cuda").reshape(len(queries), blocks).int()
    starts_cpu = [0]
    for length in queries:
        starts_cpu.append(starts_cpu[-1] + length)
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    lengths = torch.full((len(queries),), context, device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    key_scale = torch.tensor([0.5], device="cuda")
    value_scale = torch.tensor([1.75], device="cuda")

    def forward():
        if mode == "native":
            torch.ops.fa2_fp8kv.forward(query, keys, values, output, starts,
                lengths, table, key_scale, value_scale, max(queries), context,
                True, -1, -1, dim ** -0.5, 1, False)
            return
        # Kept only as a benchmark baseline. This is the v0.1.0 loop, including
        # its per-launch metadata allocations and host-to-device copies.
        for row, (begin, end) in enumerate(zip(starts_cpu, starts_cpu[1:])):
            for first in range(begin, end, 2048):
                size = min(2048, end - first)
                visible = lengths[row:row + 1] - (end - first - size)
                cu_q = torch.tensor([0, size], device="cuda", dtype=torch.int32)
                torch.ops.fa2_fp8kv.forward(query[first:first + size], keys,
                    values, output[first:first + size], cu_q, visible,
                    table[row:row + 1], key_scale, value_scale, size, context,
                    True, -1, -1, dim ** -0.5, 1, False)

    # Each of the 3 warmup / 5 measured samples contains enough work to avoid
    # clock ramp and Python scheduling dominating the shortest operation.
    iterations = 64 if context <= 4096 else 4 if context <= 32768 else 1
    for _ in range(3):
        for _ in range(iterations):
            forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    gpu_ms, wall_ms = [], []
    for _ in range(5):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        begin.record()
        for _ in range(iterations):
            forward()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - wall) * 1000 / iterations)
        gpu_ms.append(begin.elapsed_time(end) / iterations)
    return {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "mode": mode, "queries": queries, "context": context,
            "dim": dim, "heads": heads, "kv_heads": kv_heads,
            "warmups": 3, "runs": 5, "iterations_per_sample": iterations,
            "gpu_ms": gpu_ms, "wall_ms": wall_ms,
            "gpu_mean_ms": statistics.mean(gpu_ms),
            "wall_mean_ms": statistics.mean(wall_ms),
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "extra_allocated_mib": (torch.cuda.max_memory_allocated() - allocated) / 2**20}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True)
parser.add_argument("--mode", choices=("native", "chunked"), required=True)
args = parser.parse_args()
torch.ops.load_library(args.library)
with torch.inference_mode():
    for queries, context in [((1, 2049), 4096), ((1, 4096), 32768),
                             ((4096, 4096), 32768), ((8, 8192), 131072)]:
        print(json.dumps(measure(args.mode, queries, context, 256, 12, 2)), flush=True)
