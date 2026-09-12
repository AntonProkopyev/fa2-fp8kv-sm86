"""Matched prefill-operation timings, including FP8 unpacking and merging."""
import argparse
import json
import math
import statistics
import time
import torch
from fa2_prefill import Fa2Prefill

parser = argparse.ArgumentParser()
parser.add_argument("paged_library")
parser.add_argument("prefill_library")
args = parser.parse_args()
torch.ops.load_library(args.paged_library)
torch.ops.load_library(args.prefill_library)
torch.manual_seed(257)

with torch.inference_mode():
    for length in (32768, 131072, 261000):
        qlen, dim, heads, kv_heads, page = 1648, 256, 12, 2, 16
        blocks = math.ceil(length / page)
        q = torch.randn((qlen, heads, dim), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((blocks, page, kv_heads, dim), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        v = torch.randn_like(k.float()).to(torch.float8_e4m3fn)
        table = torch.arange(blocks, device="cuda", dtype=torch.int32)[None]
        starts = torch.tensor([0, qlen], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
        scale = torch.ones(1, device="cuda")
        first = torch.empty_like(q)
        second = torch.empty_like(q)
        operations = {
            "paged_fp8": lambda: torch.ops.fa2_fp8kv.forward(q, k, v, first, starts, lengths, table,
                scale, scale, qlen, length, True, -1, -1, 1/16, 1, False),
            "unpack_fa2": lambda: Fa2Prefill().forward(q, k, v, table, scale, scale, length, 1/16, second),
        }
        row = {"q": qlen, "kv": length, "q_heads": heads, "kv_heads": kv_heads, "head_dim": dim}
        for name, operation in operations.items():
            for _ in range(3): operation()
            torch.cuda.synchronize()
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                operation()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
            row[name] = {"mean_ms": statistics.mean(samples), "std_ms": statistics.stdev(samples)}
        torch.testing.assert_close(first, second, atol=0.025, rtol=0.025)
        row["max_abs_difference"] = float((first.float() - second.float()).abs().max())
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        print(json.dumps(row), flush=True)
