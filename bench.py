"""Numerics and matched CUDA-graph timings for Ornith's per-rank attention."""

import argparse
import functools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import flashinfer
import torch


@dataclass(frozen=True)
class CapturedAttention:
    operation: object

    def timing(self):
        for _ in range(3):
            self.operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.operation()
        for _ in range(3):
            graph.replay()
        samples = []
        for _ in range(5):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 10)
        graph.reset()
        return {"mean_us": statistics.mean(samples), "std_us": statistics.stdev(samples)}


parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
parser.add_argument("--baseline-library", default="")
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--nonunit-scales", action="store_true")
args = parser.parse_args()
if args.baseline_library:
    torch.ops.load_library(args.baseline_library)
torch.ops.load_library(args.library)
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.20)
torch.set_float32_matmul_precision("highest")
torch.manual_seed(53)
cases = (
    [(1, 17), (4, 4), (4, 7), (4, 31), (4, 2048), (4, 262143)]
    if args.check_only else
    [(1, 256), (1, 32768), (1, 262144), (4, 256), (4, 2048),
     (4, 32768), (4, 262144), (2048, 4096)]
)

with torch.inference_mode():
    for query_len, length in cases:
        dim, heads, kv_heads, page = 256, 8, 1, 16
        blocks = math.ceil(length / page)
        query = torch.randn((query_len, heads, dim), device="cuda", dtype=torch.bfloat16)
        shape = (blocks, page, kv_heads, dim)
        keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        values = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        cu_q = torch.tensor([0, query_len], device="cuda", dtype=torch.int32)
        seq_k = torch.tensor([length], device="cuda", dtype=torch.int32)
        indices = torch.arange(blocks, device="cuda", dtype=torch.int32).flip(0)
        table = indices.reshape(1, -1)
        indptr = torch.tensor([0, blocks], device="cuda", dtype=torch.int32)
        last_page = torch.tensor([length % page or page], device="cuda", dtype=torch.int32)
        k_factor = 0.5 if args.check_only or args.nonunit_scales else 1.0
        v_factor = 1.75 if args.check_only or args.nonunit_scales else 1.0
        k_scale = torch.tensor([k_factor], device="cuda")
        v_scale = torch.tensor([v_factor], device="cuda")
        workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD", backend="fa2")
        wrapper.plan(cu_q, indptr, indices, last_page, heads, kv_heads, dim, page,
                     causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn)
        # Decode the exact quantized values; FP32 matmul is an independent oracle.
        dense_k = keys.view(torch.uint8)[indices.long()].view(torch.float8_e4m3fn).float().reshape(-1, dim)[:length] * k_factor
        dense_v = values.view(torch.uint8)[indices.long()].view(torch.float8_e4m3fn).float().reshape(-1, dim)[:length] * v_factor
        scores = torch.einsum("qhd,kd->hqk", query.float(), dense_k) / math.sqrt(dim)
        future = torch.arange(length, device="cuda")[None, :] > (
            length - query_len + torch.arange(query_len, device="cuda")[:, None]
        )
        scores.masked_fill_(future[None], -torch.inf)
        reference_lse = scores.logsumexp(-1)
        reference = torch.einsum("hqk,kd->qhd", scores.softmax(-1), dense_v).bfloat16()
        implementations = []
        if args.baseline_library:
            implementations.append(("baseline", torch.ops.qwen38_fa2_fp8.forward, False))
        if args.library:
            implementations.append(("candidate_unpacked", torch.ops.ornith_fa2_fp8.forward, False))
            if query_len * heads <= 64:
                implementations.append(("candidate_gqa", torch.ops.ornith_fa2_fp8.forward, True))
        row = {"query_len": query_len, "kv_len": length, "head_dim": dim,
               "query_heads": heads, "kv_heads": kv_heads, "dtype": "fp8_e4m3",
               "kv_scales": [k_factor, v_factor]}
        for label, operator, grouped in implementations:
            guarded = torch.full((query_len + 2, heads, dim), 11, device="cuda", dtype=torch.bfloat16)
            output = guarded[1:-1]
            params = [query, keys, values, output, cu_q, seq_k, table,
                      k_scale, v_scale, query_len, length, True, -1, -1, 1 / math.sqrt(dim), 0]
            if label != "baseline":
                params.append(grouped)
            operation = functools.partial(operator, *params)
            _, lse = operation()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, atol=0.025, rtol=0.025)
            torch.testing.assert_close(lse, reference_lse, atol=0.025, rtol=0.005)
            assert bool((guarded[[0, -1]] == 11).all()), "output guard overwritten"
            result = {"max_abs_error": (output.float() - reference.float()).abs().max().item()}
            if not args.check_only:
                result.update(CapturedAttention(operation).timing())
            row[label] = result
        output = torch.empty_like(query)
        operation = functools.partial(wrapper.run, query, (keys, values), out=output,
                                      k_scale=k_factor, v_scale=v_factor)
        operation()
        torch.testing.assert_close(output, reference, atol=0.025, rtol=0.025)
        row["flashinfer"] = {"max_abs_error": (output.float() - reference.float()).abs().max().item()}
        if not args.check_only:
            row["flashinfer"].update(CapturedAttention(operation).timing())
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        print(json.dumps(row), flush=True)
