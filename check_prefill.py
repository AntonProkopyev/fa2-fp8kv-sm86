"""Compare bounded FA2 prefill and FP8 unpacking with an FP32 oracle."""
import argparse
import json
import math
import torch
from fa2_prefill import Fa2Prefill
from paged_prefill import PagedPrefill

parser = argparse.ArgumentParser()
parser.add_argument("library")
parser.add_argument("--paged-library", default="")
args = parser.parse_args()
torch.ops.load_library(args.library)
if args.paged_library:
    torch.ops.load_library(args.paged_library)
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.35)
torch.set_float32_matmul_precision("highest")
torch.manual_seed(211)

cases = [(8, 19, 128, 8, 2, 16), (65, 131, 256, 12, 2, 64),
         (128, 128, 256, 8, 1, 64), (2048, 4096, 256, 12, 2, 1024),
         (4096, 5000, 256, 12, 2, 2048), (8, 262143, 256, 12, 2, 65536)]
with torch.inference_mode():
    codes = torch.tensor([i for i in range(256) if i not in (127, 255)], device="cuda", dtype=torch.uint8)
    raw = codes.repeat(math.ceil(4096 / codes.numel()))[:4096].reshape(1, 16, 1, 256)
    finite_k = raw.view(torch.float8_e4m3fn)
    finite_v = raw.flip(-1).view(torch.float8_e4m3fn)
    one_page = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    finite_scale = torch.tensor([0.713], device="cuda")
    unpacked = torch.ops.fa2_fp8kv_prefill.gather(finite_k, finite_v, one_page, finite_scale, finite_scale, 0, 16, 0)
    exact = torch.stack(((finite_k.float() * finite_scale).bfloat16()[0],
                         (finite_v.float() * finite_scale).bfloat16()[0]))
    torch.testing.assert_close(unpacked.view(torch.int16), exact.view(torch.int16), atol=0, rtol=0)
    print('all finite E4M3 bytes: bitwise BF16 conversion PASS', flush=True)
    for qlen, length, dim, heads, kv_heads, chunk in cases:
        page = 16
        blocks = math.ceil(length / page)
        shape = (blocks + 1, 2, page, kv_heads, dim)
        keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)[:, 0]
        values = torch.randn(shape, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)[:, 1]
        keys[0] = float("nan")
        values[0] = float("nan")
        table = torch.arange(blocks, 0, -1, device="cuda", dtype=torch.int32)[None]
        kscale = torch.tensor([0.713], device="cuda")
        vscale = torch.tensor([1.317], device="cuda")
        ids = table[0].long()
        dense_k = (keys.view(torch.uint8)[ids].view(torch.float8_e4m3fn).float() * kscale).bfloat16().reshape(-1, kv_heads, dim)[:length]
        dense_v = (values.view(torch.uint8)[ids].view(torch.float8_e4m3fn).float() * vscale).bfloat16().reshape(-1, kv_heads, dim)[:length]
        for head in range(kv_heads):
            start, count = 7, min(37, length - 7)
            gathered = torch.ops.fa2_fp8kv_prefill.gather(keys, values, table, kscale, vscale, start, count, head)
            expected = torch.stack((dense_k[start:start + count, head:head + 1], dense_v[start:start + count, head:head + 1]))
            torch.testing.assert_close(gathered, expected, atol=0, rtol=0)
        query = torch.randn((qlen, heads, dim), device="cuda", dtype=torch.bfloat16)
        guarded = torch.full((qlen + 2, 2, heads, dim), 11, device="cuda", dtype=torch.bfloat16)
        output = guarded[1:-1, 0]
        Fa2Prefill(chunk).forward(query, keys, values, table, kscale, vscale, length, dim**-0.5, output)
        q = query.float().reshape(qlen, kv_heads, heads // kv_heads, dim)
        scores = torch.einsum("qhgd,khd->hgqk", q, dense_k.float()).reshape(heads, qlen, length) * dim**-0.5
        future = torch.arange(length, device="cuda")[None] > length - qlen + torch.arange(qlen, device="cuda")[:, None]
        scores.masked_fill_(future[None], -torch.inf)
        probabilities = scores.softmax(-1).reshape(kv_heads, heads // kv_heads, qlen, length)
        expected = torch.einsum("hgqk,khd->qhgd", probabilities, dense_v.float()).reshape(qlen, heads, dim).bfloat16()
        torch.testing.assert_close(output, expected, atol=0.025, rtol=0.025)
        assert bool((guarded[[0, -1]] == 11).all()) and bool((guarded[:, 1] == 11).all())
        if args.paged_library:
            fallback_guard = torch.full_like(guarded, 11)
            fallback = fallback_guard[1:-1, 0]
            PagedPrefill().forward(query, keys, values, table, kscale, vscale, length, dim**-0.5, fallback)
            torch.testing.assert_close(fallback, expected, atol=0.025, rtol=0.025)
            assert bool((fallback_guard[[0, -1]] == 11).all()) and bool((fallback_guard[:, 1] == 11).all())
            del fallback, fallback_guard
        print(json.dumps({"query": qlen, "kv": length, "dim": dim, "heads": heads,
                          "kv_heads": kv_heads, "chunk": chunk, "result": "PASS",
                          "max_error": float((output.float() - expected.float()).abs().max()),
                          "peak_mib": torch.cuda.max_memory_allocated() / 2**20}), flush=True)
        del keys, values, dense_k, dense_v, scores, probabilities, query, q, output, guarded, expected
