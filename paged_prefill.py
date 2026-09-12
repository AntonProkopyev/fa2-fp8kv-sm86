"""Bounded-query fallback using the original paged FP8 FA2 operation."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PagedPrefill:
    query_tokens: int = 2048

    def forward(self, query, keys, values, table, key_scale, value_scale,
                context_length, softmax_scale, output):
        total = query.shape[0]
        if context_length < total:
            raise ValueError("Invalid causal prefill length")
        for start in range(0, total, self.query_tokens):
            count = min(self.query_tokens, total - start)
            length = context_length - (total - start - count)
            starts = torch.tensor([0, count], device=query.device, dtype=torch.int32)
            lengths = torch.tensor([length], device=query.device, dtype=torch.int32)
            torch.ops.fa2_fp8kv.forward(query[start:start + count], keys, values,
                output[start:start + count], starts, lengths, table, key_scale,
                value_scale, count, length, True, -1, -1, softmax_scale, 1, False)
        return output
