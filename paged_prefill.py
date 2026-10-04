"""Single-request paged FA2 entrypoint for the legacy vLLM adapter."""
import torch


class PagedPrefill:
    def forward(self, query, keys, values, table, key_scale, value_scale,
                context_length, softmax_scale, output):
        total = query.shape[0]
        if context_length < total:
            raise ValueError("Invalid causal prefill length")
        starts = torch.tensor([0, total], device=query.device, dtype=torch.int32)
        lengths = torch.tensor([context_length], device=query.device, dtype=torch.int32)
        torch.ops.fa2_fp8kv.forward(query, keys, values, output, starts, lengths,
            table, key_scale, value_scale, total, context_length,
            True, -1, -1, softmax_scale, 1, False)
        return output
