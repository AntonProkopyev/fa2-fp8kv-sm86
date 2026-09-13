# Comparing this project with upstream FA2

`flash-attention/` is an unmodified Git submodule of
[`vllm-project/flash-attention`](https://github.com/vllm-project/flash-attention),
pinned to **`28e862d21806bc3580207aa0ad4e2759151e9827`**. This is the actual
source revision recorded in this project's `NOTICE`, not a newer upstream
snapshot chosen for the comparison. The vLLM repository is itself a fork of
Dao-AILab/flash-attention.

[`../patches/fa2-fp8kv.patch`](../patches/fa2-fp8kv.patch) is the CUDA source
diff against that revision. It modifies four upstream headers:

- `csrc/flash_attn/src/flash.h`
- `csrc/flash_attn/src/flash_fwd_kernel.h`
- `csrc/flash_attn/src/mask.h`
- `csrc/flash_attn/src/utils.h`

The other twelve vendored headers are byte-identical to upstream. The patch
also adds the project's two new CUDA entrypoints, `fp8_attn.cu` and
`prefill.cu`, alongside those headers for review. They have no upstream file
to compare with and appear as additions from `/dev/null`.

Mapping back to the standalone build:

| Patched upstream path | Project path |
|---|---|
| `csrc/flash_attn/src/<header>` | `headers/<header>` |
| `csrc/flash_attn/src/fp8_attn.cu` | `fp8_attn.cu` |
| `csrc/flash_attn/src/prefill.cu` | `prefill.cu` |

This is a source comparison, not an integration into upstream's Python API
or build system. The standalone PyTorch operator, Python prefill orchestration,
build script, tests and vLLM adapter remain in this repository. The CUDA diff
does not claim to cover those separate integration files. Original BSD
notices remain in the headers; new entrypoints retain their Apache-2.0 notices.

## Verify the patch

From this repository's root:

```bash
git submodule update --init upstream/flash-attention
python3 check_upstream.py
```

The check verifies the submodule commit, regenerates the expected diff from
Git blobs, and compares it with the committed patch. It then applies that
patch in a temporary tree and compares all **18 mapped files byte-for-byte**
with the standalone sources. It does not modify the submodule or use a GPU.

For manual inspection:

```bash
git -C upstream/flash-attention apply --stat ../../patches/fa2-fp8kv.patch
git -C upstream/flash-attention apply --check ../../patches/fa2-fp8kv.patch
```

After an intentional CUDA-source change, regenerate the review artifact
without replacing a good patch if generation fails:

```bash
python3 check_upstream.py --emit > patches/fa2-fp8kv.patch.tmp &&
  mv patches/fa2-fp8kv.patch.tmp patches/fa2-fp8kv.patch
python3 check_upstream.py
```

Builds continue to use the vendored modified headers; submodule initialization
is needed only for this comparison. GitHub source archives omit submodule
contents, so use a Git checkout for verification.
