# Prebuilt kernel artifacts

The artifact image contains two CUDA libraries, the vLLM plugin wheel,
checksums, an ABI manifest and license notices. It contains no model weights.
Its entrypoint extracts the files into `/export/<artifact_id>` and exits.

Image published on September 13, 2026:

```text
ghcr.io/antonprokopyev/fa2-fp8kv-sm86@sha256:da040941fa048fd5fdfce520503341c0beda4ce41436a1c1fecaf3f8a99777c7
```

Artifact ID:

```text
731d1942112a7cf35be4f0add0919072c06cac47e3bad4f674ff0157edad0e3f
```

Source revision: `0fa02cbb760fbcc4a94ba1ee376e89825f8f43f4`.
Build ABI: Linux x86-64, CPython 3.12, PyTorch 2.13.0+cu130, CUDA 13.0,
C++11 ABI enabled. Both extensions contain SM86, SM89 and SM120 code.
Only SM86 has physical-GPU validation. Compilation for SM89/SM120 does not
establish runtime correctness or performance on those cards.

Extract the artifact after pulling the image:

```bash
docker run --rm --network none -v "$PWD/artifacts:/export" \
  ghcr.io/antonprokopyev/fa2-fp8kv-sm86@sha256:da040941fa048fd5fdfce520503341c0beda4ce41436a1c1fecaf3f8a99777c7
```

Repeated extraction verifies the existing files against checksums from the
image. It refuses a corrupted cache. The serving integration must verify
`manifest.json` against its pinned artifact ID and runtime ABI before loading
the wheel and libraries. The ultramax integration in
[club-3090 PR #1274](https://github.com/noonghunna/club-3090/pull/1274)
provides that consumer and a read-only runtime mount. Registry visibility must
be public before an unauthenticated user can pull the image.

## Build from the pinned source

Use a clean checkout of the source revision above. The Dockerfile pins the
stock vLLM builder and BusyBox export image by digest. The build script pins
CUTLASS by revision and archive SHA256. No GPU is required to compile.

```bash
git checkout 0fa02cbb760fbcc4a94ba1ee376e89825f8f43f4
docker build -f containers/Dockerfile \
  --build-arg SOURCE_REVISION=0fa02cbb760fbcc4a94ba1ee376e89825f8f43f4 \
  -t fa2-kernels:local .
```

The build performs CPU library-load checks. GPU checks are separate:
`check_backend.py` exercises native KV writes, reference attention and CUDA
Graph replay. The published artifact passed those checks on two RTX 3090
cards after offline extraction and installation. First export, repeated
export and deliberate cache corruption were also tested.

The source comparison against upstream FA2 is independent of image delivery:
see [`../upstream/README.md`](../upstream/README.md) and `check_upstream.py`.
