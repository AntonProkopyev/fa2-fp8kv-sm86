#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"
cd "$(dirname "${BASH_SOURCE[0]}")"
cutlass_revision=62750a2b75c802660e4894434dc55e839f322277
command -v git >/dev/null || { echo 'Install git before fetching CUTLASS.' >&2; exit 2; }
if [[ ! -d third_party/cutlass/.git ]]; then
  mkdir -p third_party/cutlass
  git -C third_party/cutlass init
  git -C third_party/cutlass remote add origin https://github.com/NVIDIA/cutlass.git
fi
if [[ -n "$(git -C third_party/cutlass status --porcelain)" ]]; then
  echo 'CUTLASS has local changes; preserve them before running setup.' >&2
  exit 2
fi
git -C third_party/cutlass fetch --depth=1 origin "$cutlass_revision"
git -C third_party/cutlass checkout --detach "$cutlass_revision"
if [[ "${1:-}" == --fetch-only ]]; then exit 0; fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"
python3 build.py --pipeline "$@"
