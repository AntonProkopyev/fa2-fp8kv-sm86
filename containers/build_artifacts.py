"""Build the CUDA libraries and Python wheel in the pinned build image."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig
import tarfile
import urllib.request

import torch

CUTLASS = "62750a2b75c802660e4894434dc55e839f322277"
CUTLASS_SHA = "78816d6c6d97793b5b59ef2a702174cb85b78dfcefc8fe2489964de2e42f17d2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
revision = os.environ.get("SOURCE_REVISION", "")
if not re.fullmatch("[0-9a-f]{40}", revision):
    raise SystemExit("SOURCE_REVISION must be the full source commit SHA")
if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.6;8.9;12.0":
    raise SystemExit("This recipe builds SM86, SM89 and SM120; runtime validation is separate")
source = Path.cwd()
args.out.mkdir(parents=True, exist_ok=True)
# A fixed build path keeps dependency paths in CUDA line information stable.
dependency = Path("/tmp/fa2-build-deps")
dependency.mkdir(exist_ok=False)
with urllib.request.urlopen(f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS}", timeout=120) as response:
    archive = response.read()
if hashlib.sha256(archive).hexdigest() != CUTLASS_SHA:
    raise SystemExit("CUTLASS archive checksum mismatch")
with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
    contents.extractall(dependency, filter="data")
cutlass = dependency / f"cutlass-{CUTLASS}"
subprocess.run(["python3", "build.py", "--pipeline", "--prefill",
                "--cutlass-include", str(cutlass / "include")], check=True)
subprocess.run(["python3", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                "--no-index", ".", "--wheel-dir", str(args.out)], check=True)
for filename, directory in [("fa2_fp8kv.so", "build-pipeline"),
                            ("fa2_fp8kv_prefill.so", "build-prefill")]:
    shutil.copy2(source / directory / filename, args.out / filename)
    torch.ops.load_library(str(args.out / filename))
shutil.copytree(source / "licenses", args.out / "licenses")
shutil.copy2(cutlass / "LICENSE.txt", args.out / "licenses/CUTLASS-LICENSE")
for name in ("LICENSE", "NOTICE"):
    shutil.copy2(source / name, args.out / name)
files = {p.relative_to(args.out).as_posix(): digest(p)
         for p in sorted(args.out.rglob("*")) if p.is_file()}
manifest = {
    "schema": 1, "source_revision": revision,
    "abi": {"torch": torch.__version__, "cuda": torch.version.cuda,
            "python_soabi": sysconfig.get_config_var("SOABI"),
            "machine": platform.machine(), "system": platform.system(),
            "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI},
    "compiled_sm": ["8.6", "8.9", "12.0"],
    "build_provenance": {"vllm": importlib.metadata.version("vllm"),
                         "flashinfer_headers": importlib.metadata.version("flashinfer-python"),
                         "cutlass": CUTLASS,
                         "fa2_base": "28e862d21806bc3580207aa0ad4e2759151e9827"},
    "files": files,
}
identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
manifest["artifact_id"] = identity
(args.out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
(args.out / "ARTIFACT_ID").write_text(identity + "\n", encoding="utf-8")
sums = "".join(f"{digest(p)}  {p.relative_to(args.out).as_posix()}\n"
               for p in sorted(args.out.rglob("*")) if p.is_file())
(args.out / "SHA256SUMS").write_text(sums, encoding="utf-8")
print(json.dumps({"artifact_id": identity, "files": files}), flush=True)
