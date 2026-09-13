"""Verify the CUDA source diff against the pinned upstream FA2 submodule."""
import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile

REVISION = "28e862d21806bc3580207aa0ad4e2759151e9827"
SOURCE = Path("csrc/flash_attn/src")


@dataclass(frozen=True)
class SourceComparison:
    root: Path

    def mappings(self):
        headers = tuple((SOURCE / path.name, path)
                        for path in sorted((self.root / "headers").iterdir())
                        if path.is_file())
        additions = tuple((SOURCE / name, self.root / name)
                          for name in ("fp8_attn.cu", "prefill.cu"))
        return headers + additions

    def original(self, path):
        # The two standalone extension entrypoints are new project code.
        if path.name in ("fp8_attn.cu", "prefill.cu"):
            return b""
        return subprocess.run(
            ["git", "-C", str(self.root / "upstream/flash-attention"),
             "show", f"{REVISION}:{path.as_posix()}"],
            check=True, capture_output=True,
        ).stdout

    def patch(self):
        changes = []
        for upstream, local in self.mappings():
            original, modified = self.original(upstream), local.read_bytes()
            if original == modified:
                continue
            name = upstream.as_posix()
            changes.append(f"diff --git a/{name} b/{name}\n")
            if not original:
                changes.append("new file mode 100644\n")
            changes.extend(difflib.unified_diff(
                original.decode("utf-8").splitlines(keepends=True),
                modified.decode("utf-8").splitlines(keepends=True),
                fromfile=f"a/{name}" if original else "/dev/null",
                tofile=f"b/{name}",
            ))
        return "".join(changes)

    def verified(self):
        upstream = self.root / "upstream/flash-attention"
        head = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True, capture_output=True, encoding="utf-8",
        ).stdout.strip()
        if head != REVISION:
            raise ValueError(f"Upstream must be {REVISION}; found {head}")
        patch = self.root / "patches/fa2-fp8kv.patch"
        if patch.read_text(encoding="utf-8") != self.patch():
            raise ValueError("CUDA sources differ from patches/fa2-fp8kv.patch; regenerate it")
        # Apply in a disposable tree. The submodule remains an untouched baseline.
        with tempfile.TemporaryDirectory(prefix="fa2-upstream-check-") as directory:
            tree = Path(directory)
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            for mapped, _ in self.mappings():
                original = self.original(mapped)
                if original:
                    target = tree / mapped
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(original)
            for flags in (["--check"], []):
                subprocess.run(["git", "apply", *flags, str(patch)], cwd=tree,
                               check=True, capture_output=True, encoding="utf-8")
            for mapped, local in self.mappings():
                if (tree / mapped).read_bytes() != local.read_bytes():
                    raise ValueError(f"Patched source differs: {mapped}")
        return f"PASS: patch applies to {REVISION}; all {len(self.mappings())} mapped files match byte-for-byte"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="Print the current CUDA source diff")
    args = parser.parse_args()
    comparison = SourceComparison(Path(__file__).resolve().parent)
    print(comparison.patch(), end="") if args.emit else print(comparison.verified())
