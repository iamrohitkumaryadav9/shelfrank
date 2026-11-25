"""Check that each compiled dependency loads on this machine.

Each import runs in its own process because Windows Smart App Control can reject a single DLL, and
one rejection would otherwise hide the state of everything after it.
"""

import shutil
import subprocess
import sys

CORE = ["numpy", "pandas", "yaml"]
MIN_FREE_GB = 8
BENCH = [
    "scipy.stats",
    "sklearn.linear_model",
    "lightgbm",
    "bm25s",
    "faiss",
    "torch",
    "sentence_transformers",
    "ranx",
    "google.genai",
    "pydantic",
]
APP = ["duckdb", "pyarrow.parquet", "onnxruntime", "tokenizers", "streamlit"]


def check(module: str) -> str | None:
    """Return the final error line, or None if the import works."""
    run = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, timeout=300
    )
    if run.returncode == 0:
        return None
    return (run.stderr.strip().splitlines() or ["no output"])[-1]


def low_disk_warning(free_bytes: int, minimum_gb: float = MIN_FREE_GB) -> str | None:
    """Warn early: an embedding or model write that runs out of disk leaves a corrupt cache."""
    free_gb = free_bytes / 1e9
    if free_gb >= minimum_gb:
        return None
    return f"WARNING: {free_gb:.1f} GB free, below the {minimum_gb} GB minimum"


def main() -> int:
    """Probe the groups named on the command line, or all of them."""
    groups = {"core": CORE, "bench": BENCH, "app": APP}
    wanted = sys.argv[1:] or list(groups)
    failures = 0
    for group in wanted:
        for module in groups[group]:
            error = check(module)
            failures += error is not None
            print(f"{'FAIL' if error else 'ok  '} {group:5s} {module} {error or ''}".rstrip())
    free = shutil.disk_usage(__file__).free
    print(low_disk_warning(free) or f"ok   disk  {free / 1e9:.1f} GB free")
    return min(failures, 1)


if __name__ == "__main__":
    sys.exit(main())
