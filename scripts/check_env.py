"""Check that each compiled dependency loads on this machine.

Each import runs in its own process because Windows Smart App Control can reject a single DLL, and
one rejection would otherwise hide the state of everything after it.
"""

import subprocess
import sys

CORE = ["numpy", "pandas", "yaml"]
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
    return min(failures, 1)


if __name__ == "__main__":
    sys.exit(main())
