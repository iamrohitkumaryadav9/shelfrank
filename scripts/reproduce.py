"""Rebuild the results from committed files with zero API calls, and check that nothing changes.

    python scripts/reproduce.py --offline             every step that the local files allow
    python scripts/reproduce.py --offline --fixture   what CI runs: committed files and the fixture only

Network access and the API key are switched off for every step, and the spend ledger must not change.
Steps that need the WANDS download or large artifacts are listed as skipped, never silently left out.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results/llm_spend.jsonl"
RAW = ROOT / "data/raw/wands/product.csv"
EMBEDDINGS = ROOT / "artifacts/embeddings/bge_small/embeddings.npy"
TUNED = ROOT / "artifacts/embeddings/bge_small_esci/embeddings.npy"
APP_DATA = ROOT / "app_data/meta.json"

# Runs before the step: no sockets except to this machine, and no key.
GUARD = """
import os, runpy, socket, sys
os.environ.pop("GOOGLE_API_KEY", None)
os.environ["HF_HUB_OFFLINE"] = "1"
real = socket.socket.connect
def connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise OSError("network access is off in reproduce.py")
    return real(self, address)
socket.socket.connect = connect
sys.path.insert(0, os.getcwd())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@dataclass(frozen=True)
class Step:
    """One script, the files it may rewrite, and what it needs."""

    name: str
    command: tuple[str, ...]
    outputs: tuple[str, ...]
    needs: tuple[Path, ...] = ()
    in_fixture: bool = False
    extra: dict[str, str] = field(default_factory=dict)


LABELS = "data/labels"
STEPS = (
    Step(
        "teacher classes, ESCI slice",
        ("scripts/label.py", "class", "esci_dev"),
        (f"{LABELS}/class_esci_dev_class-v1.jsonl",),
        in_fixture=True,
    ),
    Step(
        "teacher slots, ESCI slice",
        ("scripts/label.py", "slots", "esci_dev"),
        (f"{LABELS}/slots_esci_dev_slots-v1.jsonl",),
        in_fixture=True,
    ),
    Step(
        "teacher classes, ESCI seed",
        ("scripts/label.py", "class", "esci_seed"),
        (f"{LABELS}/class_esci_seed_class-v1.jsonl",),
        in_fixture=True,
    ),
    Step(
        "teacher slots, ESCI seed",
        ("scripts/label.py", "slots", "esci_seed"),
        (f"{LABELS}/slots_esci_seed_slots-v1.jsonl",),
        in_fixture=True,
    ),
    Step(
        "teacher classes, WANDS dev",
        ("scripts/label.py", "class", "wands_dev"),
        (f"{LABELS}/class_wands_dev_class-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "silver slots, WANDS dev",
        ("scripts/label.py", "slots", "wands_dev"),
        (f"{LABELS}/slots_wands_dev_slots-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "teacher classes, WANDS primary",
        ("scripts/label.py", "class", "wands_primary"),
        (f"{LABELS}/class_wands_primary_class-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "silver slots, WANDS primary",
        ("scripts/label.py", "slots", "wands_primary"),
        (f"{LABELS}/slots_wands_primary_slots-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "query rewrites, WANDS dev",
        ("scripts/rewrite_queries.py", "--split", "dev"),
        (f"{LABELS}/rewrites_wands_dev_rewrite-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "query rewrites, WANDS primary",
        ("scripts/rewrite_queries.py", "--split", "primary"),
        (f"{LABELS}/rewrites_wands_primary_rewrite-v1.jsonl",),
        (RAW,),
    ),
    Step(
        "data audit",
        ("scripts/data_audit.py",),
        ("results/data_audit.json", "docs/data_audit.md"),
        (RAW,),
    ),
    Step("slot lexicon", ("scripts/build_slot_lexicon.py",), ("data/slot_lexicon.json",), (RAW,)),
    Step(
        "class ceiling, dev",
        ("scripts/class_ceiling.py", "--split", "dev"),
        ("results/class_ceiling_dev.json",),
        (RAW,),
    ),
    Step(
        "class ceiling, primary",
        ("scripts/class_ceiling.py", "--split", "primary"),
        ("results/class_ceiling_primary.json",),
        (RAW,),
    ),
    Step("slots on all queries", ("scripts/slots_all.py",), ("results/slots_all.json",), (RAW,)),
    Step(
        "retrieval baselines, dev",
        ("scripts/run_retrieval.py", "--split", "dev"),
        (),
        (RAW, EMBEDDINGS),
    ),
    Step(
        "more like this, dev",
        ("scripts/similar_products.py",),
        ("results/similar_dev.json",),
        (RAW, EMBEDDINGS, TUNED),
    ),
    Step(
        "more like this, primary",
        ("scripts/similar_products.py", "--split", "primary"),
        ("results/similar_primary.json",),
        (RAW, EMBEDDINGS, TUNED),
    ),
    Step("tables", ("scripts/make_tables.py",), ("README.md", "docs/*.md"), in_fixture=True),
)


def digest(path: Path) -> str:
    """Hash of a file, or "missing"."""
    if not path.exists():
        return "missing"
    # Line endings differ between checkouts and are not content.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def files_of(step: Step) -> list[Path]:
    """Output files of a step that exist."""
    return sorted(p for pattern in step.outputs for p in ROOT.glob(pattern))


def run_step(step: Step) -> str:
    """Run one step without network and say whether its outputs are unchanged."""
    files = files_of(step)
    before = {p: (digest(p), p.read_bytes()) for p in files}
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"} | {"PYTHONUTF8": "1"}
    done = subprocess.run(
        [sys.executable, "-c", GUARD, *step.command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    changed = [p.relative_to(ROOT).as_posix() for p in files if digest(p) != before[p][0]]
    if done.returncode:
        for path, (_, body) in before.items():
            path.write_bytes(body)
        return "FAILED: " + (done.stderr.strip().splitlines() or ["no output"])[-1]
    if changed:
        return "CHANGED: " + ", ".join(changed)
    return f"identical ({len(files)} files)" if files else "ran"


def fixture_search() -> str:
    """Build a complete small app_data from the fixture and search it."""
    with tempfile.TemporaryDirectory() as folder:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_app_support.py",
                "tests/test_metrics.py",
                "-p",
                "no:cacheprovider",
                f"--basetemp={folder}/t",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    return "passed" if done.returncode == 0 else "FAILED: " + done.stdout[-300:]


def main() -> int:
    """Run the steps and print one line for each."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    ledger = digest(LEDGER)
    problems = skipped = 0
    for step in STEPS:
        missing = [p.relative_to(ROOT).as_posix() for p in step.needs if not p.exists()]
        labels = [p for p in step.outputs if p.startswith(LABELS)]
        if args.fixture and not step.in_fixture:
            outcome = "skipped: not part of the fixture run"
        elif missing:
            outcome = "skipped: needs " + ", ".join(missing)
        elif labels and not files_of(step):
            outcome = "skipped: nothing committed to compare with"
        else:
            outcome = run_step(step)
        skipped += outcome.startswith("skipped")
        problems += outcome.startswith(("FAILED", "CHANGED"))
        print(f"{step.name:34s} {outcome}", flush=True)
    if args.fixture:
        outcome = fixture_search()
        problems += outcome.startswith("FAILED")
        print(f"{'search on the fixture catalogue':34s} {outcome}")
    calls = "zero API calls" if digest(LEDGER) == ledger else "THE LEDGER CHANGED"
    problems += calls != "zero API calls"
    print(f"{len(STEPS) - skipped} steps run, {skipped} skipped, {problems} problems, {calls}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
