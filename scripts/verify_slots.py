"""Verify slot labels by hand, blind first.

Practice:  .venv\\Scripts\\python.exe scripts\\verify_slots.py --practice
Round A:   .venv\\Scripts\\python.exe scripts\\verify_slots.py --round A
Summary:   .venv\\Scripts\\python.exe scripts\\verify_slots.py --round A --summary
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.access import ProtocolError, queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.qu.guidelines import (
    SLOT_EXAMPLES,
    SLOT_VERSION,
    RulesNotFinalError,
    require_final,
    slot_rules_text,
)
from shelfrank.qu.slot_vocab import load_vocabulary
from shelfrank.qu.verification import Slots, VerificationLog, room_check, run_session, summary

SEED = 20260
FOLDER = REPO_ROOT / "data" / "labels"
ROUNDS = {
    "A": ("dev", f"slots_wands_dev_{SLOT_VERSION}.jsonl"),
    "B": ("primary", f"slots_wands_round_b_{SLOT_VERSION}.jsonl"),
}


def load_teacher(path: Path) -> dict[int, Slots]:
    """Read teacher labels by query id."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(row["id"]): row["label"] for row in rows}


def practice() -> int:
    """Try the tool on invented queries. Nothing is saved."""
    queries = {i: q for i, (q, _) in enumerate(SLOT_EXAMPLES)}
    answers = {i: a for i, (_, a) in enumerate(SLOT_EXAMPLES)}
    print(slot_rules_text(), "\n\nPractice: invented queries, nothing is saved.")
    with tempfile.TemporaryDirectory() as folder:
        log = VerificationLog(Path(folder) / "practice.jsonl")
        run_session(queries, answers.__getitem__, log, SLOT_VERSION, SEED, input, print)
        print(summary(log, answers))
    return 0


def main() -> int:
    """Run practice, a round, or its summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", choices=sorted(ROUNDS))
    parser.add_argument("--practice", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.practice:
        return practice()
    if not args.round:
        parser.error("choose --practice or --round A")
    try:
        require_final(SLOT_VERSION)
        split, silver = ROUNDS[args.round]
        cfg = load_config("data")
        ids = queries_for(split, load_folds(REPO_ROOT / cfg["splits"]["path"]))
    except (RulesNotFinalError, ProtocolError) as error:
        print(f"not yet: {error}")
        return 1
    teacher_path = FOLDER / silver
    if not teacher_path.exists():
        print(f"not yet: {teacher_path.name} does not exist")
        return 1
    teacher = load_teacher(teacher_path)
    log = VerificationLog(FOLDER / f"slots_round_{args.round.lower()}.jsonl")
    texts = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id")["query"]
    queries = {q: str(texts[q]) for q in ids if q in teacher}
    if args.summary:
        raw = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
        out = {
            "rules": SLOT_VERSION,
            **summary(log, teacher),
            "room_rule": room_check(log, queries, load_vocabulary(raw)[2]),
        }
        path = REPO_ROOT / "results" / f"slots_round_{args.round.lower()}.json"
        path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(out, indent=1))
        return 0
    print(slot_rules_text())
    done = run_session(queries, teacher.__getitem__, log, SLOT_VERSION, SEED, input, print)
    print(f"{done} of {len(queries)} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
