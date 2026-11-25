"""Label slots by hand. Practice comes first and unlocks the rounds.

Practice:  .venv\\Scripts\\python.exe scripts\\verify_slots.py --practice
Round A:   .venv\\Scripts\\python.exe scripts\\verify_slots.py --round A
Summary:   .venv\\Scripts\\python.exe scripts\\verify_slots.py --round A --summary
"""

import argparse
import json
import random
import sys
from collections.abc import Collection
from dataclasses import asdict
from pathlib import Path
from typing import Any

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.access import ProtocolError, queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.qu.annotator import PASS_MARK, PRACTICE, read_attempts, unlocked
from shelfrank.qu.guidelines import (
    SLOT_VERSION,
    RulesNotFinalError,
    require_final,
    slot_rules_text,
)
from shelfrank.qu.slot_vocab import Term, load_vocabulary
from shelfrank.qu.verification import (
    Session,
    Slots,
    VerificationLog,
    room_check,
    run_practice,
    run_session,
    summary,
)

SEED = 20260
FOLDER = REPO_ROOT / "data" / "labels"
PRACTICE_LOG = FOLDER / "slots_practice.jsonl"
ROUNDS = {
    "A": ("dev", f"slots_wands_dev_{SLOT_VERSION}.jsonl"),
    "B": ("primary", f"slots_wands_round_b_{SLOT_VERSION}.jsonl"),
}
# Labelled like the rest, but not counted in agreement: in the discarded attempt its label was
# changed to the teacher's answer.
EXCLUDED = {"A": frozenset({371}), "B": frozenset()}


def load_teacher(path: Path) -> dict[int, Slots]:
    """Read teacher labels by query id."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(row["id"]): row["label"] for row in rows}


def vocabulary() -> tuple[list[str], Collection[Term], set[Term]]:
    """Style names to show, style terms to check against, and product-type phrases."""
    raw = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
    _, styles, phrases = load_vocabulary(raw)
    return list(raw["styles"]), styles.keys(), phrases


def practice() -> int:
    """Run the practice queries with feedback and record the score."""
    names, terms, _ = vocabulary()
    attempts = read_attempts(PRACTICE_LOG)
    # A new order per attempt, so a retry is not answered from memory of the sequence.
    items = random.Random(SEED + len(attempts)).sample(PRACTICE, len(PRACTICE))
    print(slot_rules_text(), "\n")
    attempt = run_practice(items, SLOT_VERSION, input, print, names, terms)
    if attempt is None:
        return 1
    with PRACTICE_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(attempt)) + "\n")
    passed = attempt.unlocks(SLOT_VERSION)
    print(f"\nScore {attempt.correct}/{attempt.total}. Needed {PASS_MARK}.")
    print("Round A is unlocked." if passed else "Not yet. Read the answers above and try again.")
    return 0


def write_summary(name: str, log: VerificationLog, queries: dict[int, str], path: Path) -> None:
    """Save and print agreement. The teacher's labels are read only here."""
    out: dict[str, Any] = {
        "rules": SLOT_VERSION,
        "practice": [f"{a.correct}/{a.total}" for a in read_attempts(PRACTICE_LOG)],
        **summary(log, load_teacher(path), EXCLUDED[name]),
        "room_rule": room_check(log, queries, vocabulary()[2], EXCLUDED[name]),
    }
    target = REPO_ROOT / "results" / f"slots_round_{name.lower()}.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


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
    if not unlocked(PRACTICE_LOG, SLOT_VERSION):
        print(f"locked: score at least {PASS_MARK}/{len(PRACTICE)} in --practice first")
        return 1
    split, silver = ROUNDS[args.round]
    try:
        require_final(SLOT_VERSION)
        cfg = load_config("data")
        ids = queries_for(split, load_folds(REPO_ROOT / cfg["splits"]["path"]))
    except (RulesNotFinalError, ProtocolError) as error:
        print(f"not yet: {error}")
        return 1
    if args.round == "B":
        # The committed draw of 100 ids does not exist before the freeze.
        print("not yet: round B has no committed query draw")
        return 1
    log = VerificationLog(FOLDER / f"slots_round_{args.round.lower()}.jsonl")
    texts = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id")["query"]
    queries = {q: str(texts[q]) for q in ids}
    if args.summary:
        done = len(log.finished())
        if done < len(queries):
            # Agreement seen halfway would influence the remaining labels.
            print(f"not yet: {done} of {len(queries)} finished")
            return 1
        write_summary(args.round, log, queries, FOLDER / silver)
        return 0
    print(slot_rules_text())
    names, terms, _ = vocabulary()
    # No teacher label file is opened on this path, in either round.
    session = Session(log, SLOT_VERSION, input, print, names, terms)
    done = run_session(session, queries, SEED)
    print(f"{done} of {len(queries)} finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
