"""Summarise teacher labels on the ESCI prompt-development slice into results/teacher_slice.json."""

import json
from collections import Counter
from typing import Any

from shelfrank.config import REPO_ROOT
from shelfrank.data.taxonomy import OTHER
from shelfrank.qu.guidelines import CLASS_VERSION, SLOT_VERSION, SLOTS
from shelfrank.qu.slot_vocab import load_vocabulary, room_slots, room_spans
from shelfrank.text.normalize import tokenize


def load(name: str) -> list[dict[str, Any]]:
    """Read a label file."""
    path = REPO_ROOT / "data/labels" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def slot_density(rows: list[dict[str, Any]], target: int) -> dict[str, Any]:
    """Queries with each slot, and the positives that rate would give in a larger sample."""
    n = len(rows)
    with_slot = {s: sum(bool(r["label"][s]) for r in rows) for s in SLOTS}
    return {
        "queries": n,
        "any": sum(any(r["label"][s] for s in SLOTS) for r in rows),
        "per_slot": with_slot,
        "projected": {s: round(c / n * target) for s, c in with_slot.items()},
        "invented": sum(len(r["invented"]) for r in rows),
        "out_of_vocabulary": sum(len(r["out_of_vocabulary"]) for r in rows),
        "spans": sum(len(r["label"][s]) for r in rows for s in SLOTS),
    }


def room_agreement(rows: list[dict[str, Any]], queries: list[str], phrases: set) -> dict[str, int]:
    """Teacher room labels against the room rule, on queries with a room word."""
    cases = disagree = 0
    for row in rows:
        query = queries[row["id"]]
        if room_spans(query):
            cases += 1
            mine = sorted(" ".join(tokenize(s)) for s in row["label"]["room"])
            disagree += mine != sorted(room_slots(query, phrases))
    return {"cases": cases, "disagreements": disagree}


def main() -> None:
    """Summarise and print, including the labels themselves: these are ESCI queries."""
    path = REPO_ROOT / "data/esci/esci_prompt_dev.txt"
    queries = path.read_text(encoding="utf-8").splitlines()
    raw = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
    phrases = load_vocabulary(raw)[2]
    classes = load(f"class_esci_dev_{CLASS_VERSION}")
    slots = load(f"slots_esci_dev_{SLOT_VERSION}")
    labels = [row["label"] for row in classes]
    out = {
        "rules": {"class": CLASS_VERSION, "slots": SLOT_VERSION},
        "class": {
            "queries": len(labels),
            "outside_taxonomy": labels.count(None),
            "other": labels.count(OTHER),
            "distinct": len(set(labels) - {None, OTHER}),
            "top": Counter(x for x in labels if x and x != OTHER).most_common(5),
        },
        "slots": slot_density(slots, 2000),
        "room_rule": room_agreement(slots, queries, phrases),
    }
    target = REPO_ROOT / "results/teacher_slice.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    for row in slots:
        shown = {s: row["label"][s] for s in SLOTS if row["label"][s]}
        extra = {k: row[k] for k in ("invented", "out_of_vocabulary") if row[k]}
        print(f"{queries[row['id']]!r:58s} {labels[row['id']]!s:32s} {shown or ''} {extra or ''}")


if __name__ == "__main__":
    main()
