"""Slot student and lexicon against silver labels after the freeze. Writes results/slots_all.json.

Silver labels come from the LLM teacher. Human verification (Round B) is pending.
"""

import json

from shelfrank.config import REPO_ROOT
from shelfrank.ltr.pipeline import load_split
from shelfrank.qu.slot_lexicon import span_scores
from shelfrank.qu.slot_model import labelled_slots, load_lexicon, load_slot_model
from shelfrank.qu.slot_scores import f1_interval


def main() -> None:
    """Score both models on dev, primary and all queries."""
    silver = labelled_slots("wands_dev") | labelled_slots("wands_primary")
    models = {"student": load_slot_model(), "lexicon": load_lexicon()}
    out: dict[str, object] = {"reference": "LLM silver labels; human verification pending"}
    for split in ("dev", "primary", "all"):
        ids, texts, _ = load_split(split)
        truth = [silver[q] for q in ids]
        rows: dict[str, object] = {
            "n": len(ids),
            "queries_with_a_slot": sum(any(t.values()) for t in truth),
        }
        for name, model in models.items():
            predicted = [model.slots(t) for t in texts]
            scores = span_scores(truth, predicted)
            scores["all"] |= f1_interval(truth, predicted)
            rows[name] = scores
        out[split] = rows
        print(split, {n: round(rows[n]["all"]["f1"], 3) for n in models})  # type: ignore[index]
    path = REPO_ROOT / "results/slots_all.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
