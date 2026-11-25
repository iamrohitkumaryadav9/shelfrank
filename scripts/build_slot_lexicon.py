"""Write data/slot_lexicon.json from the catalogue, and score the matcher on the ESCI slice.

The score is agreement with the teacher on 100 ESCI queries. WANDS dev is scored only after Round A.
"""

import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_products
from shelfrank.qu.guidelines import SLOT_VERSION
from shelfrank.qu.slot_lexicon import LexiconMatcher, build_lexicon, span_scores


def main() -> None:
    """Build, save and score."""
    cfg = load_config("data")
    lexicon = build_lexicon(load_products(REPO_ROOT / cfg["wands"]["raw_dir"]))
    path = REPO_ROOT / "data/slot_lexicon.json"
    text = json.dumps(lexicon, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print({k: len(v) for k, v in lexicon.items()})

    queries = (REPO_ROOT / "data/esci/esci_prompt_dev.txt").read_text(encoding="utf-8").splitlines()
    labels = REPO_ROOT / f"data/labels/slots_esci_dev_{SLOT_VERSION}.jsonl"
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines()]
    matcher = LexiconMatcher(lexicon)
    predicted = [matcher.slots(queries[row["id"]]) for row in rows]
    out = {
        "reference": "teacher labels on the 100 ESCI prompt-development queries",
        "lexicon_terms": {k: len(v) for k, v in lexicon.items()},
        "scores": span_scores([row["label"] for row in rows], predicted),
    }
    target = REPO_ROOT / "results/slot_lexicon_esci.json"
    target.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out["scores"], indent=1))
    for row, mine in zip(rows, predicted, strict=True):
        theirs = {k: v for k, v in row["label"].items() if v}
        own = {k: v for k, v in mine.items() if v}
        if theirs != own:
            print(f"{queries[row['id']]!r:55s} teacher {theirs}  lexicon {own}")


if __name__ == "__main__":
    main()
