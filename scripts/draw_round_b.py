"""Draw the Round B queries by id, once. No query text is read."""

import json
import random

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.eval.splits import load_folds

SEED, SIZE = 20260, 50
PATH = REPO_ROOT / "data/splits/round_b_draw.json"


def main() -> None:
    """Write the ids, and refuse to overwrite them."""
    if PATH.exists():
        raise SystemExit(f"{PATH.name} exists and is never redrawn")
    folds = load_folds(REPO_ROOT / load_config("data")["splits"]["path"])
    ids = sorted(random.Random(SEED).sample(folds.primary, SIZE))
    body = {"seed": SEED, "drawn_from": "folds 1 to 4", "query_ids": ids}
    PATH.write_text(json.dumps(body) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(ids)} ids")


if __name__ == "__main__":
    main()
