"""Create data/splits/wands_folds.json once.

Refuses to overwrite: after any result exists, changing the folds would invalidate it.
"""

import sys

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.splits import Folds, assign_folds, save_folds
from shelfrank.text.near_duplicates import near_duplicate_groups


def main() -> int:
    """Build and save the assignment."""
    cfg = load_config("data")
    split_cfg = cfg["splits"]
    path = REPO_ROOT / split_cfg["path"]
    if path.exists():
        print(f"{path.name} already exists; folds are frozen and will not be regenerated")
        return 1

    queries = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id")
    classes = {
        int(q): (c if isinstance(c, str) else None) for q, c in queries["query_class"].items()
    }
    texts = {int(q): str(t) for q, t in queries["query"].items()}
    groups = near_duplicate_groups(texts, split_cfg["near_duplicate_jaccard"])
    assignment = assign_folds(classes, groups, split_cfg["n_folds"], split_cfg["seed"])
    save_folds(
        Folds(assignment, split_cfg["n_folds"], split_cfg["dev_fold"]),
        path,
        {
            "method": "greedy stratified by query_class (classes below n_folds pooled), equal sizes",
            "seed": split_cfg["seed"],
            "near_duplicate_jaccard": split_cfg["near_duplicate_jaccard"],
            "near_duplicate_groups": groups,
            "wands_commit": cfg["wands"]["commit"],
        },
    )
    print(f"wrote {path.name}: {len(assignment)} queries, {len(groups)} near-duplicate groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
