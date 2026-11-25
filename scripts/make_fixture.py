"""Cut the 5-query, 30-product sample that CI runs on.

Only dev-fold queries are used: the fixture is read while writing code, so a test-fold query in it
would break the protocol.
"""

import random

import pandas as pd

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import judgments, load_labels, load_products, load_queries, split_classes
from shelfrank.eval.splits import load_folds

SEED = 7
PER_GRADE = 2
TARGET = REPO_ROOT / "tests" / "fixtures" / "wands"


def main() -> None:
    """Select and write the sample in the original file format."""
    cfg = load_config("data")
    raw = REPO_ROOT / cfg["wands"]["raw_dir"]
    queries, products, labels = load_queries(raw), load_products(raw), load_labels(raw)
    judged = judgments(labels)
    dev = load_folds(REPO_ROOT / cfg["splits"]["path"]).dev
    rng = random.Random(SEED)

    counts = judged.groupby(["query_id", "grade"]).size().unstack(fill_value=0)
    rich = [q for q in dev if (counts.loc[q] >= 3 * PER_GRADE).all()]
    no_exact = [q for q in dev if counts.loc[q, 2] == 0 and counts.loc[q, 1] >= 6]
    chosen = sorted([*rng.sample(rich, 4), *rng.sample(no_exact, 1)])

    # Prefer awkward products so the sample exercises missing and multi-valued cells.
    awkward = (
        products["product_class"].map(lambda c: len(split_classes(c)) != 1)
        | products["average_rating"].isna()
        | products["product_description"].isna()
    )
    awkward_ids = set(products.loc[awkward, "product_id"])
    picked: list[int] = []
    for query_id in chosen:
        pool = judged[judged["query_id"] == query_id]
        for grade, want in ((2, PER_GRADE), (1, PER_GRADE), (0, PER_GRADE)):
            ids = [
                int(p) for p in pool.loc[pool["grade"] == grade, "product_id"] if p not in picked
            ]
            rng.shuffle(ids)
            ids.sort(key=lambda p: p not in awkward_ids)
            # One awkward product and the rest ordinary, so neither kind dominates.
            picked.extend(ids[:1] + ids[::-1][: want - 1])
    spare = [int(p) for p in judged.loc[judged["query_id"].isin(chosen), "product_id"].unique()]
    rng.shuffle(spare)
    for product_id in spare:
        if len(set(picked)) >= 30:
            break
        picked.append(product_id)
    keep = sorted(set(picked))[:30]

    TARGET.mkdir(parents=True, exist_ok=True)
    sample = {
        "query.csv": queries[queries["query_id"].isin(chosen)],
        "product.csv": products[products["product_id"].isin(keep)].rename(
            columns={"category_hierarchy": "category hierarchy"}
        ),
        "label.csv": labels[labels["query_id"].isin(chosen) & labels["product_id"].isin(keep)],
    }
    for name, frame in sample.items():
        pd.DataFrame(frame).to_csv(
            TARGET / name, sep="\t", index=False, encoding="utf-8", lineterminator="\n"
        )
        print(f"wrote {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
