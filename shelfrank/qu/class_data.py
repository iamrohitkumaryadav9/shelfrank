"""Training examples of each class-student arm, shared by the comparison and the ranker."""

import json
from pathlib import Path

import pandas as pd

from shelfrank.config import REPO_ROOT
from shelfrank.qu.class_student import Examples, mix, name_examples, query_examples, split_seed
from shelfrank.qu.guidelines import CLASS_VERSION

SEED, HELD_SHARE = 20260, 0.2
NAMES, QUERIES, BOTH = "names", "queries", "both"
Labelled = tuple[list[str], list[str | None]]


def labelled(source: str, text_file: str, root: Path = REPO_ROOT) -> Labelled | None:
    """Teacher-labelled ESCI queries, or None before they exist."""
    path = root / "data/labels" / f"class_{source}_{CLASS_VERSION}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    texts = (root / "data/esci" / text_file).read_text(encoding="utf-8").splitlines()
    return [texts[row["id"]] for row in rows], [row["label"] for row in rows]


def seed_parts(root: Path = REPO_ROOT) -> tuple[Examples, Labelled] | None:
    """Seed queries to train on, and the held-out ones to select on."""
    seed = labelled("esci_seed", "esci_seed_queries.txt", root)
    if seed is None:
        return None
    train, held = split_seed(len(seed[0]), HELD_SHARE, SEED)
    examples = query_examples([seed[0][i] for i in train], [seed[1][i] for i in train])
    return examples, ([seed[0][i] for i in held], [seed[1][i] for i in held])


def arm_examples(
    arm: str,
    tail: int | None,
    query_share: float | None,
    products: pd.DataFrame,
    taxonomy: list[str],
    queries: Examples | None,
) -> Examples:
    """What one arm trains on."""
    if arm == QUERIES:
        assert queries is not None
        return queries
    names = name_examples(products, taxonomy, tail)
    if arm == NAMES:
        return names
    assert queries is not None and query_share is not None
    return mix(names, queries, query_share)
