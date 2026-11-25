"""Choose the classifier's label set from the catalogue alone."""

from collections.abc import Iterable

import pandas as pd

from shelfrank.data.wands import split_classes

OTHER = "other"


def class_support(product_classes: Iterable[object]) -> pd.Series:
    """Count products per atomic class, ties broken by name so the order is deterministic.

    A multi-class product counts once for each of its classes.
    """
    counts: dict[str, int] = {}
    for cell in product_classes:
        for name in set(split_classes(cell)):
            counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return pd.Series({name: count for name, count in ordered}, dtype=int)


def build_taxonomy(support: pd.Series, min_support: int) -> list[str]:
    """Keep classes with enough product names to learn from.

    The rule reads no query and no query_class, so the label set cannot be fitted to the test
    folds. Everything below the threshold is predicted as ``other``.
    """
    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    return [str(name) for name, count in support.items() if count >= min_support]


def product_coverage(product_classes: Iterable[object], taxonomy: Iterable[str]) -> float:
    """Share of products that have at least one class in the taxonomy."""
    kept = set(taxonomy)
    flags = [bool(kept & set(split_classes(cell))) for cell in product_classes]
    return sum(flags) / len(flags) if flags else 0.0


def query_coverage(query_classes: Iterable[object], taxonomy: Iterable[str]) -> float:
    """Share of queries whose reference class is in the taxonomy; a missing class is uncovered."""
    kept = set(taxonomy)
    flags = [cell in kept for cell in query_classes]
    return sum(flags) / len(flags) if flags else 0.0
