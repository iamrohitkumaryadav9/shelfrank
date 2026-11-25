"""Feature groups. A column belongs to the group named by its prefix."""

from collections.abc import Iterable, Sequence

GROUPS = {
    "G1": "lexical match",
    "G2": "semantic similarity",
    "G3": "query class",
    "G4": "attribute match",
    "G5": "product priors",
    "G6": "text shape",
}


def group_of(column: str) -> str:
    """Group of a feature column."""
    group = column.split("_", 1)[0].upper()
    if group not in GROUPS:
        raise ValueError(f"{column} has no feature group")
    return group


def columns_of(columns: Iterable[str], groups: Sequence[str]) -> list[str]:
    """Columns in the given groups, in their original order."""
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")
    return [c for c in columns if group_of(c) in groups]
