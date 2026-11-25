"""G6: lengths, and the numbers a query mentions."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

NAMES = (
    "query_words",
    "query_numbers",
    "name_words",
    "log_description_words",
    "numbers_in_name",
    "numbers_anywhere",
)


def numbers(tokens: Sequence[str]) -> set[str]:
    """Numeric tokens."""
    return {t for t in tokens if t[0].isdigit()}


def shape_features(
    pairs: pd.DataFrame,
    query_tokens: Sequence[Sequence[str]],
    tokens: Mapping[str, list[list[str]]],
) -> pd.DataFrame:
    """Columns g6_*. Number match is missing when the query has no number."""
    rows = []
    for q, p in zip(pairs["qi"].tolist(), pairs["pi"].tolist(), strict=True):
        query, name = query_tokens[q], tokens["name"][p]
        wanted = numbers(query)
        anywhere = set(name) | set(tokens["features"][p])
        rows.append(
            (
                len(query),
                len(wanted),
                len(name),
                np.log1p(len(tokens["description"][p])),
                len(wanted & set(name)) / len(wanted) if wanted else np.nan,
                len(wanted & anywhere) / len(wanted) if wanted else np.nan,
            )
        )
    return pd.DataFrame(rows, columns=[f"g6_{c}" for c in NAMES])
