"""Candidate rows with features and grades, and the two training variants."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shelfrank.eval.harness import Qrels

JUDGED_ONLY, UNJUDGED_ZERO = "judged_only", "unjudged_zero"
VARIANTS = (JUDGED_ONLY, UNJUDGED_ZERO)


@dataclass
class RankingData:
    """One row per candidate pair, sorted by query. Grade is missing when unjudged."""

    query_id: np.ndarray
    product_id: np.ndarray
    features: pd.DataFrame
    grade: np.ndarray

    def __post_init__(self) -> None:
        """Rows of a query must be adjacent, because LightGBM reads groups by position."""
        if len(self.query_id) and np.any(np.diff(self.query_id) < 0):
            raise ValueError("rows must be sorted by query_id")

    def rows(self, query_ids: Sequence[int], variant: str | None = None) -> np.ndarray:
        """Row numbers of some queries. A variant selects training rows; None keeps all."""
        mask = np.isin(self.query_id, list(query_ids))
        if variant == JUDGED_ONLY:
            mask &= ~np.isnan(self.grade)
        elif variant not in (UNJUDGED_ZERO, None):
            raise ValueError(f"unknown variant {variant}")
        return np.flatnonzero(mask)

    def labels(self, rows: np.ndarray) -> np.ndarray:
        """Integer grades, with unjudged as 0."""
        return np.nan_to_num(self.grade[rows], nan=0.0).astype(int)


def build_data(pairs: pd.DataFrame, features: pd.DataFrame, qrels: Qrels) -> RankingData:
    """Attach grades to candidate pairs. Features are built before any grade is known."""
    if len(pairs) != len(features):
        raise ValueError("features and pairs differ in length")
    keys = zip(pairs["query_id"], pairs["product_id"], strict=True)
    return RankingData(
        pairs["query_id"].to_numpy(),
        pairs["product_id"].to_numpy(),
        features.reset_index(drop=True),
        np.asarray([qrels[q].get(p, np.nan) for q, p in keys], dtype=float),
    )


def group_sizes(query_id: np.ndarray) -> list[int]:
    """Rows per query, in order."""
    if len(query_id) == 0:
        return []
    starts = np.flatnonzero(np.r_[True, np.diff(query_id) != 0])
    return np.diff(np.r_[starts, len(query_id)]).tolist()
