"""Per-field BM25: scores stay separate so that weights can be tuned and reused as features."""

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain

import bm25s
import numpy as np
import pandas as pd

from shelfrank.data.product_features import parse_features
from shelfrank.text.normalize import tokenize

FIELDS = ("name", "class", "hierarchy", "features", "description")


def feature_text(raw: object) -> str:
    """Join the informative feature values.

    Keys are internal names ("overallwidth-sidetoside") that no shopper types. Yes/no flags and bare
    numbers are dropped because without their key they would match unrelated queries.
    """
    values = dict.fromkeys(value for _, value in parse_features(raw).pairs)
    return " ".join(
        v for v in values if v not in ("yes", "no") and any(char.isalpha() for char in v)
    )


def field_tokens(products: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Tokenise each field once for all variants."""
    return {
        "name": [tokenize(t) for t in products["product_name"]],
        "class": [tokenize(t) for t in products["product_class"]],
        "hierarchy": [tokenize(t) for t in products["category_hierarchy"]],
        "features": [tokenize(feature_text(t)) for t in products["product_features"]],
        "description": [tokenize(t) for t in products["product_description"]],
    }


def concat_fields(tokens: Mapping[str, list[list[str]]], names: Sequence[str]) -> list[list[str]]:
    """Merge fields into one document, for single-index variants."""
    columns = [tokens[name] for name in names]
    return [list(chain.from_iterable(parts)) for parts in zip(*columns, strict=True)]


class BM25Index:
    """BM25 over one field."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float, b: float) -> None:
        """Index documents that are already tokenised."""
        self.size = len(documents)
        self._vocabulary = {token for document in documents for token in document}
        self._engine = bm25s.BM25(k1=k1, b=b, method="lucene")
        if self._vocabulary:
            # A blank term keeps one row per product when a product lacks the field.
            self._engine.index([list(d) or [""] for d in documents], show_progress=False)

    def scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        """Score every document. Repeated and unknown query terms are dropped."""
        terms = [t for t in dict.fromkeys(query_tokens) if t in self._vocabulary]
        if not terms:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self._engine.get_scores(terms), dtype=np.float32)


def score_matrix(index: BM25Index, queries: Sequence[str]) -> np.ndarray:
    """Queries by products."""
    return np.stack([index.scores(tokenize(query)) for query in queries])


def combine(matrices: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Weighted sum of field scores. An unknown field name is an error rather than a zero."""
    unknown = set(weights) - set(matrices)
    if unknown:
        raise KeyError(f"no scores for fields: {sorted(unknown)}")
    first = next(iter(matrices.values()))
    total = np.zeros_like(first)
    for name, weight in weights.items():
        if weight:
            total += np.float32(weight) * matrices[name]
    return total
