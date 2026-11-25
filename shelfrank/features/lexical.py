"""G1: how the query's words match the product's text."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise

import numpy as np
import pandas as pd

from shelfrank.ltr.first_stage import FirstStage

COVERED = ("name", "class", "features", "description")


def idf_table(documents: Sequence[Sequence[str]]) -> dict[str, float]:
    """Inverse document frequency from the catalogue's names."""
    counts = Counter(t for d in documents for t in set(d))
    return {t: math.log(1 + len(documents) / n) for t, n in counts.items()}


def coverage(query: Sequence[str], document: set[str]) -> float:
    """Share of the query's distinct words that the text contains."""
    words = set(query)
    return len(words & document) / len(words) if words else 0.0


def weighted_coverage(query: Sequence[str], document: set[str], idf: Mapping[str, float]) -> float:
    """Coverage where rare words count more. A word no name contains counts as the rarest."""
    top = max(idf.values(), default=1.0)
    weights = {t: idf.get(t, top) for t in sorted(set(query))}
    total = sum(weights.values())
    return sum(v for t, v in weights.items() if t in document) / total if total else 0.0


def bigram_share(query: Sequence[str], document: Sequence[str]) -> float:
    """Share of adjacent query word pairs that are adjacent in the text."""
    pairs = set(pairwise(query))
    if not pairs:
        return 0.0
    return len(pairs & set(pairwise(document))) / len(pairs)


def contains_phrase(query: Sequence[str], document: Sequence[str]) -> bool:
    """Whether the whole query occurs in the text in order."""
    n, words = len(query), list(query)
    return n > 0 and any(list(document[i : i + n]) == words for i in range(len(document) - n + 1))


def relative(matrix: np.ndarray) -> np.ndarray:
    """Scores divided by the query's best score, so that queries are comparable."""
    best = matrix.max(axis=1, keepdims=True)
    return np.divide(matrix, best, out=np.zeros_like(matrix), where=best > 0)


def reciprocal_ranks(
    pairs: pd.DataFrame, run: Mapping[int, Sequence[int]], k: int = 60
) -> list[float]:
    """1 / (k + rank), and 0 for a product outside the run."""
    ranks = {q: {p: i for i, p in enumerate(ranking, 1)} for q, ranking in run.items()}
    out = []
    for q, p in zip(pairs["query_id"], pairs["product_id"], strict=True):
        rank = ranks[q].get(p)
        out.append(1 / (k + rank) if rank else 0.0)
    return out


def lexical_features(
    pairs: pd.DataFrame,
    stage: FirstStage,
    query_tokens: Sequence[Sequence[str]],
    tokens: Mapping[str, list[list[str]]],
) -> pd.DataFrame:
    """Columns g1_*."""
    qi, pi = pairs["qi"].to_numpy(), pairs["pi"].to_numpy()
    out: dict[str, object] = {}
    for name, matrix in stage.fields.items():
        out[f"g1_bm25_{name}"] = matrix[qi, pi]
        out[f"g1_bm25_{name}_rel"] = relative(matrix)[qi, pi]
    out["g1_bm25_tuned"] = stage.tuned[qi, pi]
    out["g1_bm25_tuned_rel"] = relative(stage.tuned)[qi, pi]
    out["g1_rr_bm25"] = reciprocal_ranks(pairs, stage.runs["bm25_multi"])

    idf = idf_table(tokens["name"])
    wanted = sorted(set(pi.tolist()))
    sets = {f: {p: set(tokens[f][p]) for p in wanted} for f in COVERED}
    rows = []
    for q, p in zip(qi.tolist(), pi.tolist(), strict=True):
        query, name = query_tokens[q], tokens["name"][p]
        everything = set().union(*(sets[f][p] for f in COVERED))
        rows.append(
            (
                *(coverage(query, sets[f][p]) for f in COVERED),
                coverage(query, everything),
                weighted_coverage(query, sets["name"][p], idf),
                bigram_share(query, name),
                float(contains_phrase(query, name)),
                float(bool(query) and query[-1] in sets["name"][p]),
                float(bool(query) and query[-1] in sets["class"][p]),
            )
        )
    columns = [f"g1_cover_{f}" for f in COVERED]
    columns += ["g1_cover_any", "g1_cover_name_idf", "g1_bigrams_name", "g1_phrase_name"]
    columns += ["g1_last_word_in_name", "g1_last_word_in_class"]
    return pd.concat([pd.DataFrame(out), pd.DataFrame(rows, columns=columns)], axis=1)
