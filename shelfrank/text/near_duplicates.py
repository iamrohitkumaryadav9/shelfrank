"""Find queries similar enough that putting them in different folds would leak labels."""

import re
from collections.abc import Mapping
from itertools import combinations

_TOKEN = re.compile(r"[a-z0-9]+")


def token_key(query: str) -> frozenset[str]:
    """Reduce a query to a word-order-free set with plural endings removed.

    Deliberately crude: this only has to over-group near-identical queries, not rank anything.
    """
    tokens = _TOKEN.findall(query.lower())
    return frozenset(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in tokens)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Set overlap; two empty sets are unrelated so that blank keys never group together."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def near_duplicate_groups(queries: Mapping[int, str], threshold: float) -> list[list[int]]:
    """Return connected components (size two or more) of the similarity graph.

    Components rather than pairs: if A resembles B and B resembles C, all three must travel
    together even when A and C fall just under the threshold.
    """
    parent = {query_id: query_id for query_id in queries}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    keys = {query_id: token_key(text) for query_id, text in queries.items()}
    for a, b in combinations(sorted(keys), 2):
        if jaccard(keys[a], keys[b]) >= threshold:
            parent[find(b)] = find(a)

    groups: dict[int, list[int]] = {}
    for query_id in sorted(parent):
        groups.setdefault(find(query_id), []).append(query_id)
    return [members for members in groups.values() if len(members) > 1]
