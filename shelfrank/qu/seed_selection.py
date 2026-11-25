"""Choose ESCI queries to label. Runs inside the Colab notebook, tested here.

Only pyarrow and the standard library are used, so the notebook needs no installation.
"""

import hashlib
import random
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shelfrank.text.near_duplicates import jaccard, leakage_keys
from shelfrank.text.normalize import tokenize

_ASIN = re.compile(r"^b0[a-z0-9]{8}$")
NO_CLASS = "__no_class__"


@dataclass
class QueryEvidence:
    """How many of a query's Exact products matched each class."""

    text: str
    products: list[str] = field(default_factory=list)
    votes: Counter = field(default_factory=Counter)


def is_usable(query: str) -> bool:
    """Keep plain-text queries a shopper could type; drop product codes and other scripts."""
    words = query.split()
    letters = sum(char.isalpha() for char in query)
    if not query.isascii() or _ASIN.match(query.lower()):
        return False
    return 1 <= len(words) <= 10 and letters >= 3


def term_index(class_terms: Mapping[str, Iterable[str]]) -> dict[tuple[str, ...], str]:
    """Invert class terms for lookup."""
    return {tuple(term.split()): label for label, terms in class_terms.items() for term in terms}


def match_class(title: str, index: Mapping[tuple[str, ...], str], max_n: int = 4) -> str | None:
    """Class of the longest term in a title, earliest on a tie.

    Longest first, so "desk lamp" is read as a lamp and not as a desk. Earliest next, because
    marketplace titles put the product type before the search keywords.
    """
    tokens = tokenize(title)
    for size in range(min(max_n, len(tokens)), 0, -1):
        for start in range(len(tokens) - size + 1):
            label = index.get(tuple(tokens[start : start + size]))
            if label is not None:
                return label
    return None


def read_exact_pairs(examples: Path, locale: str, split: str) -> dict[str, QueryEvidence]:
    """Queries with their Exact products. The filter is pushed into the parquet scan."""
    import pyarrow.dataset as ds

    wanted = (
        (ds.field("product_locale") == locale)
        & (ds.field("esci_label") == "E")
        & (ds.field("split") == split)
    )
    queries: dict[str, QueryEvidence] = {}
    scanner = ds.dataset(examples).scanner(
        columns=["query_id", "query", "product_id"], filter=wanted
    )
    for batch in scanner.to_batches():
        columns = [batch.column(i).to_pylist() for i in range(3)]
        for query_id, text, product_id in zip(*columns, strict=True):
            evidence = queries.setdefault(str(query_id), QueryEvidence(str(text).strip()))
            evidence.products.append(product_id)
    return queries


def add_votes(
    queries: Mapping[str, QueryEvidence],
    products: Path,
    locale: str,
    index: Mapping[tuple[str, ...], str],
    batch_size: int = 20_000,
) -> int:
    """Stream product titles and count class matches per query; return the titles read.

    Batches are dropped after use and only the two needed columns are read, so memory stays flat
    however large the products file is.
    """
    import pyarrow.dataset as ds

    wanted: dict[str, list[str]] = defaultdict(list)
    for query_id, evidence in queries.items():
        for product_id in evidence.products:
            wanted[product_id].append(query_id)
    read = 0
    scanner = ds.dataset(products).scanner(
        columns=["product_id", "product_title"],
        filter=ds.field("product_locale") == locale,
        batch_size=batch_size,
        batch_readahead=2,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        ids, titles = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        for product_id, title in zip(ids, titles, strict=True):
            if product_id in wanted and title:
                read += 1
                label = match_class(title, index)
                if label is not None:
                    for query_id in wanted[product_id]:
                        queries[query_id].votes[label] += 1
    return read


def assign_pools(
    queries: Mapping[str, QueryEvidence], min_votes: int, min_share: float
) -> dict[str, list[str]]:
    """Group query ids by their majority class, or by having no class evidence at all.

    Queries with weak evidence join neither pool. The teacher decides every label, so this only has
    to raise the share of home-goods queries in the sample.
    """
    pools: dict[str, list[str]] = defaultdict(list)
    for query_id, evidence in sorted(queries.items()):
        if not evidence.votes:
            if len(evidence.products) >= min_votes:
                pools[NO_CLASS].append(query_id)
            continue
        label, count = min(evidence.votes.items(), key=lambda item: (-item[1], item[0]))
        if count >= min_votes and count / len(evidence.products) >= min_share:
            pools[label].append(query_id)
    return dict(pools)


def spread(pools: Mapping[str, Sequence[str]], total: int, rng: random.Random) -> list[str]:
    """Take one query per class in turn until ``total``, so rare classes are represented."""
    shuffled = {label: rng.sample(list(ids), len(ids)) for label, ids in sorted(pools.items())}
    chosen: list[str] = []
    while len(chosen) < total and any(shuffled.values()):
        for label in sorted(shuffled):
            if shuffled[label] and len(chosen) < total:
                chosen.append(shuffled[label].pop())
    return chosen


def resembles(query: str, keys: Collection[frozenset[str]], threshold: float) -> bool:
    """Whether a query is the same as, or nearly the same as, any protected query."""
    return any(jaccard(own, key) >= threshold for own in leakage_keys(query) for key in keys)


def filter_queries(
    queries: Mapping[str, QueryEvidence], protected: Iterable[str], threshold: float
) -> tuple[dict[str, QueryEvidence], Counter]:
    """Drop unusable queries, repeats, and anything that resembles a protected query."""
    by_token: dict[str, set[frozenset[str]]] = defaultdict(set)
    for text in protected:
        for key in leakage_keys(text):
            for token in key:
                by_token[token].add(key)
    kept: dict[str, QueryEvidence] = {}
    seen: set[frozenset[str]] = set()
    dropped: Counter = Counter()
    for query_id, evidence in sorted(queries.items()):
        keys = leakage_keys(evidence.text)
        tokens = frozenset().union(*keys)
        nearby = set().union(*(by_token[t] for t in tokens if t in by_token))
        if not keys or not is_usable(evidence.text):
            dropped["unusable"] += 1
        elif seen.intersection(keys):
            dropped["repeat"] += 1
        elif resembles(evidence.text, nearby, threshold):
            dropped["resembles a protected query"] += 1
        else:
            seen.update(keys)
            kept[query_id] = evidence
    return kept, dropped


def select(
    pools: Mapping[str, Sequence[str]],
    n_seed: int,
    n_dev: int,
    home_share: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Draw disjoint seed and prompt-development samples from the same mixture."""
    rng = random.Random(seed)
    total = n_seed + n_dev
    n_home = round(total * home_share)
    home = spread({k: v for k, v in pools.items() if k != NO_CLASS}, n_home, rng)
    outside = list(pools.get(NO_CLASS, []))
    chosen = home + rng.sample(outside, min(total - n_home, len(outside)))
    if len(chosen) < total:
        raise ValueError(f"only {len(chosen)} of {total} queries available: loosen the settings")
    rng.shuffle(chosen)
    return chosen[n_dev:], chosen[:n_dev]


def write_outputs(
    directory: Path, seed_texts: Sequence[str], dev_texts: Sequence[str]
) -> dict[str, Any]:
    """Write one query per line and return the hashes for the manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, texts in (("esci_seed_queries.txt", seed_texts), ("esci_prompt_dev.txt", dev_texts)):
        body = "".join(" ".join(text.split()) + "\n" for text in texts).encode("utf-8")
        (directory / name).write_bytes(body)
        out[name] = {"lines": len(texts), "sha256": hashlib.sha256(body).hexdigest()}
    return out
