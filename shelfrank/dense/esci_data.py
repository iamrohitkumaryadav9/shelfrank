"""Build ESCI training triplets. Runs in Colab, tested here on synthetic files.

Only pyarrow and the standard library are used.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from shelfrank.qu.seed_selection import QueryEvidence, filter_queries

# Gains of the ESCI paper, used only for the held-out ESCI check.
ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
Judged = dict[str, dict[str, str]]


@dataclass
class TrainingData:
    """Queries, passages and the judgments that connect them."""

    queries: dict[str, str] = field(default_factory=dict)
    passages: dict[str, str] = field(default_factory=dict)
    train: Judged = field(default_factory=dict)
    held_out: Judged = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Write to one file, under a temporary name first so that a half file is never read."""
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(asdict(self)), encoding="utf-8")
        partial.replace(path)

    @staticmethod
    def load(path: Path) -> "TrainingData":
        """Read a saved file."""
        return TrainingData(**json.loads(path.read_text(encoding="utf-8")))


def read_judgments(examples: Path, locale: str, split: str) -> tuple[dict[str, str], Judged]:
    """Queries and judgments of the small version, with the filter pushed into the scan."""
    import pyarrow.dataset as ds

    wanted = (
        (ds.field("product_locale") == locale)
        & (ds.field("split") == split)
        & (ds.field("small_version") == 1)
    )
    columns = ["query_id", "query", "product_id", "esci_label"]
    queries: dict[str, str] = {}
    judged: Judged = defaultdict(dict)
    for batch in ds.dataset(examples).scanner(columns=columns, filter=wanted).to_batches():
        values = [batch.column(i).to_pylist() for i in range(4)]
        for query_id, text, product, label in zip(*values, strict=True):
            queries[str(query_id)] = str(text).strip()
            judged[str(query_id)][product] = label
    return queries, dict(judged)


def passage(row: Mapping[str, object], max_chars: int) -> str:
    """Title, brand, colour and the start of the bullet points, most specific first."""
    parts = [row.get("product_title"), row.get("product_brand"), row.get("product_color")]
    bullets = row.get("product_bullet_point")
    if isinstance(bullets, str):
        parts.append(bullets[:max_chars])
    return " . ".join(" ".join(p.split()) for p in parts if isinstance(p, str) and p.strip())


def read_passages(
    products: Path, locale: str, wanted: set[str], max_chars: int, batch_size: int = 20_000
) -> dict[str, str]:
    """Stream the product file and keep only judged products."""
    import pyarrow.dataset as ds

    columns = ["product_title", "product_brand", "product_color", "product_bullet_point"]
    out: dict[str, str] = {}
    scanner = ds.dataset(products).scanner(
        columns=["product_id", *columns],
        filter=ds.field("product_locale") == locale,
        batch_size=batch_size,
        batch_readahead=2,
        fragment_readahead=1,
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if row["product_id"] in wanted:
                text = passage(row, max_chars)
                if text:
                    out[row["product_id"]] = text
    return out


def prepare(
    examples: Path,
    products: Path,
    protected: Iterable[str],
    settings: Mapping[str, Any],
) -> TrainingData:
    """Read ESCI, drop look-alikes of protected queries, and hold queries out."""
    queries, judged = read_judgments(examples, settings["locale"], settings["split"])
    evidence = {q: QueryEvidence(text) for q, text in queries.items()}
    kept, dropped = filter_queries(evidence, protected, settings["near_duplicate_jaccard"])
    ids = sorted(kept)
    random.Random(settings["seed"]).shuffle(ids)
    held_ids, train_ids = ids[: settings["n_held_out"]], ids[settings["n_held_out"] :]
    # A query with no Exact product gives no positive.
    train_ids = [q for q in train_ids if "E" in judged[q].values()]
    wanted = {p for q in (*held_ids, *train_ids) for p in judged[q]}
    passages = read_passages(products, settings["locale"], wanted, settings["max_bullet_chars"])

    def with_text(ids: Sequence[str]) -> Judged:
        return {q: {p: v for p, v in judged[q].items() if p in passages} for q in ids}

    data = TrainingData(
        {q: queries[q] for q in (*held_ids, *train_ids)},
        passages,
        with_text(train_ids),
        with_text(held_ids),
    )
    data.report = {
        "queries_read": len(queries),
        "dropped": dict(sorted(dropped.items())),
        "train_queries": len(data.train),
        "held_out_queries": len(data.held_out),
        "passages": len(passages),
        "positives": sum(list(v.values()).count("E") for v in data.train.values()),
    }
    return data


@dataclass(frozen=True)
class Triplet:
    """A query, an Exact product and an Irrelevant product judged for the same query."""

    query: str
    positive: str
    negative: str
    hard: bool


def make_triplets(judged: Judged, passages: Mapping[str, str], seed: int) -> list[list[Triplet]]:
    """Triplets grouped in rounds; a round holds at most one triplet per query.

    A query with no Irrelevant judgment gets a random product, recorded as not hard.
    """
    rng = random.Random(seed)
    everything = sorted(passages)
    rounds: dict[int, list[Triplet]] = defaultdict(list)
    for query, labels in sorted(judged.items()):
        positives = sorted(p for p, v in labels.items() if v == "E")
        negatives = sorted(p for p, v in labels.items() if v == "I")
        rng.shuffle(positives)
        rng.shuffle(negatives)
        for i, positive in enumerate(positives):
            if negatives:
                negative, hard = negatives[i % len(negatives)], True
            else:
                negative, hard = rng.choice(everything), False
            rounds[i].append(Triplet(query, positive, negative, hard))
    for group in rounds.values():
        rng.shuffle(group)
    return [rounds[i] for i in sorted(rounds)]


def make_batches(rounds: Sequence[Sequence[Triplet]], size: int, seed: int) -> list[list[Triplet]]:
    """Batches with distinct queries, in random order. Short leftovers are dropped.

    Distinct queries matter: two positives of one query in a batch would be each other's negative.
    """
    batches = [
        list(group[start : start + size])
        for group in rounds
        for start in range(0, len(group) - size + 1, size)
    ]
    random.Random(seed).shuffle(batches)
    return batches
