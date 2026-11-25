"""Product-side work done once at build time: field tokens and listed attributes.

The features need the tokens of every candidate and its attribute values. Neither depends on the query,
and computing them per search took most of the ranker's time. Stored values come from the same
functions, so the features do not change.
"""

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from shelfrank.features.attributes import Listed

FIELDS = ("name", "class", "features", "description")


class TokenTable:
    """Token lists of every product and field, as ids into one vocabulary."""

    def __init__(
        self, vocabulary: Sequence[str], fields: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Keep a vocabulary and, per field, row starts and token ids."""
        self.vocabulary = list(vocabulary)
        self.fields = dict(fields)

    @classmethod
    def build(
        cls: type["TokenTable"], tokens: Mapping[str, Sequence[Sequence[str]]]
    ) -> "TokenTable":
        """From token lists, in product row order."""
        index: dict[str, int] = {}
        fields = {}
        for name in FIELDS:
            starts, ids = [0], []
            for row in tokens[name]:
                ids += [index.setdefault(t, len(index)) for t in row]
                starts.append(len(ids))
            fields[name] = (np.asarray(starts, dtype=np.int64), np.asarray(ids, dtype=np.int32))
        return cls(sorted(index, key=index.__getitem__), fields)

    def row(self, field: str, row: int) -> list[str]:
        """Tokens of one product, in order."""
        starts, ids = self.fields[field]
        return [self.vocabulary[i] for i in ids[starts[row] : starts[row + 1]].tolist()]

    def view(self, field: str) -> "FieldView":
        """Indexable like a list of token lists."""
        return FieldView(self, field)

    def save(self, path: Path) -> None:
        """One compressed file."""
        arrays = {
            f"{n}_{k}": a
            for n, (s, i) in self.fields.items()
            for k, a in (("starts", s), ("ids", i))
        }
        np.savez_compressed(path, vocabulary=np.array(self.vocabulary), **arrays)

    @classmethod
    def load(cls: type["TokenTable"], path: Path) -> "TokenTable":
        """Read it back."""
        with np.load(path, allow_pickle=False) as body:
            fields = {n: (body[f"{n}_starts"], body[f"{n}_ids"]) for n in FIELDS}
            return cls(body["vocabulary"].tolist(), fields)


class FieldView:
    """One field of a token table, read with ``view[row]``."""

    def __init__(self, table: TokenTable, field: str) -> None:
        self.table, self.field = table, field
        self.cache: dict[int, list[str]] = {}

    def __getitem__(self, row: int) -> list[str]:
        if row not in self.cache:
            self.cache[row] = self.table.row(self.field, row)
        return self.cache[row]


def _terms(values: set[tuple[str, ...]]) -> list[list[str]]:
    return [list(v) for v in sorted(values)]


def save_listed(listed: Sequence[Listed], path: Path) -> None:
    """Attribute values of every product, in row order."""
    body = [[_terms(x.union), _terms(x.style), _terms(x.room), sorted(x.mentioned)] for x in listed]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(body, handle, separators=(",", ":"))


def load_listed(path: Path) -> list[Listed]:
    """Read them back."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        body = json.load(handle)
    return [
        Listed({tuple(t) for t in u}, {tuple(t) for t in s}, {tuple(t) for t in r}, set(m))
        for u, s, r, m in body
    ]
