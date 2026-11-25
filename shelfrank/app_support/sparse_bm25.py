"""A BM25 index as plain arrays: one column of document scores per term."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from shelfrank.bm25.fields import BM25Index


class SparseBM25:
    """Scores equal to ``BM25Index`` without building an index at start-up."""

    def __init__(
        self,
        terms: np.ndarray,
        indptr: np.ndarray,
        indices: np.ndarray,
        data: np.ndarray,
        size: int,
    ) -> None:
        """Keep the postings."""
        self.column = {str(t): i for i, t in enumerate(terms)}
        self.indptr, self.indices, self.data, self.size = indptr, indices, data, size

    def scores(self, query_tokens: Iterable[str]) -> np.ndarray:
        """Score every document. Repeated and unknown terms are dropped."""
        out = np.zeros(self.size, dtype=np.float32)
        for token in dict.fromkeys(query_tokens):
            c = self.column.get(token)
            if c is not None:
                part = slice(self.indptr[c], self.indptr[c + 1])
                out[self.indices[part]] += self.data[part]
        return out

    def save(self, path: Path) -> None:
        """Write one compressed file."""
        np.savez_compressed(
            path,
            terms=np.array(list(self.column)),
            indptr=self.indptr,
            indices=self.indices,
            data=self.data,
            size=np.array(self.size),
        )

    @classmethod
    def load(cls: type["SparseBM25"], path: Path) -> "SparseBM25":
        """Read a saved file."""
        with np.load(path, allow_pickle=False) as body:
            arrays = (body["terms"], body["indptr"], body["indices"], body["data"])
            return cls(*arrays, int(body["size"]))

    @classmethod
    def from_index(cls: type["SparseBM25"], index: BM25Index) -> "SparseBM25":
        """Copy a built index."""
        engine = index._engine
        vocabulary = {t: i for t, i in engine.vocab_dict.items() if t in index._vocabulary}
        terms = sorted(vocabulary)
        starts, rows, values = [0], [], []
        indptr = np.asarray(engine.scores["indptr"])
        for term in terms:
            part = slice(indptr[vocabulary[term]], indptr[vocabulary[term] + 1])
            rows.append(np.asarray(engine.scores["indices"])[part])
            values.append(np.asarray(engine.scores["data"])[part])
            starts.append(starts[-1] + len(rows[-1]))
        empty = np.zeros(0)
        return cls(
            np.array(terms),
            np.asarray(starts, dtype=np.int64),
            np.concatenate([*rows, empty]).astype(np.int32),
            np.concatenate([*values, empty]).astype(np.float32),
            index.size,
        )
