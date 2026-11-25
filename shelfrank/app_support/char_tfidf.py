"""The class student without scikit-learn: character n-gram TF-IDF and one linear row per class."""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from shelfrank.qu.class_student import LinearClassifier, clean

_SPACES = re.compile(r"\s\s+")


def char_wb_ngrams(text: str, low: int, high: int) -> list[str]:
    """Character n-grams inside word boundaries, in the manner of scikit-learn's char_wb."""
    out = []
    for word in _SPACES.sub(" ", text).split():
        padded = f" {word} "
        for n in range(low, high + 1):
            offset = 0
            out.append(padded[offset : offset + n])
            while offset + n < len(padded):
                offset += 1
                out.append(padded[offset : offset + n])
            if offset == 0:  # a short word is counted once
                break
    return out


class PlainClassifier:
    """Same probabilities as ``LinearClassifier``, from plain arrays."""

    def __init__(self, body: Mapping[str, Any]) -> None:
        """Keep vocabulary, weights and settings."""
        self.vocabulary: dict[str, int] = {t: i for i, t in enumerate(body["terms"])}
        self.idf = np.asarray(body["idf"], dtype=np.float32)
        self.weights = np.asarray(body["weights"], dtype=np.float32)
        self.bias = np.asarray(body["bias"], dtype=np.float32)
        self.classes = [str(c) for c in body["classes"]]
        self.low, self.high = (int(n) for n in body["ngrams"])

    def probabilities(self, queries: Sequence[str]) -> np.ndarray:
        """Queries by classes, rows summing to 1."""
        out = np.empty((len(queries), len(self.bias)))
        for i, query in enumerate(queries):
            grams = char_wb_ngrams(clean(query), self.low, self.high)
            counts = Counter(g for g in grams if g in self.vocabulary)
            columns = np.array(sorted(self.vocabulary[g] for g in counts), dtype=int)
            by_column = {self.vocabulary[g]: n for g, n in counts.items()}
            values = np.array([1 + np.log(by_column[c]) for c in columns], dtype=np.float32)
            values = values * self.idf[columns]
            norm = float(np.sqrt((values**2).sum()))
            score = self.bias + (values / norm @ self.weights[columns] if norm else 0)
            out[i] = 1 / (1 + np.exp(-score.astype(float)))
        return out / out.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        """Half precision: the file is committed, and the ranking of classes does not need more."""
        terms = np.array(sorted(self.vocabulary, key=self.vocabulary.__getitem__))
        np.savez_compressed(
            path,
            terms=terms,
            idf=self.idf,
            weights=self.weights.astype(np.float16),
            bias=self.bias,
            classes=np.array(self.classes),
            ngrams=np.array([self.low, self.high]),
        )

    @classmethod
    def load(cls: type["PlainClassifier"], path: Path) -> "PlainClassifier":
        """Read a saved file."""
        with np.load(path, allow_pickle=False) as body:
            return cls({name: body[name] for name in body.files})

    @classmethod
    def from_student(cls: type["PlainClassifier"], student: LinearClassifier) -> "PlainClassifier":
        """Copy a trained student."""
        vectorizer = student.vectorizer
        terms = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.__getitem__)
        return cls(
            {
                "terms": terms,
                "idf": vectorizer.idf_,
                "weights": student.weights,
                "bias": student.bias,
                "classes": student.classes,
                "ngrams": vectorizer.ngram_range,
            }
        )
