"""Small product-class classifiers trained without any WANDS query.

Two sources of supervision. Product names carry the catalogue's own class, so they are free and
cover every class, but they are not queries. ESCI queries labelled by the teacher look like
queries, but cover fewer classes. An arm is one source or both.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from shelfrank.data.taxonomy import OTHER
from shelfrank.data.wands import split_classes
from shelfrank.text.normalize import tokenize


@dataclass
class Examples:
    """Weighted training texts."""

    texts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def add(self, text: str, label: str, weight: float) -> None:
        """Append one example."""
        self.texts.append(text)
        self.labels.append(label)
        self.weights.append(weight)

    def scaled(self, total: float) -> "Examples":
        """Rescale so that the weights sum to ``total``."""
        factor = total / sum(self.weights)
        return Examples(list(self.texts), list(self.labels), [w * factor for w in self.weights])

    def __add__(self, other: "Examples") -> "Examples":
        """Concatenate."""
        return Examples(
            self.texts + other.texts, self.labels + other.labels, self.weights + other.weights
        )


def clean(text: object) -> str:
    """Normalise like the rest of the system, so plural and spelling variants share n-grams."""
    return " ".join(tokenize(text))


def name_examples(
    products: pd.DataFrame, taxonomy: Sequence[str], tail: int | None = None
) -> Examples:
    """Product names with their classes.

    A product with several classes gives one example per class and one unit of weight in all. A
    product whose classes are all too small for the taxonomy teaches "other". ``tail`` adds the last
    words of the name as a second example: names end with the product type, and the result is as
    short as a query.
    """
    known = set(taxonomy)
    out = Examples()
    for name, cell in zip(products["product_name"], products["product_class"], strict=True):
        classes = list(dict.fromkeys(split_classes(cell)))
        labels = [c for c in classes if c in known] or ([OTHER] if classes else [])
        words = tokenize(name)
        texts = {" ".join(words)}
        if tail:
            texts.add(" ".join(words[-tail:]))
        for label in labels:
            for text in sorted(texts - {""}):
                out.add(text, label, 1 / (len(labels) * len(texts)))
    return out


def query_examples(queries: Sequence[str], labels: Sequence[str | None]) -> Examples:
    """Teacher-labelled queries. A label outside the taxonomy is dropped, not guessed."""
    out = Examples()
    for query, label in zip(queries, labels, strict=True):
        if label is not None:
            out.add(clean(query), label, 1.0)
    return out


def split_seed(n: int, held_share: float, seed: int) -> tuple[list[int], list[int]]:
    """Fixed random split of the seed queries into training and held-out positions."""
    order = np.random.default_rng(seed).permutation(n)
    cut = round(n * held_share)
    return sorted(int(i) for i in order[cut:]), sorted(int(i) for i in order[:cut])


def mix(names: Examples, queries: Examples, query_share: float) -> Examples:
    """Both sources, with the queries holding ``query_share`` of the total weight.

    Unscaled, 2,000 queries would disappear beside 37,000 names.
    """
    total = sum(names.weights)
    return names.scaled(total * (1 - query_share)) + queries.scaled(total * query_share)


class LinearClassifier:
    """Character n-gram TF-IDF with logistic regression, one model per class."""

    def __init__(
        self,
        ngrams: tuple[int, int] = (2, 5),
        alpha: float = 1e-6,
        max_features: int = 120_000,
        seed: int = 20260,
    ) -> None:
        """Set hyperparameters. scikit-learn is imported here because the core package lacks it."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngrams,
            sublinear_tf=True,
            max_features=max_features,
            dtype=np.float32,
        )
        self.model = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=12, tol=None, average=True, random_state=seed
        )
        self.model.n_jobs = -1

    def fit(self, examples: Examples) -> "LinearClassifier":
        """Train."""
        matrix = self.vectorizer.fit_transform(examples.texts)
        weights = np.asarray(examples.weights)
        self.model.fit(matrix, examples.labels, sample_weight=weights / weights.mean())
        coef, bias = self.model.coef_, self.model.intercept_
        if len(self.model.classes_) == 2:  # scikit-learn keeps one row for two classes
            coef, bias = np.vstack([-coef, coef]), np.concatenate([-bias, bias])
        # One row per n-gram: a short query then reads a few rows and not the whole matrix, which
        # made scikit-learn's own predict_proba take 56 ms.
        self.weights = np.ascontiguousarray(coef.T, dtype=np.float32)
        self.bias = bias.astype(np.float32)
        return self

    @property
    def classes(self) -> list[str]:
        """Labels in the order of the probability columns."""
        return [str(c) for c in self.model.classes_]

    def probabilities(self, queries: Sequence[str]) -> np.ndarray:
        """Queries by classes, rows summing to 1. Same values as scikit-learn's predict_proba."""
        matrix = self.vectorizer.transform([clean(q) for q in queries]).tocsr()
        out = np.empty((len(queries), len(self.bias)))
        for i in range(len(queries)):
            part = slice(matrix.indptr[i], matrix.indptr[i + 1])
            score = matrix.data[part] @ self.weights[matrix.indices[part]] + self.bias
            out[i] = 1 / (1 + np.exp(-score.astype(float)))
        return out / out.sum(axis=1, keepdims=True)


def ranked_labels(probabilities: np.ndarray, classes: Sequence[str], k: int) -> list[list[str]]:
    """Best ``k`` labels per row; ties go to the earlier column, so the result is reproducible."""
    order = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]
    return [[classes[i] for i in row] for row in order]


def to_reference(value: object, taxonomy: Sequence[str]) -> str:
    """Reference class of a query: a class outside the taxonomy is "other"."""
    return value if isinstance(value, str) and value in set(taxonomy) else OTHER


def macro_f1(truth: Sequence[str], predicted: Sequence[str], min_support: int) -> tuple[float, int]:
    """Mean F1 over classes with enough reference queries, and how many classes that is."""
    labels = sorted({t for t in truth if truth.count(t) >= min_support})
    scores = []
    for label in labels:
        hits = sum(t == p == label for t, p in zip(truth, predicted, strict=True))
        made, wanted = predicted.count(label), truth.count(label)
        scores.append(2 * hits / (made + wanted))
    return (sum(scores) / len(scores) if scores else float("nan")), len(labels)


def evaluate_classes(
    truth: Sequence[str],
    ranked: Sequence[Sequence[str]],
    departments: Mapping[str, str],
    seeded: set[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, top-3, department accuracy and macro-F1, with the n of each.

    ``seeded`` splits accuracy into classes with and without any labelled query.
    """
    first = [row[0] for row in ranked]
    right = [t == p for t, p in zip(truth, first, strict=True)]

    def department(label: str) -> str:
        return departments.get(label, OTHER)

    f1, f1_classes = macro_f1(list(truth), first, 5)
    out: dict[str, Any] = {
        "n": len(truth),
        "accuracy": float(np.mean(right)),
        "top3": float(np.mean([t in row[:3] for t, row in zip(truth, ranked, strict=True)])),
        "department": float(
            np.mean([department(t) == department(p) for t, p in zip(truth, first, strict=True)])
        ),
        "macro_f1": f1,
        "macro_f1_classes": f1_classes,
    }
    if seeded is not None:
        for name, inside in (("seeded", True), ("names_only", False)):
            part = [r for t, r in zip(truth, right, strict=True) if (t in seeded) is inside]
            out[name] = {"n": len(part), "accuracy": float(np.mean(part)) if part else float("nan")}
    return out
