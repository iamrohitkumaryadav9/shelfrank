"""Interval of a micro-F1 over queries."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from shelfrank.qu.class_data import SEED
from shelfrank.qu.slot_lexicon import span_scores


def f1_interval(truth: Sequence[Any], predicted: Sequence[Any]) -> dict[str, float]:
    """Bootstrap interval of the micro-F1, resampling queries."""
    counts = []
    for t, p in zip(truth, predicted, strict=True):
        s = span_scores([t], [p])["all"]
        hits = 0.0 if not s["support"] or not s["predicted"] else s["recall"] * s["support"]
        counts.append((hits, s["predicted"], s["support"]))
    table = np.asarray(counts, dtype=float)
    picks = np.random.default_rng(SEED).integers(0, len(table), (10_000, len(table)))
    sums = table[picks].sum(axis=1)
    scores = 2 * sums[:, 0] / np.maximum(sums[:, 1] + sums[:, 2], 1)
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}
