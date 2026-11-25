"""Bootstrap over queries, the unit that was sampled."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A mean with a 95% percentile interval."""

    mean: float
    low: float
    high: float
    n: int

    def excludes_zero(self) -> bool:
        """Whether the interval lies on one side of 0."""
        return self.low > 0 or self.high < 0


def bootstrap_mean(
    values: Sequence[float], n_resamples: int = 10_000, seed: int = 20260
) -> Interval:
    """Percentile interval for a mean, resampling queries with replacement."""
    data = np.asarray(values, dtype=float)
    if data.size == 0 or np.isnan(data).any():
        raise ValueError("need at least one value and no NaN")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, 1000):  # chunked so memory stays flat
        stop = min(start + 1000, n_resamples)
        picks = rng.integers(0, data.size, size=(stop - start, data.size))
        means[start:stop] = data[picks].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return Interval(float(data.mean()), float(low), float(high), int(data.size))


def paired_difference(
    system: Sequence[float],
    baseline: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 20260,
) -> Interval:
    """Interval for the mean per-query difference.

    Pairing removes the large variation in difficulty between queries, which would otherwise swamp
    the difference between two systems.
    """
    if len(system) != len(baseline):
        raise ValueError("paired samples must have equal length")
    diff = np.asarray(system, dtype=float) - np.asarray(baseline, dtype=float)
    return bootstrap_mean(diff, n_resamples, seed)
