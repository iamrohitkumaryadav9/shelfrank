"""The pool rule of docs/PROTOCOL.md section 7, as code."""

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from shelfrank.eval.harness import PerQuery
from shelfrank.stats.bootstrap import Interval, paired_difference

GAIN, LOSS, NONE = "gain", "loss", "no clear difference"
UNCLEAR = "direction depends on unjudged handling"


def verdict(standard: Interval, condensed: Interval) -> str:
    """Claim a gain only when both intervals are above 0.

    "No clear difference" is reserved for the case where neither interval excludes 0. A difference
    that shows under one treatment of unjudged products only is never called a gain.
    """
    if standard.low > 0 and condensed.low > 0:
        return GAIN
    if standard.high < 0 and condensed.high < 0:
        return LOSS
    if not standard.excludes_zero() and not condensed.excludes_zero():
        return NONE
    return UNCLEAR


def compare(system: PerQuery, baseline: PerQuery, **bootstrap: int) -> dict[str, Any]:
    """Paired differences under both metrics, and the verdict."""
    if set(system) != set(baseline):
        raise ValueError("both systems must cover the same queries")
    ids = sorted(baseline)
    out: dict[str, Any] = {}
    intervals = []
    for metric in ("ndcg10", "condensed_ndcg10"):
        ours = [float(system[q][metric] or 0) for q in ids]
        theirs = [float(baseline[q][metric] or 0) for q in ids]
        intervals.append(paired_difference(ours, theirs, **bootstrap))
        out[metric] = asdict(intervals[-1])
    return out | {"verdict": verdict(*intervals)}


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values for a declared family."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    out: dict[str, float] = {}
    worst = 0.0
    for i, (name, p) in enumerate(ordered):
        worst = max(worst, min(1.0, (len(ordered) - i) * p))
        out[name] = worst
    return out
