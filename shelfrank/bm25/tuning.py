"""Coordinate search over field weights, with a held-out estimate of the tuned score."""

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

Weights = dict[str, float]
Objective = Callable[[Weights], float]


@dataclass
class TuningResult:
    """The chosen weights and every configuration that was tried."""

    weights: Weights
    score: float
    log: list[dict[str, object]] = field(default_factory=list)


def coordinate_search(
    fields: Sequence[str],
    anchor: str,
    grid: Sequence[float],
    objective: Objective,
    min_gain: float,
) -> TuningResult:
    """Change one weight at a time until no change gains at least ``min_gain``.

    Chosen over a full grid for two reasons: each accepted step can be read and explained, and far
    fewer configurations are compared, which matters when the dev fold has only 96 queries.
    """
    weights: Weights = {name: 1.0 for name in fields}
    result = TuningResult(weights, objective(weights))
    result.log.append({"weights": dict(weights), "score": result.score})
    seen = {tuple(weights.values())}
    improved = True
    while improved:
        improved = False
        for name in fields:
            if name == anchor:
                continue
            best_value, best_score = weights[name], result.score
            for value in grid:
                trial = {**weights, name: value}
                key = tuple(trial.values())
                if key in seen:
                    continue
                seen.add(key)
                score = objective(trial)
                result.log.append({"weights": trial, "score": score})
                if score > best_score:
                    best_value, best_score = value, score
            if best_score - result.score >= min_gain:
                weights[name], result.score = best_value, best_score
                improved = True
    return result


def inner_folds(query_ids: Sequence[int], n_folds: int, seed: int) -> list[list[int]]:
    """Split the dev queries again, deterministically."""
    shuffled = sorted(query_ids)
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[i::n_folds]) for i in range(n_folds)]


def held_out_scores(
    query_ids: Sequence[int],
    n_folds: int,
    seed: int,
    tune: Callable[[list[int]], Weights],
    score: Callable[[Weights, list[int]], Mapping[int, float]],
) -> tuple[dict[int, float], list[Weights]]:
    """Tune on part of the dev fold and score the rest, so tuned and untuned compare fairly."""
    out: dict[int, float] = {}
    chosen = []
    for held in inner_folds(query_ids, n_folds, seed):
        rest = sorted(set(query_ids) - set(held))
        weights = tune(rest)
        chosen.append(weights)
        out.update(score(weights, held))
    return dict(sorted(out.items())), chosen
