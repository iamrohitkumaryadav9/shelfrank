"""Out-of-fold rankings, so that no query is ever ranked by a model that saw its labels."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from shelfrank.ltr.dataset import RankingData
from shelfrank.ltr.model import LtrConfig, Ranker, to_run
from shelfrank.text.near_duplicates import near_duplicate_groups


@dataclass
class OutOfFold:
    """Rankings of every query and what was learned in each fold."""

    run: dict[int, list[int]]
    trees: list[list[int]] = field(default_factory=list)
    gain: list[dict[str, float]] = field(default_factory=list)


def grouped_folds(
    texts: Mapping[int, str], n_folds: int, seed: int, jaccard: float
) -> list[list[int]]:
    """Folds that keep near-duplicate queries together."""
    groups = [sorted(g) for g in near_duplicate_groups(dict(texts), jaccard)]
    inside = {q for g in groups for q in g}
    units = groups + [[q] for q in sorted(set(texts) - inside)]
    random.Random(seed).shuffle(units)
    units.sort(key=len, reverse=True)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for unit in units:
        min(folds, key=len).extend(unit)
    return [sorted(f) for f in folds]


def out_of_fold(
    data: RankingData,
    folds: Sequence[Sequence[int]],
    config: LtrConfig,
    train_limit: int | None = None,
) -> OutOfFold:
    """Train on the other folds, rank every candidate of this one.

    ``train_limit`` trains on a fixed random subset, for learning curves.
    """
    everything = [q for fold in folds for q in fold]
    if len(everything) != len(set(everything)):
        raise ValueError("folds overlap")
    out = OutOfFold({})
    for held in folds:
        train = sorted(set(everything) - set(held))
        if train_limit is not None:
            train = sorted(random.Random(config.seeds[0]).sample(train, train_limit))
        ranker = Ranker(config).fit(data, train)
        rows = data.rows(held)
        out.run |= to_run(data, rows, ranker.scores(data, rows))
        out.trees.append(ranker.trees)
        out.gain.append(ranker.gain_importance())
    return out
