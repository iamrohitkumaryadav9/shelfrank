"""LambdaMART and the pointwise baseline, with early stopping on queries held out of training."""

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shelfrank.ltr.dataset import RankingData, group_sizes

LAMBDAMART, POINTWISE = "lambdamart", "pointwise"


@dataclass(frozen=True)
class LtrConfig:
    """One training configuration."""

    columns: tuple[str, ...]
    variant: str
    objective: str = LAMBDAMART
    label_gain: tuple[int, ...] = (0, 1, 2)
    params: dict[str, Any] = field(default_factory=dict)
    rounds: int = 600
    patience: int = 60
    valid_share: float = 0.2
    seeds: tuple[int, ...] = (20260, 20261, 20262)

    def describe(self) -> dict[str, Any]:
        """Everything needed to repeat it, for the log."""
        return {
            "columns": list(self.columns),
            "groups": sorted({c.split("_")[0].upper() for c in self.columns}),
            "variant": self.variant,
            "objective": self.objective,
            "label_gain": list(self.label_gain),
            "params": self.params,
            "rounds": self.rounds,
            "patience": self.patience,
            "valid_share": self.valid_share,
            "seeds": list(self.seeds),
        }


def split_queries(query_ids: Sequence[int], share: float, seed: int) -> tuple[list[int], list[int]]:
    """Training and validation queries. Validation picks the number of trees and nothing else."""
    ids = sorted(query_ids)
    random.Random(seed).shuffle(ids)
    cut = max(1, round(len(ids) * share))
    return sorted(ids[cut:]), sorted(ids[:cut])


def lightgbm_params(config: LtrConfig, seed: int) -> dict[str, Any]:
    """Parameters for one seed. Single-machine determinism is forced."""
    base: dict[str, Any] = {
        **config.params,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 4,
    }
    if config.objective == POINTWISE:
        return {**base, "objective": "regression", "metric": "l2"}
    if config.objective != LAMBDAMART:
        raise ValueError(f"unknown objective {config.objective}")
    return {
        **base,
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "label_gain": list(config.label_gain),
        "lambdarank_truncation_level": 20,
    }


class Ranker:
    """Average of one model per seed, which steadies training on few queries."""

    def __init__(self, config: LtrConfig) -> None:
        """Keep the configuration."""
        self.config = config
        self.boosters: list[Any] = []

    def _matrix(
        self,
        data: RankingData,
        queries: Sequence[int],
        reference: Any = None,  # noqa: ANN401 - LightGBM is imported lazily
    ) -> Any:  # noqa: ANN401
        import lightgbm as lgb

        rows = data.rows(queries, self.config.variant)
        ranking = self.config.objective == LAMBDAMART
        return lgb.Dataset(
            data.features.iloc[rows][list(self.config.columns)],
            label=data.labels(rows),
            group=group_sizes(data.query_id[rows]) if ranking else None,
            reference=reference,
        )

    def fit(self, data: RankingData, query_ids: Sequence[int]) -> "Ranker":
        """Train on the given queries only."""
        import lightgbm as lgb

        self.boosters = []
        for seed in self.config.seeds:
            train, valid = split_queries(query_ids, self.config.valid_share, seed)
            fitted = self._matrix(data, train)
            self.boosters.append(
                lgb.train(
                    lightgbm_params(self.config, seed),
                    fitted,
                    num_boost_round=self.config.rounds,
                    valid_sets=[self._matrix(data, valid, fitted)],
                    callbacks=[lgb.early_stopping(self.config.patience, verbose=False)],
                )
            )
        return self

    def scores(self, data: RankingData, rows: np.ndarray) -> np.ndarray:
        """Mean score per row."""
        table = data.features.iloc[rows][list(self.config.columns)]
        parts = [b.predict(table, num_iteration=b.best_iteration) for b in self.boosters]
        return np.mean(parts, axis=0)

    def gain_importance(self) -> dict[str, float]:
        """Share of total split gain per feature."""
        parts = [b.feature_importance("gain", iteration=b.best_iteration) for b in self.boosters]
        total = np.sum(parts, axis=0)
        share = total / total.sum() if total.sum() else total
        return dict(zip(self.config.columns, share.tolist(), strict=True))

    @property
    def trees(self) -> list[int]:
        """Trees chosen by early stopping, per seed."""
        return [int(b.best_iteration) for b in self.boosters]


def to_run(data: RankingData, rows: np.ndarray, scores: np.ndarray) -> dict[int, list[int]]:
    """Rankings from scores. Ties go to the lower product id."""
    run: dict[int, list[tuple[float, int]]] = {}
    for row, score in zip(rows, scores, strict=True):
        pair = (-float(score), int(data.product_id[row]))
        run.setdefault(int(data.query_id[row]), []).append(pair)
    return {q: [p for _, p in sorted(items)] for q, items in run.items()}
