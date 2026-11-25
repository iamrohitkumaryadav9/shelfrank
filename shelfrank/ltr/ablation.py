"""Which configurations are compared."""

from collections.abc import Sequence
from dataclasses import replace

from shelfrank.features.groups import columns_of
from shelfrank.ltr.dataset import VARIANTS
from shelfrank.ltr.model import CV, POINTWISE, LtrConfig


def variants(base: LtrConfig) -> dict[str, LtrConfig]:
    """The two treatments of unjudged candidates."""
    return {f"full_{v}": replace(base, variant=v) for v in VARIANTS}


SMALLER = {
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.5,
    "lambda_l2": 10.0,
}


def training_choices(base: LtrConfig) -> dict[str, LtrConfig]:
    """Stopping rule and model size, crossed. The base is one of the four."""
    smaller = {**base.params, **SMALLER}
    return {
        "stop_holdout_smaller": replace(base, params=smaller),
        "stop_cv": replace(base, stopping=CV),
        "stop_cv_smaller": replace(base, stopping=CV, params=smaller),
    }


def alternatives(base: LtrConfig) -> dict[str, LtrConfig]:
    """Changes of method around the full model."""
    return {
        "pointwise": replace(base, objective=POINTWISE),
        "label_gain_013": replace(base, label_gain=(0, 1, 3)),
    }


def ablations(
    base: LtrConfig, columns: Sequence[str], groups: Sequence[str]
) -> dict[str, LtrConfig]:
    """Leave one group out, and add groups one at a time in order."""
    out = {}
    for group in groups:
        rest = [g for g in groups if g != group]
        out[f"without_{group}"] = replace(base, columns=tuple(columns_of(columns, rest)))
    for n in range(1, len(groups)):
        name = "only_" + "_".join(groups[:n])
        out[name] = replace(base, columns=tuple(columns_of(columns, groups[:n])))
    return out
