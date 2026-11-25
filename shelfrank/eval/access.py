"""Stop development code from touching the primary queries before the protocol is frozen."""

from pathlib import Path

from shelfrank.config import REPO_ROOT
from shelfrank.eval.splits import Folds

FROZEN = REPO_ROOT / "configs" / "frozen.yaml"


class ProtocolError(RuntimeError):
    """The requested queries may not be used yet."""


def queries_for(split: str, folds: Folds, frozen: Path = FROZEN) -> list[int]:
    """Return the query ids of a split, refusing anything but dev until frozen.yaml exists.

    A check in code, because a mistyped flag should not be able to spend the test set.
    """
    if split == "dev":
        return folds.dev
    if split not in ("primary", "all"):
        raise ValueError(f"unknown split: {split}")
    if not frozen.exists():
        raise ProtocolError(f"{split} queries are locked until {frozen.name} is committed")
    return folds.primary if split == "primary" else folds.queries(*range(folds.n_folds))
