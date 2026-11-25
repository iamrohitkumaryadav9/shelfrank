"""Which training option is used, by the rule in configs/ltr.yaml, and how often dev may be run."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from shelfrank.eval.access import ProtocolError
from shelfrank.ltr.dataset import JUDGED_ONLY
from shelfrank.ltr.model import LtrConfig

INTERIM, FINAL = "interim", "final"


def selection_score(system: Mapping[str, Any]) -> float:
    """Mean of standard and condensed nDCG@10."""
    return (float(system["ndcg10"]["mean"]) + float(system["condensed_ndcg10"]["mean"])) / 2


def select(systems: Mapping[str, Any], options: Mapping[str, LtrConfig]) -> str:
    """Name of the chosen option."""

    def key(name: str) -> tuple[float, bool]:
        return selection_score(systems[name]), options[name].variant == JUDGED_ONLY

    return max(options, key=key)


def output_path(stage: str, groups: list[str], bug_fix: str, root: Path) -> Path:
    """Where a dev comparison is written. Each stage runs once, except for a logged bug fix."""
    if (stage == FINAL) != ("G4" in groups):
        raise ProtocolError("the interim run has G4 off and the final run has it on")
    path = root / "results" / f"ltr_{stage}.json"
    if path.exists() and not bug_fix:
        raise ProtocolError(f"{path.name} exists. A rerun needs --bug-fix with the reason.")
    if (root / "results" / f"ltr_{FINAL}.json").exists() and stage == INTERIM:
        raise ProtocolError("the final dev comparison has been run")
    return path


def bug_fixes(path: Path, reason: str) -> list[str]:
    """Reasons of every rerun so far, including this one."""
    earlier = []
    if path.exists():
        earlier = json.loads(path.read_text(encoding="utf-8"))["bug_fixes"]
    return [*earlier, reason] if reason else earlier
