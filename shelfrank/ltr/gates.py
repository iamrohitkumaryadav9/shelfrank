"""Work that may not start yet, refused in code."""

import json
from collections.abc import Sequence
from pathlib import Path

from shelfrank.config import REPO_ROOT
from shelfrank.eval.access import ProtocolError

MAX_ROOM_DISAGREEMENTS = 2


def slot_blockers(root: Path = REPO_ROOT) -> list[str]:
    """Why G4 and the slot seed batch must wait. Empty when they may proceed."""
    summary = root / "results" / "slots_round_a.json"
    if not summary.exists():
        return ["Round A has no summary"]
    found = json.loads(summary.read_text(encoding="utf-8"))["room_rule"]["disagreements"]
    if found > MAX_ROOM_DISAGREEMENTS:
        return [f"the room rule failed Round A with {found} disagreements"]
    return []


def allowed_groups(groups: Sequence[str], root: Path = REPO_ROOT) -> list[str]:
    """Return the groups, or refuse when one of them is G4 and its conditions are not met."""
    if "G4" in groups:
        blockers = slot_blockers(root)
        if not (root / "configs" / "slot_student.yaml").exists():
            blockers.append("no slot student has been selected")
        if blockers:
            raise ProtocolError("G4 is disabled: " + "; ".join(blockers))
    return list(groups)
