"""Work that may not start yet, refused in code."""

from collections.abc import Sequence
from pathlib import Path

from shelfrank.config import REPO_ROOT
from shelfrank.eval.access import ProtocolError

SLOT_STUDENT = "configs/slot_student.yaml"


def allowed_groups(groups: Sequence[str], root: Path = REPO_ROOT) -> list[str]:
    """Return the groups, or refuse when one of them is G4 and no slot student is selected."""
    if "G4" in groups and not (root / SLOT_STUDENT).exists():
        raise ProtocolError("G4 is disabled: no slot student has been selected")
    return list(groups)
