"""What G4 and the recommendations need from the slot lexicon."""

from pathlib import Path

from shelfrank.config import REPO_ROOT
from shelfrank.qu.slot_model import load_lexicon_data
from shelfrank.qu.slot_vocab import Term


def term_slots(root: Path = REPO_ROOT) -> dict[Term, str]:
    """Slot of every known color and material_finish value."""
    lexicon = load_lexicon_data(root)
    return {
        tuple(name.split()): slot for slot in ("color", "material_finish") for name in lexicon[slot]
    }
