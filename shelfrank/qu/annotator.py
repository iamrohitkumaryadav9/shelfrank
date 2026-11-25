"""What the human annotator sees: a guide at every slot prompt, and practice with feedback.

Guide examples come from the labelling rules and their examples. Practice queries were invented and
are checked against WANDS and the ESCI files by the leakage tests. Their answers follow from the
rules; the parts that code can derive (room, vocabulary, one slot per word) are tested.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from shelfrank.qu.guidelines import SLOTS

Slots = dict[str, list[str]]
PASS_MARK = 13


@dataclass(frozen=True)
class SlotGuide:
    """One-line rule, with phrases that do and do not fill the slot."""

    rule: str
    yes: tuple[tuple[str, str], ...]
    no: tuple[tuple[str, str], ...]


SLOT_GUIDE: dict[str, SlotGuide] = {
    "color": SlotGuide(
        "A colour, colour family or shade. Copy the whole phrase. Plain gold, silver, brass, "
        "bronze and copper are colors.",
        (("dark gray linen headboard queen", "dark gray"), ("silver candle snuffer", "silver")),
        (
            ("walnut letter tray", "walnut is wood: material_finish"),
            ("gold plated picture ledge", "'gold plated' names a finish: material_finish"),
        ),
    ),
    "material_finish": SlotGuide(
        "What it is made of or finished in: fabric, stone, metal, any wood, metal finishes. A "
        "metal word counts when the query says finish, plated or metal, or names the finish.",
        (("brushed nickel towel ring", "brushed nickel"), ("walnut letter tray", "walnut")),
        (
            ("silver candle snuffer", "silver alone is a color"),
            ("gold mirror", "gold alone is a color"),
        ),
    ),
    "style": SlotGuide(
        "A NAMED DESIGN STYLE only. Not product words, features, patterns, shapes, sizes or "
        "subjects.",
        (("farmhouse plant stand for sunroom", "farmhouse"), ("mid century", "mid century")),
        (
            ("striped balcony cushion", "striped is a pattern: no slot"),
            ("dining chairs set of 6", "product words and quantities: no slot"),
        ),
    ),
    "room": SlotGuide(
        "Where the product goes, indoors or outdoors. Not when the room word is inside a product "
        "category name.",
        (("farmhouse plant stand for sunroom", "sunroom"), ("bathroom rug", "bathroom")),
        (
            ("dining chairs set of 6", "'dining' is part of the product's name"),
            ("bathroom vanity", "'bathroom' is part of the product's name"),
        ),
    ),
}


@dataclass(frozen=True)
class PracticeItem:
    """An invented query, its answer and the rule that gives the answer."""

    query: str
    answer: Slots
    why: str


def _item(query: str, why: str, **slots: list[str]) -> PracticeItem:
    return PracticeItem(query, {slot: slots.get(slot, []) for slot in SLOTS}, why)


PRACTICE: tuple[PracticeItem, ...] = (
    _item(
        "dark green velvet doorstop",
        "Copy the whole colour phrase. Velvet is a fabric, so material_finish.",
        color=["dark green"],
        material_finish=["velvet"],
    ),
    _item(
        "sturdy foldable shoe bench",
        "'sturdy' and 'foldable' describe the product. They are not design styles, so no slot.",
    ),
    _item(
        "scandinavian oak wine cubby",
        "'scandinavian' is a named design style. Every wood species is material_finish.",
        style=["scandinavian"],
        material_finish=["oak"],
    ),
    _item(
        "copper mail sorter",
        "A plain metal word with no finish, plated or metal is a color.",
        color=["copper"],
    ),
    _item(
        "copper plated napkin caddy",
        "'plated' makes it material_finish. A word belongs to one slot, so color stays empty.",
        material_finish=["copper plated"],
    ),
    _item(
        "kitchen island on casters with towel bar, espresso",
        "'kitchen island' is a category name, so no room. Espresso is a shade.",
        color=["espresso"],
    ),
    _item(
        "bathroom step ladder bamboo",
        "'bathroom step ladder' is not a category, so bathroom says where it goes: room. Bamboo is "
        "what it is made of.",
        material_finish=["bamboo"],
        room=["bathroom"],
    ),
    _item(
        "geometric round trinket dish",
        "geometric is a pattern and round is a shape. Neither is a style, so no slot.",
    ),
    _item(
        "matte black key hook rail",
        "'matte black' is a named finish, so the whole phrase is material_finish. Color is empty "
        "because a word belongs to one slot.",
        material_finish=["matte black"],
    ),
    _item(
        "coastal bedroom tray",
        "'coastal' is a named style. Bedroom says where it goes.",
        style=["coastal"],
        room=["bedroom"],
    ),
    _item(
        "elegant extra wide cutlery organizer",
        "'elegant' is praise, not a named design style. 'extra wide' is a size. No slot.",
    ),
    _item(
        "dining table protector cover white vinyl",
        "'dining table' is a category name, so no room. White is a color. Vinyl is a " "material.",
        color=["white"],
        material_finish=["vinyl"],
    ),
    _item(
        "patio towel hook brushed nickel",
        "'patio towel hook' is not a category, so patio is room. 'brushed nickel' is a finish.",
        material_finish=["brushed nickel"],
        room=["patio"],
    ),
    _item(
        "mid century cherry magazine rack",
        "'mid century' is a named style. Cherry is a wood species, so material_finish and not "
        "color.",
        style=["mid century"],
        material_finish=["cherry"],
    ),
    _item(
        "plaid pet bowl mat 2 pack",
        "plaid is a pattern, 'pet bowl mat' is the product and '2 pack' is a quantity. No slot.",
    ),
)


def practice_id() -> str:
    """Fingerprint of the practice set, so that a pass on other questions does not count."""
    body = json.dumps([(p.query, p.answer) for p in PRACTICE], sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Attempt:
    """One finished practice run."""

    correct: int
    total: int
    version: str
    practice: str
    time: str

    def unlocks(self, version: str) -> bool:
        """Whether it counts for the current rules and questions."""
        current = self.version == version and self.practice == practice_id()
        return current and self.total == len(PRACTICE) and self.correct >= PASS_MARK


def read_attempts(path: Path) -> list[Attempt]:
    """Every finished practice run, passes and failures."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Attempt(**json.loads(line)) for line in lines if line.strip()]


def unlocked(path: Path, version: str) -> bool:
    """Whether labelling may start."""
    return any(attempt.unlocks(version) for attempt in read_attempts(path))
