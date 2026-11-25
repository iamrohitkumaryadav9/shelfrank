"""Labelling rules and examples, shared by the teacher prompt and the human annotator.

GOVERNING PRINCIPLE: Every slot value must exist in the catalogue's attribute vocabulary. G4
matching is key-agnostic: a COLOUR or MATERIAL_FINISH value is looked up across the union of colour,
finish and material keys, so a match never depends on which key the catalogue happened to store it
under.

docs/slot_evidence.md shows the vocabulary. One source for both labellers: agreement is meaningless
if they follow different rules. The examples were invented for this file. Examples and the phrases
quoted in the rules are checked against WANDS by the leakage tests.
"""

import re

SLOTS = ("color", "material_finish", "style", "room")

# Recorded in the cache entry of every label. "-rc" marks rules that are not final, and paid slot
# labelling is refused while it is there.
SLOT_VERSION = "slots-v1"
CLASS_VERSION = "class-v1"

SLOT_RULES = (
    "Copy words exactly as they appear in the query, including misspellings. Never add a word.",
    "Leave a slot empty when the query does not state it. Do not infer it from the product type.",
    "color: a colour, colour family or shade ('navy', 'light gray', 'espresso', 'multicolor'). "
    "Copy the whole phrase.",
    "material_finish: what the product is made of or finished in. This includes fabrics, stone, "
    "metals, every wood species ('walnut', 'oak', 'cherry'), and metal finishes ('chrome', "
    "'nickel', 'brushed nickel', 'matte black', 'oil rubbed bronze').",
    "gold, silver, brass, bronze and copper used as plain colour words are color ('gold mirror': "
    "color gold). They are material_finish when the query says finish, plated or metal, or names "
    "the finish ('gold plated picture ledge': material_finish gold plated; 'antique brass': "
    "material_finish).",
    "style: a named design style ('modern', 'farmhouse', 'mid century', 'boho', 'rustic', "
    "'coastal', 'vintage'). Patterns (floral, striped, geometric, plaid, solid), "
    "shapes, sizes and subjects are not styles and have no slot.",
    "room: where the product goes, indoors or outdoors ('bedroom', 'bathroom', 'patio', 'garden', "
    "'balcony', 'deck'). It is not a slot when the room word is inside a product category name "
    "('dining table', 'kitchen island', 'bathroom vanity'). Otherwise it is ('bathroom rug', "
    "'patio umbrella', 'bedroom lamp').",
    "A word belongs to one slot.",
    "Brand names, collection names, quantities and dimensions belong to no slot.",
)


def _slots(**values: list[str]) -> dict[str, list[str]]:
    return {slot: values.get(slot, []) for slot in SLOTS}


SLOT_EXAMPLES: tuple[tuple[str, dict[str, list[str]]], ...] = (
    ("dark gray linen headboard queen", _slots(color=["dark gray"], material_finish=["linen"])),
    ("farmhouse plant stand for sunroom", _slots(style=["farmhouse"], room=["sunroom"])),
    ("brushed nickel towel ring", _slots(material_finish=["brushed nickel"])),
    ("silver candle snuffer", _slots(color=["silver"])),
    ("gold plated picture ledge", _slots(material_finish=["gold plated"])),
    ("walnut letter tray", _slots(material_finish=["walnut"])),
    ("dining chairs set of 6", _slots()),
    ("striped balcony cushion", _slots(room=["balcony"])),
    ("usb charging cable", _slots()),
)

CLASS_RULES = (
    "Choose the one class whose products the shopper wants to see. Copy the class name exactly.",
    "Judge by the product itself, not by a room, theme, size, colour or brand in the query.",
    "Prefer the most specific class that fits. An accessory or replacement part belongs to the "
    "accessory's class when there is one.",
    "If the query names only a theme or brand, choose the class such a shopper most likely means.",
    "Answer 'other' when the product is not home goods, or when no listed class fits.",
)

CLASS_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("swivel counter height chair with back", "Bar Stools"),
    ("running shoes size 9", "other"),
    ("linen panels 96 inch long blackout", "Curtains & Drapes"),
)


class RulesNotFinalError(RuntimeError):
    """Paid labelling was asked for while the rules may still change."""


def require_final(version: str) -> None:
    """Refuse a release candidate, because changed rules would waste the money and the labels."""
    if "-rc" in version:
        raise RulesNotFinalError(f"{version} is not final")


def slot_rules_text() -> str:
    """Render the slot rules as numbered lines."""
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(SLOT_RULES, 1))


def example_queries() -> list[str]:
    """Every query used as an example, for the leakage tests."""
    return [q for q, _ in SLOT_EXAMPLES] + [q for q, _ in CLASS_EXAMPLES]


def quoted_phrases() -> list[str]:
    """Phrases quoted in the rules, for the leakage tests."""
    return [p for rule in (*SLOT_RULES, *CLASS_RULES) for p in re.findall(r"'([a-z ]+)'", rule)]
