"""Labelling rules and examples, shared by the teacher prompt and the human annotator.

GOVERNING PRINCIPLE: a slot's value set is defined by the WANDS catalogue's own attribute keys,
because slots exist to be matched against product features. docs/slot_evidence.md shows the keys.

One source for both labellers: agreement is meaningless if they follow different rules. The examples
were invented for this file and are checked against WANDS by the leakage tests.
"""

SLOTS = ("color", "material", "style", "room")

# Recorded in the cache entry of every label. "-rc" marks rules that are not final, and paid slot
# labelling is refused while it is there.
SLOT_VERSION = "slots-v1-rc1"
CLASS_VERSION = "class-v1"

SLOT_RULES = (
    "Copy words exactly as they appear in the query, including misspellings. Never add a word.",
    "Leave a slot empty when the query does not state it. Do not infer it from the product type.",
    "color: a colour, colour family or shade (navy, light gray, espresso, multicolor). Copy the "
    "whole phrase.",
    "material: what the product is made of, covered with or finished in. This includes wood "
    "species, fabrics, stone, metals and metal finishes (oak, velvet, marble, brushed nickel, "
    "chrome, matte black when it is a finish).",
    "style: a named design style (modern, farmhouse, mid century, boho, rustic, coastal, "
    "industrial). Patterns (floral, striped, geometric, plaid, solid), shapes, sizes and subjects "
    "are not styles and have no slot.",
    "room: where the product goes, indoors or outdoors (bedroom, bathroom, patio, garden, "
    "balcony). It is not a slot when the room word and the product noun together name a product "
    "category ('dining table', 'kitchen island', 'bathroom vanity'). Otherwise it is "
    "('bathroom rug', 'patio umbrella', 'bedroom lamp').",
    "A word belongs to one slot.",
    "Brand names, collection names, quantities and dimensions belong to no slot.",
)

SLOT_EXAMPLES: tuple[tuple[str, dict[str, list[str]]], ...] = (
    (
        "dark gray linen headboard queen",
        {"color": ["dark gray"], "material": ["linen"], "style": [], "room": []},
    ),
    (
        "farmhouse plant stand for sunroom",
        {"color": [], "material": [], "style": ["farmhouse"], "room": ["sunroom"]},
    ),
    (
        "brushed nickel towel ring",
        {"color": [], "material": ["brushed nickel"], "style": [], "room": []},
    ),
    ("dining chairs set of 6", {"color": [], "material": [], "style": [], "room": []}),
    ("striped balcony cushion", {"color": [], "material": [], "style": [], "room": ["balcony"]}),
    ("usb charging cable", {"color": [], "material": [], "style": [], "room": []}),
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
