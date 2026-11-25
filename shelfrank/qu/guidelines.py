"""Labelling rules and examples, shared by the teacher prompt and the human annotator.

One source for both: agreement between the two is meaningless if they follow different rules. The
examples were invented for this file and are checked against WANDS by the leakage tests.
"""

SLOTS = ("color", "material", "style", "room")

SLOT_RULES = (
    "Copy words exactly as they appear in the query, including misspellings. Never add a word.",
    "Leave a slot empty when the query does not state it. Do not infer it from the product type.",
    "color: a colour or colour family, including shades and wood tones used as a colour "
    "(navy, espresso, light gray, multicolor). Copy the whole phrase.",
    "material: what it is made of or covered with, including wood species, fabrics, stone and "
    "metals (oak, velvet, marble, brushed nickel, wicker). A metal finish is a material.",
    "style: a named design style or period (modern, farmhouse, mid century, boho, rustic, "
    "coastal). Shapes, sizes, patterns and themes are not styles.",
    "room: where it is used, indoors or outdoors (bedroom, kitchen, office, patio, outdoor). "
    "Only when it says where the product goes, not when it is part of the product's name "
    "('dining table' and 'kitchen sink' have no room).",
    "A word belongs to one slot. If it could be a colour or a material, choose material when it "
    "describes what the product is made of.",
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


def slot_rules_text() -> str:
    """Render the slot rules as numbered lines."""
    return "\n".join(f"{i}. {rule}" for i, rule in enumerate(SLOT_RULES, 1))


def example_queries() -> list[str]:
    """Every query used as an example, for the leakage tests."""
    return [q for q, _ in SLOT_EXAMPLES] + [q for q, _ in CLASS_EXAMPLES]
