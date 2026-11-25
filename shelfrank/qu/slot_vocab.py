"""Slot vocabularies and their evidence, read from the catalogue only."""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from shelfrank.data.product_features import STYLE_KEYS, parse_features
from shelfrank.data.taxonomy import class_support
from shelfrank.data.wands import split_hierarchy
from shelfrank.qu.class_terms import name_phrases
from shelfrank.text.normalize import tokenize

Term = tuple[str, ...]

# Style values that describe an audience, a grade or a product type rather than a design style.
NOT_A_STYLE = frozenset(
    "no style|style agnostic|top grade|seasonal|utility|utilitarian|animal|specialty|garage|basic|"
    "decorative|convertible chair|children's|youth|teen|storage|org|shag|classroom|educational|"
    "global|electric geometric|weathered worn down".split("|")
)
# Spellings of one catalogue style that shoppers shorten.
STYLE_ALIASES = {
    "mid century": "mid-century modern",
    "midcentury": "mid-century modern",
    "midcentury modern": "mid-century modern",
}
_AUDIENCE = re.compile(r"^(youth|teen)\s+")
_PARTS = re.compile(r"\s*(?:&|/| - )\s*")

# Room words as the catalogue writes them (roomuse values, category path segments, class names).
# Written from that text and not from queries. Longer names come first when matching.
ROOM_WORDS = (
    "living room|dining room|family room|laundry room|kids room|mud room|sun room|game room|"
    "conference room|home office|home theater|bedroom|bathroom|bath|kitchen|office|nursery|"
    "entryway|foyer|hallway|closet|pantry|basement|garage|sunroom|mudroom|classroom|dining|"
    "living|den|loft|outdoor|indoor|patio|garden|balcony|porch|deck|backyard|lawn|poolside"
).split("|")
# One place for the purpose of the room rule and of matching.
SAME_PLACE = {"patio": "outdoor", "garden": "outdoor", "balcony": "outdoor", "deck": "outdoor"}
# A value needs this many products to be in the catalogue vocabulary.
MIN_PRODUCTS = 3
UNION = ("color", "finish", "material")


def key_group(key: str) -> str | None:
    """Attribute family of a feature key.

    Finish is separate from colour and material because the catalogue keeps it in its own keys.
    """
    if key in STYLE_KEYS:
        return "style"
    if key.endswith(("finish", "finishes")) or key == "finishcolor":
        return "finish"
    if key.endswith("color") or key == "dswoodtone":
        return "color"
    if key.startswith("contains"):
        return None
    if key.endswith(("material", "materialdetails", "woodspecies")):
        return "material"
    return None


def value_table(products: pd.DataFrame) -> dict[str, Counter[str]]:
    """Products per attribute family for each value, counting a product once per family."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for raw in products["product_features"]:
        seen = {(key_group(k), v) for k, v in parse_features(raw).pairs}
        for group, value in seen:
            if group and value and value not in ("yes", "no"):
                table[value][group] += 1
    return dict(table)


def style_terms(
    table: Mapping[str, Counter[str]], min_products: int = MIN_PRODUCTS
) -> dict[str, int]:
    """Style names in the catalogue's style keys, with product counts.

    "Country / farmhouse" names two styles, and an audience prefix ("youth boho") is not part of
    the style, so values are split and stripped before counting.
    """
    counts: Counter[str] = Counter()
    for value, groups in table.items():
        if not groups["style"]:
            continue
        for part in _PARTS.split(value):
            name = _AUDIENCE.sub("", part.strip())
            if name and name not in NOT_A_STYLE:
                counts[name] += groups["style"]
    return {name: n for name, n in counts.most_common() if n >= min_products}


def majority_terms(
    table: Mapping[str, Counter[str]], groups: Sequence[str], min_products: int = 20
) -> dict[Term, tuple[str, float, int]]:
    """Normalised value -> (family that holds it most often, its share, total products)."""
    merged: dict[Term, Counter[str]] = defaultdict(Counter)
    for value, counts in table.items():
        term = tuple(tokenize(value))
        if 1 <= len(term) <= 3 and all(t.isalpha() and len(t) > 2 for t in term):
            merged[term].update({g: n for g, n in counts.items() if g in groups})
    out = {}
    for term, counts in merged.items():
        total = sum(counts.values())
        if total >= min_products:
            group = min(counts, key=lambda g: (-counts[g], g))
            out[term] = (group, counts[group] / total, total)
    return out


def product_type_phrases(
    products: pd.DataFrame, min_class: int = 30, min_path: int = 100
) -> set[Term]:
    """Phrases that name a product type, from class names and category path segments."""
    support = class_support(products["product_class"])
    names = [name for name, count in support.items() if count >= min_class]
    segments: Counter[str] = Counter()
    for path in products["category_hierarchy"]:
        segments.update(set(split_hierarchy(path)))
    names += [name for name, count in segments.items() if count >= min_path]
    return {_place(phrase) for name in names for phrase in name_phrases(name)}


def _place(tokens: Iterable[str]) -> Term:
    return tuple(SAME_PLACE.get(token, token) for token in tokens)


def room_spans(query: str) -> list[tuple[int, int]]:
    """Token ranges of room words, longest first, without overlaps."""
    tokens = tokenize(query)
    words = sorted((tuple(tokenize(w)) for w in ROOM_WORDS), key=lambda w: (-len(w), w))
    taken: set[int] = set()
    spans = []
    for word in words:
        for start in range(len(tokens) - len(word) + 1):
            span = range(start, start + len(word))
            if tuple(tokens[start : start + len(word)]) == word and not taken.intersection(span):
                taken.update(span)
                spans.append((start, start + len(word)))
    return sorted(spans)


def is_product_type_room(query: str, span: tuple[int, int], phrases: set[Term]) -> bool:
    """Whether a room word is part of the product's name and so not a room slot.

    It is when it lies inside a product-type phrase that continues past it: both room words of
    "patio dining set" belong to the name, and the head noun is what makes it one.
    """
    tokens = _place(tokenize(query))
    start, end = span
    return any(
        tokens[first : first + len(phrase)] == phrase
        for phrase in phrases
        for first in range(max(0, end + 1 - len(phrase)), start + 1)
    )


def room_slots(query: str, phrases: set[Term]) -> list[str]:
    """Room words of a query that are slots under the product-type rule."""
    tokens = tokenize(query)
    return [
        " ".join(tokens[a:b])
        for a, b in room_spans(query)
        if not is_product_type_room(query, (a, b), phrases)
    ]


def match_terms(query: str, terms: Mapping[Term, Any]) -> list[Term]:
    """Vocabulary terms in a query, longest first, without overlaps."""
    tokens = tokenize(query)
    taken: set[int] = set()
    found = []
    for size in (3, 2, 1):
        for start in range(len(tokens) - size + 1):
            span = range(start, start + size)
            term = tuple(tokens[start : start + size])
            if term in terms and not taken.intersection(span):
                taken.update(span)
                found.append(term)
    return found


def in_vocabulary(
    slot: str, span: str, union: Mapping[Term, Any], styles: Mapping[Term, Any]
) -> bool:
    """Whether a labelled span contains a catalogue value of its slot.

    Containment and not equality: "very dark gray" is a catalogue colour with a modifier. Colours
    and materials share one vocabulary, because matching is key-agnostic.
    """
    if slot == "room":
        return bool(room_spans(span))
    return bool(match_terms(span, styles if slot == "style" else union))


def build_vocabulary(products: pd.DataFrame) -> dict[str, Any]:
    """Everything the slot code needs from the catalogue, as plain data."""
    values = value_table(products)
    union = majority_terms(values, UNION, MIN_PRODUCTS)
    return {
        "min_products": MIN_PRODUCTS,
        "union": {" ".join(t): n for t, (_, _, n) in sorted(union.items())},
        "styles": style_terms(values),
        "product_types": sorted(" ".join(p) for p in product_type_phrases(products)),
    }


def load_vocabulary(raw: Mapping[str, Any]) -> tuple[dict[Term, int], dict[Term, int], set[Term]]:
    """Return the union terms, the style terms and the product-type phrases, tokenised."""
    union = {tuple(name.split()): n for name, n in raw["union"].items()}
    styles = {tuple(tokenize(name)): n for name, n in raw["styles"].items()}
    for alias, name in STYLE_ALIASES.items():
        styles[tuple(tokenize(alias))] = raw["styles"][name]
    return union, styles, {tuple(name.split()) for name in raw["product_types"]}
