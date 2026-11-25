"""Slot lexicons built from the catalogue, and the lexicon matcher that is the slot baseline.

Which of color and material_finish a term fills follows the labelling rules, not the catalogue key
it is stored under: the catalogue lists walnut mostly as a colour, and the rules call every wood a
material. No query is read.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from shelfrank.data.product_features import parse_features
from shelfrank.qu.slot_vocab import (
    UNION,
    Term,
    build_vocabulary,
    load_vocabulary,
    majority_terms,
    room_spans,
    value_table,
)
from shelfrank.qu.slot_vocab import is_product_type_room as inside_product_type
from shelfrank.text.normalize import tokenize

# Rarer values are mostly ordinary words that one seller used as a colour ("ocean", "butter").
MIN_PRODUCTS = 20
MIN_SPECIES_PRODUCTS = 20
NOT_A_SPECIES = frozenset({"no", "espresso", "wood"})
# These follow the metal rules in shelfrank/qu/guidelines.py.
PLAIN_METALS = frozenset({"gold", "silver", "brass", "bronze", "copper"})
METALS = PLAIN_METALS | frozenset(
    {"nickel", "chrome", "steel", "iron", "pewter", "zinc", "aluminum"}
)
FINISH_WORDS = frozenset({"brushed", "polished", "satin", "matte", "rubbed", "plated", "oil"})
METAL_CUES = frozenset({"finish", "plated", "metal"})
# Two catalogue spellings of one value. Both sides of a match are mapped to the right-hand form.
SYNONYMS: dict[Term, Term] = {
    ("wooden",): ("wood",),
    ("multicolor",): ("multi",),
    ("multicolored",): ("multi",),
    ("multi", "color"): ("multi",),
    ("multi", "colored"): ("multi",),
    ("navy", "blue"): ("navy",),
    ("matt", "black"): ("matte", "black"),
    ("nickle",): ("nickel",),
    ("bohemian",): ("boho",),
    ("mid", "century"): ("mid", "century", "modern"),
    ("midcentury",): ("mid", "century", "modern"),
    ("midcentury", "modern"): ("mid", "century", "modern"),
}


def canonical(term: Term, known: Mapping[Term, int] | None = None) -> Term:
    """Preferred spelling of a term, unless the term itself is already known."""
    return term if known and term in known else SYNONYMS.get(term, term)


def wood_species(products: pd.DataFrame, min_products: int = MIN_SPECIES_PRODUCTS) -> set[str]:
    """One-word species in the catalogue's wood species keys."""
    counts: Counter[str] = Counter()
    for raw in products["product_features"]:
        pairs = parse_features(raw).pairs
        counts.update({v for k, v in pairs if k.endswith("woodspecies")})
    terms = {tuple(tokenize(v)) for v, n in counts.items() if n >= min_products}
    return {t[0] for t in terms if len(t) == 1 and t[0] not in NOT_A_SPECIES}


def slot_of_term(term: Term, majority: str, species: set[str]) -> str:
    """Slot of a catalogue value under the labelling rules."""
    words = set(term)
    if len(term) == 1 and term[0] in PLAIN_METALS:
        return "color"
    if term[-1] in species or words & FINISH_WORDS:
        return "material_finish"
    if "antique" in words and words & METALS:
        return "material_finish"
    return "color" if majority == "color" else "material_finish"


def build_lexicon(products: pd.DataFrame) -> dict[str, Any]:
    """Terms per slot with product counts, as plain data."""
    vocabulary = build_vocabulary(products)
    styles = load_vocabulary(vocabulary)[1]
    species = wood_species(products)
    out: dict[str, Any] = {"color": {}, "material_finish": {}}
    terms = majority_terms(value_table(products), UNION, MIN_PRODUCTS)
    for term, (majority, _, count) in sorted(terms.items()):
        if term not in styles:
            out[slot_of_term(term, majority, species)][" ".join(term)] = count
    # The rules give finish names even when few products carry them.
    out["style"] = {" ".join(t): n for t, n in sorted(styles.items())}
    out["product_types"] = vocabulary["product_types"]
    out["species"] = sorted(species)
    return out


class LexiconMatcher:
    """Longest-match lookup with the metal, product-type and room rules."""

    def __init__(self, lexicon: Mapping[str, Any]) -> None:
        """Read terms from the committed lexicon."""
        self.terms = {
            slot: {tuple(k.split()): n for k, n in lexicon[slot].items()}
            for slot in ("color", "material_finish", "style")
        }
        self.phrases = {tuple(p.split()) for p in lexicon["product_types"]}

    def slots(self, query: str) -> dict[str, list[str]]:
        """Slots of a query, as the query's own words."""
        words, owner, tokens = surface_tokens(query)
        # Every rule below reads this form, so token positions always refer to the same list.
        query = " ".join(tokens)
        taken: set[int] = set()
        found: dict[str, list[tuple[int, int]]] = {
            s: [] for s in ("color", "material_finish", "style", "room")
        }
        for start, end in room_spans(query):
            if end <= len(tokens) and not self._in_product_type(query, start, end):
                found["room"].append((start, end))
                taken.update(range(start, end))
        # Style first: "rustic" and "vintage" are listed as colours too, and the rules say style.
        attributes = {**self.terms["color"], **self.terms["material_finish"]}
        for slot, terms in (("style", self.terms["style"]), (None, attributes)):
            for start, end in self._matches(tokens, terms):
                span = set(range(start, end))
                if taken & span or self._in_product_type(query, start, end):
                    continue
                term = canonical(tuple(tokens[start:end]), terms)
                name = slot or self._attribute(term)
                if term in {(m,) for m in PLAIN_METALS} and set(tokens[end : end + 1]) & METAL_CUES:
                    name, end = "material_finish", end + 1
                found[name].append((start, end))
                taken.update(range(start, end))
        return {
            slot: [" ".join(words[owner[a] : owner[b - 1] + 1]) for a, b in sorted(spans)]
            for slot, spans in found.items()
        }

    @staticmethod
    def _matches(tokens: Sequence[str], terms: Mapping[Term, int]) -> list[tuple[int, int]]:
        """Token ranges that spell a term, longest first."""
        return [
            (start, start + size)
            for size in (3, 2, 1)
            for start in range(len(tokens) - size + 1)
            if canonical(tuple(tokens[start : start + size]), terms) in terms
        ]

    def _attribute(self, term: Term) -> str:
        return "color" if term in self.terms["color"] else "material_finish"

    def _in_product_type(self, query: str, start: int, end: int) -> bool:
        """Whether the words are part of a product's name.

        "coffee table" holds no colour. A phrase that is the whole span counts as well, so that
        "mirror" in "round mirror" is not a material.
        """
        if inside_product_type(query, (start, end), self.phrases):
            return True
        return tuple(tokenize(query)[start:end]) in self.phrases


def surface_tokens(query: str) -> tuple[list[str], list[int], list[str]]:
    """Words of a query, and for each normalised token the word it came from."""
    words = query.split()
    owner: list[int] = []
    tokens: list[str] = []
    for i, word in enumerate(words):
        for token in tokenize(word):
            owner.append(i)
            tokens.append(token)
    return words, owner, tokens


def span_scores(
    truth: Sequence[Mapping[str, Sequence[str]]],
    predicted: Sequence[Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, float | int]]:
    """Exact-span precision, recall and F1 per slot and over all slots."""

    def key(span: str) -> str:
        return " ".join(tokenize(span)) or span.casefold()

    out: dict[str, dict[str, float | int]] = {}
    total = [0, 0, 0]
    slots = ("color", "material_finish", "style", "room")
    for slot in slots:
        hits = made = wanted = 0
        for t, p in zip(truth, predicted, strict=True):
            a, b = {key(s) for s in t.get(slot, [])}, {key(s) for s in p.get(slot, [])}
            hits, made, wanted = hits + len(a & b), made + len(b), wanted + len(a)
        out[slot] = _prf(hits, made, wanted)
        total = [x + y for x, y in zip(total, (hits, made, wanted), strict=True)]
    return out | {"all": _prf(*total)}


def _prf(hits: int, made: int, wanted: int) -> dict[str, float | int]:
    return {
        "precision": hits / made if made else float("nan"),
        "recall": hits / wanted if wanted else float("nan"),
        "f1": 2 * hits / (made + wanted) if made + wanted else float("nan"),
        "predicted": made,
        "support": wanted,
    }
