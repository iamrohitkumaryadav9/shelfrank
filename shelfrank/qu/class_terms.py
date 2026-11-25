"""Build search terms for each product class from the catalogue, never from queries."""

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import pandas as pd

from shelfrank.data.wands import split_classes
from shelfrank.text.normalize import tokenize

Term = tuple[str, ...]
# Single words that name a product type here but mean something else on a general marketplace
# ("phone case", "game system"). They still count inside longer terms such as "display case".
AMBIGUOUS = frozenset(
    "accessory assembly base bottle box case component dog head hood hook insert only organization "
    "pad panel paper plate pull recycling remote rod server stand system".split()
)
_PARTS = re.compile(r"&|,|/|\band\b", re.IGNORECASE)


def name_phrases(class_name: str) -> set[Term]:
    """Read the phrases that a class name lists.

    "Coffee & Cocktail Tables" names coffee tables and cocktail tables, so a one-word part borrows
    the head noun of the last part. When every part is one word ("Dressers & Chests"), each stands
    alone.
    """
    parts = [tuple(tokenize(part)) for part in _PARTS.split(class_name)]
    parts = [part for part in parts if part]
    if not parts:
        return set()
    head = parts[-1][-1]
    borrows = len(parts) > 1 and len(parts[-1]) > 1
    return {
        (*part, head) if borrows and len(part) == 1 and part[0] != head else part for part in parts
    }


def endings(tokens: Sequence[str], sizes: Iterable[int]) -> set[Term]:
    """Last words of a product name.

    Names end with the product type ("... polyester armchair"). Words elsewhere in the name are
    often precise inside this catalogue ("polyester", "red") and meaningless on a marketplace.
    """
    return {tuple(tokens[-size:]) for size in sizes if len(tokens) >= size}


def build_class_terms(
    products: pd.DataFrame,
    taxonomy: Sequence[str],
    min_products: int = 10,
    min_precision: float = 0.6,
    min_products_single: int = 30,
    min_precision_single: float = 0.8,
    max_mined: int = 8,
) -> dict[str, list[str]]:
    """Terms per class: phrases in its name, plus name endings that mark it.

    Single words face a higher bar because they match far more titles. A term belongs to one class
    only, the one where it is most common.
    """
    total: Counter[Term] = Counter()
    by_class: dict[str, Counter[Term]] = defaultdict(Counter)
    for name, cell in zip(products["product_name"], products["product_class"], strict=True):
        found = endings(tokenize(name), (1, 2, 3))
        total.update(found)
        for label in set(split_classes(cell)):
            by_class[label].update(found)

    def passes(term: Term, count: int) -> bool:
        single = len(term) == 1
        enough = count >= (min_products_single if single else min_products)
        precise = count / total[term] >= (min_precision_single if single else min_precision)
        return enough and precise and all(w.isalpha() and len(w) > 2 for w in term)

    owner: dict[Term, tuple[float, str]] = {}

    def claim(term: Term, strength: float, label: str) -> None:
        if len(term) == 1 and term[0] in AMBIGUOUS:
            return
        current = owner.get(term)
        if current is None or (-strength, label) < (-current[0], current[1]):
            owner[term] = (strength, label)

    for label in taxonomy:
        for phrase in name_phrases(label):
            claim(phrase, 1.0 + by_class[label][phrase], label)
        mined = [(count, term) for term, count in by_class[label].items() if passes(term, count)]
        for count, term in sorted(mined, key=lambda item: (-item[0], item[1]))[:max_mined]:
            claim(term, 1.0 + count, label)

    terms: dict[str, list[str]] = {label: [] for label in taxonomy}
    for term, (_, label) in sorted(owner.items()):
        terms[label].append(" ".join(term))
    return terms
