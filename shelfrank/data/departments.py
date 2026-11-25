"""Map product classes to top-level departments, for reporting only."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

import pandas as pd

from shelfrank.data.wands import split_classes, split_hierarchy

OTHER_DEPARTMENT = "Other"
MIN_PRODUCTS = 100


def department_map(products: pd.DataFrame, min_products: int = MIN_PRODUCTS) -> dict[str, str]:
    """Give each class the most common first level of its products' category paths.

    First levels with few products are navigation paths such as "Sale" or a brand page, not
    departments, so classes that land on one become "Other". Ties go to the alphabetically first
    level so that the mapping never depends on row order.
    """
    level_size: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for cell, path in zip(products["product_class"], products["category_hierarchy"], strict=True):
        levels = split_hierarchy(path)
        if not levels:
            continue
        level_size[levels[0]] += 1
        for name in set(split_classes(cell)):
            by_class[name][levels[0]] += 1
    mapping = {}
    for name, counts in by_class.items():
        top = min(counts, key=lambda level: (-counts[level], level))
        mapping[name] = top if level_size[top] >= min_products else OTHER_DEPARTMENT
    return mapping


def departments_of(classes: Iterable[object], mapping: Mapping[str, str]) -> list[str]:
    """Department per class cell; a missing or unmapped class is "Other"."""
    return [
        mapping.get(cell, OTHER_DEPARTMENT) if isinstance(cell, str) else OTHER_DEPARTMENT
        for cell in classes
    ]
