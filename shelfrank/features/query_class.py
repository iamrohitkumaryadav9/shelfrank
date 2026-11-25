"""G3: agreement between the query's predicted class and the product's class.

The classifier is trained without WANDS queries, so its predictions carry no relevance label. The
judgment pool was partly chosen with Wayfair's own classifier, so gains from this group need the
pool rule (docs/PROTOCOL.md section 7).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from shelfrank.data.departments import OTHER_DEPARTMENT
from shelfrank.data.taxonomy import OTHER

NAMES = ("p_class", "p_class_rel", "class_rank", "is_top_class", "p_department", "confidence")


def department_mass(
    probabilities: np.ndarray, classes: Sequence[str], departments: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """Probability of each department per query. The label "other" belongs to none."""
    members: dict[str, list[int]] = {}
    for i, name in enumerate(classes):
        if name != OTHER:
            members.setdefault(departments.get(name, OTHER_DEPARTMENT), []).append(i)
    return {d: probabilities[:, columns].sum(axis=1) for d, columns in members.items()}


def class_features(
    pairs: pd.DataFrame,
    probabilities: np.ndarray,
    classes: Sequence[str],
    product_classes: Sequence[Sequence[str]],
    departments: Mapping[str, str],
) -> pd.DataFrame:
    """Columns g3_*. Missing, not 0, for a product without a class: its class is unknown."""
    column = {name: i for i, name in enumerate(classes)}
    order = np.argsort(-probabilities, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1] + 1)
    mass = department_mass(probabilities, classes, departments)
    best = probabilities.max(axis=1)
    rows = []
    for q, p in zip(pairs["qi"].tolist(), pairs["pi"].tolist(), strict=True):
        own = [c for c in product_classes[p] if c in column]
        if not own and product_classes[p] and OTHER in column:
            own = [OTHER]
        if not own:
            rows.append((np.nan,) * 5 + (best[q],))
            continue
        chance = max(probabilities[q, column[c]] for c in own)
        place = min(rank[q, column[c]] for c in own)
        near = {departments.get(c, OTHER_DEPARTMENT) for c in product_classes[p]}
        share = max((mass[d][q] for d in near if d in mass), default=0.0)
        rows.append((chance, chance / best[q], float(place), float(place == 1), share, best[q]))
    return pd.DataFrame(rows, columns=[f"g3_{c}" for c in NAMES])
