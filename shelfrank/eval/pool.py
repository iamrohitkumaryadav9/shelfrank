"""Diagnostics for how the judgment pool affects a ranking. Analysis only, never a feature."""

from collections.abc import Iterable, Mapping

from shelfrank.eval.harness import Qrels, Run


def judged_by_class(
    run: Run,
    qrels: Qrels,
    query_class: Mapping[int, object],
    product_classes: Mapping[int, Iterable[str]],
    k: int = 10,
) -> dict[str, float | int]:
    """Judged share of the top ``k``, split by whether the product is in the query's class.

    Queries without a class are left out, because "outside the class" means nothing for them.
    """
    counts = {"in": [0, 0], "out": [0, 0]}
    for q, judged in qrels.items():
        if not isinstance(query_class.get(q), str):
            continue
        for product in run[q][:k]:
            side = counts["in" if query_class[q] in product_classes.get(product, ()) else "out"]
            side[0] += product in judged
            side[1] += 1
    return {
        "in_class_judged": counts["in"][0] / max(counts["in"][1], 1),
        "in_class_n": counts["in"][1],
        "out_of_class_judged": counts["out"][0] / max(counts["out"][1], 1),
        "out_of_class_n": counts["out"][1],
    }
