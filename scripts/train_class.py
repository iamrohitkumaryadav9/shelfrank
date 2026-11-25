"""Train the product-class student and compare its arms on dev queries. Writes results/class_dev.json.

The names-only arm needs nothing paid. The two arms that use ESCI queries run once the teacher has
labelled the seed (data/labels/class_esci_seed_*.jsonl); until then they are listed as waiting.
Every configuration tried is kept in the file.
"""

import json
import time
from typing import Any

import numpy as np

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.departments import department_map
from shelfrank.data.wands import load_products, load_queries
from shelfrank.eval.access import queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.qu.class_data import (
    BOTH,
    NAMES,
    QUERIES,
    SEED,
    arm_examples,
    labelled,
    seed_parts,
)
from shelfrank.qu.class_student import (
    Examples,
    LinearClassifier,
    evaluate_classes,
    ranked_labels,
    to_reference,
)
from shelfrank.stats.bootstrap import bootstrap_mean

NGRAMS = ((2, 4), (3, 5))
ALPHAS = (1e-6, 1e-5)
TAILS = (None, 3)
QUERY_SHARES = (0.2, 0.5, 0.8)


def latency_ms(model: LinearClassifier, queries: list[str]) -> dict[str, float]:
    """Time one query at a time, as serving does."""
    model.probabilities(queries[:3])
    times = []
    for query in queries:
        start = time.perf_counter()
        model.probabilities([query])
        times.append(1000 * (time.perf_counter() - start))
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def main() -> None:
    """Run the grid and save every configuration."""
    cfg = load_config("data")
    raw = REPO_ROOT / cfg["wands"]["raw_dir"]
    products = load_products(raw)
    taxonomy = json.loads((REPO_ROOT / cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))
    taxonomy = taxonomy["classes"]
    departments = department_map(products)
    ids = queries_for("dev", load_folds(REPO_ROOT / cfg["splits"]["path"]))
    frame = load_queries(raw).set_index("query_id").loc[ids]
    dev = frame["query"].tolist()
    truth = [to_reference(c, taxonomy) for c in frame["query_class"]]

    slice_texts, slice_labels = labelled("esci_dev", "esci_prompt_dev.txt") or ([], [])
    parts = seed_parts()
    seeded: set[str] | None = None
    train_queries: Examples | None = None
    esci_texts, esci_labels = slice_texts, slice_labels
    if parts:
        train_queries, (esci_texts, esci_labels) = parts
        seeded = set(train_queries.labels)

    def arms(tail: int | None) -> list[tuple[str, float | None]]:
        out: list[tuple[str, float | None]] = [(NAMES, None)]
        if train_queries:
            out += [(BOTH, share) for share in QUERY_SHARES]
            # Seed queries alone do not depend on the tail of product names.
            out += [(QUERIES, None)] if tail is None else []
        return out

    runs: list[dict[str, Any]] = []
    for tail in TAILS:
        for arm, share in arms(tail):
            examples = arm_examples(arm, tail, share, products, taxonomy, train_queries)
            for ngrams in NGRAMS:
                for alpha in ALPHAS:
                    start = time.perf_counter()
                    model = LinearClassifier(ngrams, alpha, 60_000, SEED).fit(examples)
                    top = ranked_labels(model.probabilities(dev), model.classes, 3)
                    esci = ranked_labels(model.probabilities(esci_texts), model.classes, 3)
                    run: dict[str, Any] = {
                        "arm": arm,
                        "query_share": share,
                        "tail": tail,
                        "ngrams": ngrams,
                        "alpha": alpha,
                        "dev": evaluate_classes(truth, top, departments, seeded),
                        "esci": evaluate_classes(
                            [str(x) for x in esci_labels], esci, departments, None
                        ),
                        "fit_seconds": time.perf_counter() - start,
                        "latency_ms": latency_ms(model, dev),
                    }
                    right = [float(t == p[0]) for t, p in zip(truth, top, strict=True)]
                    interval = bootstrap_mean(right, seed=SEED)
                    run["dev"] |= {"low": interval.low, "high": interval.high}
                    right = [
                        float(t == p[0]) for t, p in zip(map(str, esci_labels), esci, strict=True)
                    ]
                    interval = bootstrap_mean(right, seed=SEED)
                    run["esci"] |= {"low": interval.low, "high": interval.high}
                    run["selection"] = (run["dev"]["accuracy"] + run["esci"]["accuracy"]) / 2
                    runs.append(run)
                    print(
                        f"{arm:8s} share {share!s:4s} tail {tail!s:5s} {ngrams} {alpha:.0e}  dev {run['dev']['accuracy']:.3f}"
                        f"  esci {run['esci']['accuracy']:.3f}  {run['fit_seconds']:.0f}s",
                        flush=True,
                    )
    best: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run["arm"] not in best or run["selection"] > best[run["arm"]]["selection"]:
            best[run["arm"]] = run
    selected = max(best.values(), key=lambda r: r["selection"])["arm"]
    payload = {
        "reference": "Wayfair's assigned query class",
        "esci_reference": "teacher labels, "
        + ("held-out seed queries" if parts else "prompt-development slice"),
        "selection_rule": "mean of dev and ESCI accuracy",
        "waiting_for_seed_labels": parts is None,
        "dev_reference_other": truth.count("other"),
        "seed_training_queries": len(train_queries.texts) if train_queries else 0,
        "seeded_classes": len(seeded or ()),
        "selected": selected,
        "best": best,
        "runs": runs,
    }
    path = REPO_ROOT / "results/class_dev.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("selected", selected, json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
