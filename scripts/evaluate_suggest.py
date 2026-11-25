"""Offline evaluation of search suggestions on held-out ESCI queries. Writes results/suggest.json.

Corpus: the 1,600 ESCI seed queries the class student also trains on. Held out: the other 400. The
weight of the class prior is chosen by 5-fold cross-validation inside the 1,600. No WANDS query is
used anywhere.
"""

import json
import random
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_products
from shelfrank.ltr.class_for_ranking import class_model
from shelfrank.qu.class_data import HELD_SHARE, SEED, labelled
from shelfrank.qu.class_student import split_seed
from shelfrank.stats.bootstrap import bootstrap_mean, paired_difference
from shelfrank.suggest.model import (
    FrequencyBaseline,
    SuggestModel,
    prefix_cases,
    reciprocal_rank,
)

WEIGHTS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
LENGTHS = (1, 2, 3)
K = 10


def mrr(model: Any, queries: list[str], length: int) -> list[float]:  # noqa: ANN401
    """Reciprocal rank of the true next word, one value per query long enough."""
    return [
        reciprocal_rank(model.rank(words, K), truth)
        for _, words, truth in prefix_cases(queries, length)
    ]


def main() -> None:
    """Tune inside the training queries, then score once on the held-out ones."""
    seed = labelled("esci_seed", "esci_seed_queries.txt")
    assert seed is not None
    texts, labels = seed
    train_ids, held_ids = split_seed(len(texts), HELD_SHARE, SEED)
    train = [texts[i] for i in train_ids]
    classes = [str(labels[i]) for i in train_ids]
    held = [texts[i] for i in held_ids]

    cfg = load_config("data")
    products = load_products(REPO_ROOT / cfg["wands"]["raw_dir"])
    taxonomy = json.loads((REPO_ROOT / cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))[
        "classes"
    ]
    student = class_model(products, taxonomy)
    cache: dict[str, dict[str, float]] = {}

    def prior(text: str) -> dict[str, float]:
        if text not in cache:
            row = student.probabilities([text])[0]
            cache[text] = {c: float(p) for c, p in zip(student.classes, row, strict=True)}
        return cache[text]

    order = list(range(len(train)))
    random.Random(SEED).shuffle(order)
    folds = [order[i::5] for i in range(5)]
    log = []
    for weight in WEIGHTS:
        scores = []
        for fold in folds:
            inside = [i for i in order if i not in set(fold)]
            model = SuggestModel(
                [train[i] for i in inside], [classes[i] for i in inside], prior, weight
            )
            for length in LENGTHS:
                scores += mrr(model, [train[i] for i in fold], length)
        log.append({"class_weight": weight, "mrr": float(np.mean(scores)), "n": len(scores)})
        print(log[-1], flush=True)
    chosen = max(log, key=lambda r: r["mrr"])["class_weight"]

    models = {
        "frequency_only": FrequencyBaseline(train),
        "ngram": SuggestModel(train),
        "ngram_class_prior": SuggestModel(train, classes, prior, chosen),
    }
    out: dict[str, Any] = {}
    for length in LENGTHS:
        cases = prefix_cases(held, length)
        rows = {name: mrr(model, held, length) for name, model in models.items()}
        vocabulary = {w for q in train for w in q.lower().split()}
        out[str(length)] = {
            "n": len(cases),
            "next_word_in_training_vocabulary": float(
                np.mean([t in vocabulary for _, _, t in cases])
            ),
            "mrr": {name: asdict(bootstrap_mean(v)) for name, v in rows.items()},
            "ngram_minus_frequency": asdict(
                paired_difference(rows["ngram"], rows["frequency_only"])
            ),
            "class_prior_minus_ngram": asdict(
                paired_difference(rows["ngram_class_prior"], rows["ngram"])
            ),
        }
    times = []
    for _, words, _ in prefix_cases(held, 1)[:200]:
        start = time.perf_counter()
        models["ngram"].rank(words, K)
        times.append(1000 * (time.perf_counter() - start))
    payload = {
        "train_queries": len(train),
        "held_out_queries": len(held),
        "k": K,
        "class_weight": chosen,
        "tuning": log,
        "prefix_words": out,
        "ngram_latency_ms": {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
        },
    }
    path = REPO_ROOT / "results/suggest.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1), "weight", chosen)


if __name__ == "__main__":
    main()
