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
from shelfrank.qu.class_student import (
    Examples,
    LinearClassifier,
    evaluate_classes,
    mix,
    name_examples,
    query_examples,
    ranked_labels,
    split_seed,
    to_reference,
)
from shelfrank.qu.guidelines import CLASS_VERSION
from shelfrank.stats.bootstrap import bootstrap_mean

SEED, HELD_SHARE = 20260, 0.2
NGRAMS = ((2, 4), (3, 5))
ALPHAS = (1e-6, 1e-5)
TAILS = (None, 3)
QUERY_SHARES = (0.2, 0.5, 0.8)


def labelled(source: str, text_file: str) -> tuple[list[str], list[str | None]] | None:
    """Teacher-labelled ESCI queries, or None before they exist."""
    path = REPO_ROOT / "data/labels" / f"class_{source}_{CLASS_VERSION}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    texts = (REPO_ROOT / "data/esci" / text_file).read_text(encoding="utf-8").splitlines()
    return [texts[row["id"]] for row in rows], [row["label"] for row in rows]


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
    seed = labelled("esci_seed", "esci_seed_queries.txt")
    seeded: set[str] | None = None
    train_queries: Examples | None = None
    esci_texts, esci_labels = slice_texts, slice_labels
    if seed:
        train, held = split_seed(len(seed[0]), HELD_SHARE, SEED)
        train_queries = query_examples([seed[0][i] for i in train], [seed[1][i] for i in train])
        esci_texts, esci_labels = [seed[0][i] for i in held], [seed[1][i] for i in held]
        seeded = set(train_queries.labels)

    def arms(tail: int | None) -> dict[str, tuple[Examples, float | None]]:
        names = name_examples(products, taxonomy, tail)
        out: dict[str, tuple[Examples, float | None]] = {"names": (names, None)}
        if train_queries:
            out |= {f"both_{s}": (mix(names, train_queries, s), s) for s in QUERY_SHARES}
            if tail is None:
                out["queries"] = (train_queries, None)
        return out

    runs: list[dict[str, Any]] = []
    for tail in TAILS:
        for arm, (examples, share) in arms(tail).items():
            for ngrams in NGRAMS:
                for alpha in ALPHAS:
                    start = time.perf_counter()
                    model = LinearClassifier(ngrams, alpha, 60_000, SEED).fit(examples)
                    top = ranked_labels(model.probabilities(dev), model.classes, 3)
                    esci = ranked_labels(model.probabilities(esci_texts), model.classes, 3)
                    run: dict[str, Any] = {
                        "arm": arm.split("_")[0],
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
                    run["selection"] = (run["dev"]["accuracy"] + run["esci"]["accuracy"]) / 2
                    runs.append(run)
                    print(
                        f"{arm:9s} tail {tail!s:5s} {ngrams} {alpha:.0e}  dev {run['dev']['accuracy']:.3f}"
                        f"  esci {run['esci']['accuracy']:.3f}  {run['fit_seconds']:.0f}s",
                        flush=True,
                    )
    best: dict[str, dict[str, Any]] = {}
    for run in runs:
        if run["arm"] not in best or run["selection"] > best[run["arm"]]["selection"]:
            best[run["arm"]] = run
    payload = {
        "reference": "Wayfair's assigned query class",
        "esci_reference": "teacher labels, "
        + ("held-out seed queries" if seed else "prompt-development slice"),
        "selection_rule": "mean of dev and ESCI accuracy",
        "waiting_for_seed_labels": seed is None,
        "dev_reference_other": truth.count("other"),
        "best": best,
        "runs": runs,
    }
    path = REPO_ROOT / "results/class_dev.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main()
