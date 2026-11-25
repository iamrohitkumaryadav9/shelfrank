"""Evaluate "More like this". Dev writes results/similar_dev.json with every grid point.

Primary (after the freeze) reads its settings from configs/frozen.yaml and tunes nothing.
"""

import argparse
import json
from dataclasses import asdict
from itertools import product as grid_product
from typing import Any

import numpy as np

from shelfrank.bm25.fields import BM25Index, concat_fields, field_tokens
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import judgments, load_labels, load_products, split_classes
from shelfrank.eval.access import queries_for
from shelfrank.eval.claims import verdict
from shelfrank.eval.harness import to_qrels
from shelfrank.eval.splits import load_folds
from shelfrank.features.attributes import listed_values
from shelfrank.features.priors import prior_features
from shelfrank.ltr.slots_for_ranking import term_slots
from shelfrank.similar.sets import (
    DEPTH,
    METRICS,
    by_query,
    clustered_mean,
    difference,
    draw_seeds,
    evaluate_seeds,
    substitute_sets,
)
from shelfrank.similar.systems import (
    RerankWeights,
    agreement,
    bm25_rankings,
    dense_rankings,
    fuse,
    rating_prior,
    rerank,
    shared_class,
)

ENCODERS = ("bge_small", "bge_small_esci")
GRID = {
    "shared_class": (0.0, 0.1, 0.25, 0.5, 1.0),
    "attributes": (0.0, 0.1, 0.25, 0.5),
    "prior": (0.0, 0.05, 0.1),
}


def objective(means: dict[str, dict[str, float]]) -> float:
    """Mean of standard and condensed nDCG@10."""
    return (means["ndcg10"]["mean"] + means["condensed_ndcg10"]["mean"]) / 2


def main() -> None:
    """Run every system on one split."""
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "primary"])
    split = parser.parse_args().split
    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    product_ids = products["product_id"].to_numpy()
    row_of = {int(p): i for i, p in enumerate(product_ids)}
    ids = queries_for(split, load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    qrels = to_qrels(judgments(load_labels(raw)), ids)
    sets = substitute_sets(qrels)
    seeds = draw_seeds(sets)
    wanted = sorted({s.product_id for s in seeds})
    rows = {p: row_of[p] for p in wanted}
    boot = cfg["bootstrap"]
    frozen = load_config("frozen")["similar"] if split == "primary" else None

    tokens = field_tokens(products)
    k = cfg["rrf"]["k"]
    index = BM25Index(
        concat_fields(tokens, ["name", "class", "description"]), cfg["bm25"]["k1"], cfg["bm25"]["b"]
    )
    query_tokens = {p: [*tokens["name"][i], *tokens["class"][i]] for p, i in rows.items()}
    runs: dict[str, Any] = {"bm25": bm25_rankings(index, query_tokens, product_ids, DEPTH)}
    for name in ENCODERS:
        vectors = np.load(REPO_ROOT / "artifacts/embeddings" / name / "embeddings.npy")
        runs[name] = dense_rankings(vectors, rows, product_ids, DEPTH)

    def scored(run: dict[int, list[int]]) -> dict[Any, dict[str, float]]:
        return evaluate_seeds({s: run[s.product_id] for s in seeds}, sets, qrels)

    def means(scores: dict[Any, dict[str, float]], weighted: bool = False) -> dict[str, Any]:
        return {m: asdict(clustered_mean(by_query(scores, m), weighted, **boot)) for m in METRICS}

    per = {name: scored(run) for name, run in runs.items()}
    # max keeps the first on a tie, which is off-the-shelf.
    chosen = max(ENCODERS, key=lambda n: objective(means(per[n])))
    encoder = frozen["encoder"] if frozen else chosen
    runs["rrf"] = fuse(runs["bm25"], runs[encoder], k, DEPTH)
    per["rrf"] = scored(runs["rrf"])

    classes = [split_classes(c) for c in products["product_class"]]
    prior_table = prior_features(pd.DataFrame({"pi": range(len(products))}), products)
    prior = rating_prior(
        prior_table["g5_shrunk_rating"].to_numpy(),
        products["review_count"].astype(float).fillna(0.0).to_numpy(),
    )
    needed = {row_of[p] for run in runs["rrf"].values() for p in run} | set(rows.values())
    slot_of = term_slots()
    listed = {
        i: listed_values(
            products["product_features"][i], products["category_hierarchy"][i], slot_of
        )
        for i in sorted(needed)
    }
    terms = {}
    for p, ranking in runs["rrf"].items():
        a = rows[p]
        terms[p] = {
            c: (
                shared_class(classes[a], classes[row_of[c]]),
                agreement(listed[a], listed[row_of[c]]),
                float(prior[row_of[c]]),
            )
            for c in ranking
        }

    def reranked(weights: RerankWeights) -> dict[int, list[int]]:
        pair = {p: (runs["bm25"][p], runs[encoder][p]) for p in wanted}
        return {p: rerank(p, runs["rrf"][p], k, pair[p], weights, terms[p]) for p in wanted}

    log = []
    if frozen:
        weights = RerankWeights(**frozen["weights"])
    else:
        best = -1.0
        for values in grid_product(*GRID.values()):
            candidate = RerankWeights(**dict(zip(GRID, values, strict=True)))
            scores = scored(reranked(candidate))
            row = {
                m: float(np.mean([np.mean(v) for v in by_query(scores, m).values()]))
                for m in ("ndcg10", "condensed_ndcg10")
            }
            value = (row["ndcg10"] + row["condensed_ndcg10"]) / 2
            log.append({"weights": asdict(candidate), **row, "objective": value})
            if value > best:
                best, weights = value, candidate
    runs["rerank"] = reranked(weights)
    per["rerank"] = scored(runs["rerank"])

    def versus(system: str, baseline: str) -> dict[str, Any]:
        out: dict[str, Any] = {"system": system, "baseline": baseline}
        parts = {m: difference(per[system], per[baseline], m, **boot) for m in METRICS}
        out |= {m: asdict(v) for m, v in parts.items()}
        out["verdict"] = verdict(parts["ndcg10"], parts["condensed_ndcg10"])
        return out

    ceiling = clustered_mean(by_query(per["bm25"], "recall10_ceiling"), False, **boot).mean
    payload = {
        "split": split,
        "n_queries": len(sets),
        "n_seeds": len(seeds),
        "encoder": encoder,
        "weights": asdict(weights),
        "recall10_ceiling": ceiling,
        "systems": {
            n: {"query_mean": means(s), "seed_mean": means(s, True)} for n, s in per.items()
        },
        "differences": [
            versus("rerank", "rrf"),
            versus("rrf", "bm25"),
            versus("bge_small_esci", "bge_small"),
        ],
        "grid": log,
    }
    path = REPO_ROOT / f"results/similar_{split}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(split, len(sets), "queries", len(seeds), "seeds; encoder", encoder, weights)
    for name, s in payload["systems"].items():
        print(f"{name:16s}", {m: round(v["mean"], 4) for m, v in s["query_mean"].items()})
    for d in payload["differences"]:
        print(
            d["system"],
            "-",
            d["baseline"],
            d["verdict"],
            {
                m: (round(d[m]["mean"], 4), round(d[m]["low"], 4), round(d[m]["high"], 4))
                for m in ("ndcg10", "condensed_ndcg10")
            },
        )


if __name__ == "__main__":
    main()
