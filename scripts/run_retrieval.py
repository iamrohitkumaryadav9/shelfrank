"""Evaluate every first-stage retriever and candidate set, and write results/retrieval_<split>.json."""

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from shelfrank.bm25.fields import BM25Index, combine, concat_fields, field_tokens, score_matrix
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import (
    judgments,
    load_labels,
    load_products,
    load_queries,
    split_classes,
)
from shelfrank.dense.encoder import encode_cached, load_sentence_transformer, product_texts
from shelfrank.eval.access import queries_for
from shelfrank.eval.crosscheck import max_differences
from shelfrank.eval.harness import candidate_report, ceilings, evaluate, summarise, to_qrels
from shelfrank.eval.metrics import METRIC_NAMES
from shelfrank.eval.pool import judged_by_class
from shelfrank.eval.splits import load_folds
from shelfrank.ranking import rank_all, reciprocal_rank_fusion, union
from shelfrank.stats.bootstrap import bootstrap_mean, paired_difference

BASELINE = "bm25_multi"


def main() -> None:
    """Run every system on one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "primary", "all"])
    split = parser.parse_args().split

    data_cfg, cfg = load_config("data"), load_config("retrieval")
    raw = REPO_ROOT / data_cfg["wands"]["raw_dir"]
    products = load_products(raw).sort_values("product_id").reset_index(drop=True)
    ids = products["product_id"].to_numpy()
    query_ids = queries_for(split, load_folds(REPO_ROOT / data_cfg["splits"]["path"]))
    queries = load_queries(raw).set_index("query_id").loc[query_ids]
    texts = queries["query"].tolist()
    qrels = to_qrels(judgments(load_labels(raw)), query_ids)
    depth = cfg["run_depth"]

    k1, b = cfg["bm25"]["k1"], cfg["bm25"]["b"]
    weights = json.loads((REPO_ROOT / "results/bm25_tuning.json").read_text())["weights"]
    tokens = field_tokens(products)
    matrices = {f: score_matrix(BM25Index(tokens[f], k1, b), texts) for f in weights}
    title_desc = score_matrix(
        BM25Index(concat_fields(tokens, ["name", "description"]), k1, b), texts
    )

    dense_cfg = cfg["dense"]["bge_small"]
    identity = {k: dense_cfg[k] for k in ("model", "revision", "max_seq_length")}
    encode = load_sentence_transformer(**identity)
    vectors = encode_cached(
        product_texts(products), REPO_ROOT / "artifacts/embeddings/bge_small", identity, encode
    )
    started = time.perf_counter()
    query_vectors = np.stack([encode([dense_cfg["query_prefix"] + t])[0] for t in texts])
    encode_ms = 1000 * (time.perf_counter() - started) / len(texts)

    def bm25_run(matrix: np.ndarray) -> dict[int, list[int]]:
        return rank_all(matrix, query_ids, ids, depth, positive_only=True)

    runs = {
        "bm25_title": bm25_run(matrices["name"]),
        "bm25_title_desc": bm25_run(title_desc),
        "bm25_multi_equal": bm25_run(combine(matrices, {f: 1.0 for f in weights})),
        BASELINE: bm25_run(combine(matrices, weights)),
        "dense_bge_small": rank_all(query_vectors @ vectors.T, query_ids, ids, depth),
    }
    runs["hybrid_rrf"] = {
        q: reciprocal_rank_fusion(
            [runs[BASELINE][q], runs["dense_bge_small"][q]], cfg["rrf"]["k"], depth
        )
        for q in query_ids
    }

    boot = cfg["bootstrap"]
    per_query = {name: evaluate(run, qrels) for name, run in runs.items()}
    systems = {}
    for name, scores in per_query.items():
        summary = summarise(scores)
        for metric in METRIC_NAMES:
            values = [row[metric] for row in scores.values() if row[metric] is not None]
            interval = bootstrap_mean(values, **boot)
            summary[metric] |= {"low": interval.low, "high": interval.high}
        versus = {}
        for metric in METRIC_NAMES:
            pairs = [
                (scores[q][metric], per_query[BASELINE][q][metric])
                for q in query_ids
                if scores[q][metric] is not None
            ]
            ours, theirs = zip(*pairs, strict=True)
            versus[metric] = asdict(paired_difference(ours, theirs, **boot))
        systems[name] = {"summary": summary, "versus_baseline": versus, "per_query": scores}

    query_class = queries["query_class"].to_dict()
    product_classes = dict(zip(ids, products["product_class"].map(split_classes), strict=True))
    n = cfg["candidate_depth"]
    candidates = {
        "bm25_title": {q: runs["bm25_title"][q][:n] for q in query_ids},
        BASELINE: {q: runs[BASELINE][q][:n] for q in query_ids},
        "dense_bge_small": {q: runs["dense_bge_small"][q][:n] for q in query_ids},
        "hybrid_rrf": {q: runs["hybrid_rrf"][q][:n] for q in query_ids},
        "union_multi_dense": {
            q: union([runs[BASELINE][q], runs["dense_bge_small"][q]], n) for q in query_ids
        },
    }
    payload = {
        "split": split,
        "n_queries": len(query_ids),
        "baseline": BASELINE,
        "run_depth": depth,
        "candidate_depth": n,
        "weights": weights,
        "query_encode_ms_mean": encode_ms,
        "ceilings": ceilings(qrels),
        "ranx_max_difference": {
            name: max_differences(runs[name], qrels, per_query[name]) for name in runs
        },
        "candidates": {k: candidate_report(v, qrels) for k, v in candidates.items()},
        "judged_by_class": {
            name: judged_by_class(run, qrels, query_class, product_classes)
            for name, run in runs.items()
        },
        "systems": systems,
    }
    path = REPO_ROOT / "results" / f"retrieval_{split}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    worst = max(v for d in payload["ranx_max_difference"].values() for v in d.values())
    print(f"wrote {path.name}; largest difference from ranx {worst:.2e}")
    for name, system in systems.items():
        print(name, {m: round(s["mean"], 4) for m, s in system["summary"].items()})


if __name__ == "__main__":
    main()
