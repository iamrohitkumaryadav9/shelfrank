"""Build app_data from the frozen artifacts. Needs the final run and the WANDS files."""

import json
import shutil
from typing import Any

import joblib
import numpy as np
import pandas as pd

from shelfrank.app_support.build import Inputs, build
from shelfrank.app_support.char_tfidf import PlainClassifier
from shelfrank.bm25.fields import BM25Index, concat_fields
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.claims import GAIN
from shelfrank.eval.final_report import ALPHA, adjusted
from shelfrank.eval.splits import load_folds
from shelfrank.features.priors import prior_features
from shelfrank.ltr.pipeline import load_resources, load_split
from shelfrank.qu.slot_model import load_slot_model
from shelfrank.rewrite.prompt import VERSION
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

TARGET = REPO_ROOT / "app_data"
TOP = 10
DEPTH = 50  # enough for the top 10 after re-ranking, and half the work of the evaluation's 100


def substitutes(res: Any, frozen: dict[str, Any], shown: list[int]) -> dict[int, list[int]]:  # noqa: ANN401
    """Top substitutes of every product, with the frozen settings."""
    cfg = load_config("retrieval")
    ids = res.products["product_id"].to_numpy()
    row_of = {int(p): i for i, p in enumerate(ids)}
    index = BM25Index(
        concat_fields(res.tokens, ["name", "class", "description"]),
        cfg["bm25"]["k1"],
        cfg["bm25"]["b"],
    )
    vectors = np.load(REPO_ROOT / "artifacts/embeddings" / frozen["encoder"] / "embeddings.npy")
    weights = RerankWeights(**frozen["weights"])
    table = prior_features(pd.DataFrame({"pi": range(len(ids))}), res.products)
    prior = rating_prior(
        table["g5_shrunk_rating"].to_numpy(),
        res.products["review_count"].astype(float).fillna(0.0).to_numpy(),
    )
    k = cfg["rrf"]["k"]
    out: dict[int, list[int]] = {}
    for start in range(0, len(shown), 2000):
        part = shown[start : start + 2000]
        tokens = {
            p: [*res.tokens["name"][row_of[p]], *res.tokens["class"][row_of[p]]] for p in part
        }
        a = bm25_rankings(index, tokens, ids, DEPTH)
        b = dense_rankings(vectors, {p: row_of[p] for p in part}, ids, DEPTH)
        fused = fuse(a, b, k, DEPTH)
        rows = sorted(
            {row_of[c] for run in fused.values() for c in run} | {row_of[p] for p in part}
        )
        listed = res.listed_for(rows)
        for p in part:
            i = row_of[p]
            terms = {
                c: (
                    shared_class(res.classes[i], res.classes[row_of[c]]),
                    agreement(listed[i], listed[row_of[c]]),
                    float(prior[row_of[c]]),
                )
                for c in fused[p]
            }
            out[p] = rerank(p, fused[p], k, (a[p], b[p]), weights, terms)[:TOP]
        print(f"substitutes {min(start + 2000, len(shown)):,} / {len(shown):,}", flush=True)
    return out


def main() -> None:
    """Collect and build."""
    frozen = load_config("frozen")
    res = load_resources(frozen["retrieval"]["encoder"])
    cfg = load_config("data")
    folds = load_folds(REPO_ROOT / cfg["splits"]["path"])
    ids, _, qrels = load_split("all")
    frame = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id")
    final = json.loads((REPO_ROOT / "results/final_run.json").read_text(encoding="utf-8"))
    rows = final["comparisons"]["primary"]["main"]
    primary = next(i for i, r in enumerate(rows) if r["baseline"] == "hybrid_rrf")
    wins = rows[primary]["verdict"] == GAIN and adjusted(rows)[primary] < ALPHA
    rankers = joblib.load(REPO_ROOT / "artifacts/final_rankers.joblib")
    rewrites: dict[int, list[str]] = {}
    for split in ("dev", "primary"):
        path = REPO_ROOT / f"data/labels/rewrites_wands_{split}_{VERSION}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                rewrites[int(row["id"])] = row["texts"]
    trigger_path = REPO_ROOT / "configs/frozen_rewrite.yaml"
    trigger = load_config("frozen_rewrite") if trigger_path.exists() else None
    tagger = load_slot_model()
    seed = (REPO_ROOT / "data/esci/esci_seed_queries.txt").read_text(encoding="utf-8").splitlines()
    rewrite_path = REPO_ROOT / "results/rewrite_primary.json"
    apply = False
    if rewrite_path.exists():
        result = json.loads(rewrite_path.read_text(encoding="utf-8"))["overall"]
        apply = result[0]["verdict"] == GAIN and adjusted(result)[0] < ALPHA
    inputs = Inputs(
        products=res.products,
        tokens=res.tokens,
        vectors=res.vectors,
        classifier=PlainClassifier.from_student(res.student),
        tagger=tagger,  # type: ignore[arg-type]
        rankers=rankers["folds"],
        columns=rankers["columns"],
        departments=res.departments,
        slot_of=res.slot_of,
        bm25_weights=frozen["retrieval"]["bm25_weights"],
        k1=frozen["retrieval"]["k1"],
        b=frozen["retrieval"]["b"],
        rrf_k=frozen["retrieval"]["rrf_k"],
        run_depth=frozen["retrieval"]["run_depth"],
        candidate_depth=frozen["retrieval"]["candidate_depth"],
        default_ranker="ltr" if wins else "rrf",
        similar=substitutes(res, frozen["similar"], [int(p) for p in res.products["product_id"]]),
        queries=[(q, str(frame.loc[q, "query"]), folds.assignment[q]) for q in ids],
        grades=qrels,
        suggest_queries=seed,
        rewrites=rewrites,
        trigger={"signal": trigger["signal"], "threshold": trigger["threshold"]}
        if trigger
        else None,
        apply_rewrites=apply,
    )
    build(TARGET, inputs)
    shutil.copyfile(
        REPO_ROOT / "artifacts/models/bge_small_esci/tokenizer.json", TARGET / "tokenizer.json"
    )
    total = 0
    for path in sorted(TARGET.iterdir()):
        total += path.stat().st_size
        print(f"{path.name:24s} {path.stat().st_size / 1e6:8.1f} MB")
    print(f"total {total / 1e6:.1f} MB; default ranker {'ltr' if wins else 'rrf'}")


if __name__ == "__main__":
    main()
