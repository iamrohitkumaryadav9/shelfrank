"""Is the gain real? Controls on the final ranker. Writes results/leak_checks.json.

Made after the final run and because its result was larger than dev suggested. Nothing here selects
or changes anything: the frozen configuration is used as it is.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_final import config_of
from shelfrank.config import REPO_ROOT, load_config
from shelfrank.eval.claims import compare
from shelfrank.eval.harness import evaluate
from shelfrank.eval.splits import load_folds
from shelfrank.ltr.dataset import RankingData
from shelfrank.ltr.model import Ranker, to_run
from shelfrank.ltr.pipeline import load_everything
from shelfrank.text.near_duplicates import jaccard, leakage_keys

SIZES = (48, 96, 192)
SEED = 20260


def shuffled(data: RankingData, seed: int) -> RankingData:
    """Grades permuted within each query, so every feature loses its relation to relevance."""
    rng = np.random.default_rng(seed)
    grade = data.grade.copy()
    for q in np.unique(data.query_id):
        rows = np.flatnonzero(data.query_id == q)
        grade[rows] = grade[rng.permutation(rows)]
    return RankingData(data.query_id, data.product_id, data.features, grade)


def main() -> None:
    """Run every control."""
    frozen = load_config("frozen")
    boot = frozen["evaluation"]["bootstrap"]
    folds = load_folds(REPO_ROOT / load_config("data")["splits"]["path"])
    data, stage, qrels, texts = load_everything(frozen["retrieval"]["encoder"], "all")
    config = config_of(frozen)
    ids = sorted(qrels)
    per: dict[str, Any] = {"rrf": evaluate(stage.runs["hybrid_rrf"], qrels)}

    def out_of_fold(name: str, source: RankingData, size: int | None) -> None:
        run: dict[int, list[int]] = {}
        for fold in range(folds.n_folds):
            held = folds.queries(fold)
            train = [q for q in ids if q not in set(held)]
            if size:
                train = sorted(random.Random(SEED + fold).sample(train, size))
            ranker = Ranker(config).fit(source, train)
            rows = data.rows(held)
            run |= to_run(data, rows, ranker.scores(data, rows))
        per[name] = evaluate(run, qrels)
        out = compare(
            {q: per[name][q] for q in folds.primary},
            {q: per["rrf"][q] for q in folds.primary},
            **boot,
        )
        print(
            name, out["ndcg10"]["mean"], out["condensed_ndcg10"]["mean"], out["verdict"], flush=True
        )

    out_of_fold("full", data, None)
    out_of_fold("shuffled_labels", shuffled(data, SEED), None)
    for size in SIZES:
        out_of_fold(f"train_{size}", data, size)

    saved = json.loads((REPO_ROOT / "results/final_run.json").read_text(encoding="utf-8"))[
        "per_query"
    ]
    reproduced = max(
        abs(per["full"][q]["ndcg10"] - saved["ltr_full"][str(q)]["ndcg10"]) for q in ids
    )
    per["rrf_esci"] = {q: saved["hybrid_rrf_esci"][str(q)] for q in ids}
    per["ltr_other_encoder"] = {q: saved["ltr_other_encoder"][str(q)] for q in ids}

    keys = {q: leakage_keys(texts[q]) for q in ids}
    nearest = {}
    for q in ids:
        others = [o for o in ids if folds.assignment[o] != folds.assignment[q]]
        nearest[q] = max(jaccard(a, b) for o in others for a in keys[q] for b in keys[o])
    primary = folds.primary
    gain = {q: per["full"][q]["ndcg10"] - per["rrf"][q]["ndcg10"] for q in primary}
    order = sorted(primary, key=lambda q: (nearest[q], q))
    thirds = [order[i * len(order) // 3 : (i + 1) * len(order) // 3] for i in range(3)]

    def versus(a: str, b: str, part: list[int]) -> dict[str, Any]:
        return {
            "n": len(part),
            **compare({q: per[a][q] for q in part}, {q: per[b][q] for q in part}, **boot),
        }

    payload = {
        "reproduces_final_run": reproduced,
        "by_fold": {str(f): versus("full", "rrf", folds.queries(f)) for f in range(folds.n_folds)},
        "shuffled_labels": versus("shuffled_labels", "rrf", primary),
        "training_size": {str(s): versus(f"train_{s}", "rrf", primary) for s in SIZES}
        | {"384": versus("full", "rrf", primary)},
        "training_size_dev_fold": {str(s): versus(f"train_{s}", "rrf", folds.dev) for s in SIZES}
        | {"384": versus("full", "rrf", folds.dev)},
        "by_similarity_to_training": [
            {"low": nearest[part[0]], "high": nearest[part[-1]], **versus("full", "rrf", part)}
            for part in thirds
        ],
        "queries_with_similarity_at_least_half": sum(nearest[q] >= 0.5 for q in primary),
        "gain_without_those": versus("full", "rrf", [q for q in primary if nearest[q] < 0.5]),
        "correlation_gain_similarity": float(
            np.corrcoef([gain[q] for q in primary], [nearest[q] for q in primary])[0, 1]
        ),
        "against_fine_tuned_rrf": versus("full", "rrf_esci", primary),
        "fine_tuned_ltr_against_fine_tuned_rrf": versus("ltr_other_encoder", "rrf_esci", primary),
    }
    path = REPO_ROOT / "results/leak_checks.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
