"""Evaluate rewriting. Dev chooses and freezes the trigger; primary only reads it.

dev writes results/rewrite_dev.json and configs/frozen_rewrite.yaml (once). primary refuses until that
file is committed. LTR uses the out-of-fold models of the final run, so nothing is retrained.
"""

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import yaml

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.eval.claims import compare_with_p
from shelfrank.eval.harness import evaluate, summarise
from shelfrank.eval.splits import load_folds
from shelfrank.ltr.dataset import build_data
from shelfrank.ltr.first_stage import FirstStage, candidate_pairs
from shelfrank.ltr.model import to_run
from shelfrank.ltr.pipeline import Resources, load_resources, load_split, make_features, make_stage
from shelfrank.rewrite.fusion import with_rewrites
from shelfrank.rewrite.prompt import VERSION
from shelfrank.rewrite.trigger import Trigger, candidate_triggers, choose, query_signals
from shelfrank.stats.bootstrap import bootstrap_mean

FROZEN = REPO_ROOT / "configs/frozen_rewrite.yaml"


def load_rewrites(split: str) -> dict[int, list[str]]:
    """Cached rewrites by query id."""
    path = REPO_ROOT / f"data/labels/rewrites_wands_{split}_{VERSION}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(r["id"]): r["texts"] for r in rows}


def rewrite_runs(
    res: Resources, rewrites: dict[int, list[str]]
) -> tuple[dict[int, list[list[int]]], dict[int, list[list[int]]]]:
    """BM25 and dense rankings of every rewrite text."""
    flat = [(q, t) for q, texts in rewrites.items() for t in texts]
    bm25: dict[int, list[list[int]]] = {q: [] for q in rewrites}
    dense: dict[int, list[list[int]]] = {q: [] for q in rewrites}
    for start in range(0, len(flat), 200):
        part = flat[start : start + 200]
        stage = make_stage(res, list(range(len(part))), [t for _, t in part])
        for i, (q, _) in enumerate(part):
            bm25[q].append(stage.runs["bm25_multi"][i])
            dense[q].append(stage.runs["dense"][i])
    return bm25, dense


def ltr_run(res: Resources, stage: FirstStage, folds: Any) -> dict[int, list[int]]:  # noqa: ANN401
    """Rank with the saved out-of-fold models."""
    body = joblib.load(REPO_ROOT / "artifacts/final_rankers.joblib")
    pairs = candidate_pairs(stage, res.retrieval["candidate_depth"])
    data = build_data(pairs, make_features(res, stage, pairs), {q: {} for q in stage.query_ids})
    run: dict[int, list[int]] = {}
    for fold, texts in body["folds"].items():
        queries = [q for q in stage.query_ids if folds.assignment[q] == fold]
        if not queries:
            continue
        rows = data.rows(queries)
        table = data.features.iloc[rows][body["columns"]]
        scores = np.mean([lgb.Booster(model_str=t).predict(table) for t in texts], axis=0)
        run |= to_run(data, rows, scores)
    return run


def mixed(per: dict[str, Any], base: str, changed: str, fired: set[int]) -> dict[int, Any]:
    """Per-query scores with the changed system on fired queries only."""
    return {q: per[changed][q] if q in fired else per[base][q] for q in per[base]}


def main() -> None:
    """Run one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "primary"], required=True)
    split = parser.parse_args().split
    if split == "primary":
        command = ["git", "status", "--porcelain", "--", str(FROZEN)]
        dirty = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        if not FROZEN.exists() or dirty.strip():
            raise SystemExit("commit configs/frozen_rewrite.yaml first")
    boot = load_config("retrieval")["bootstrap"]
    folds = load_folds(REPO_ROOT / load_config("data")["splits"]["path"])
    res = load_resources(load_config("frozen")["retrieval"]["encoder"])
    ids, texts, qrels = load_split(split)
    stage = make_stage(res, ids, texts)
    rewrites = load_rewrites(split)
    start = time.perf_counter()
    bm25, dense = rewrite_runs(res, rewrites)
    k, depth = res.retrieval["rrf"]["k"], res.retrieval["run_depth"]
    wide = with_rewrites(stage, bm25, dense, k, depth)
    print(f"rewrite retrieval {time.perf_counter() - start:.0f}s", flush=True)
    per = {
        "ltr": evaluate(ltr_run(res, stage, folds), qrels),
        "ltr_all_rewritten": evaluate(ltr_run(res, wide, folds), qrels),
        "rrf": evaluate(stage.runs["hybrid_rrf"], qrels),
        "rrf_all_rewritten": evaluate(wide.runs["hybrid_rrf"], qrels),
    }
    if split == "dev":
        saved = json.loads((REPO_ROOT / "results/final_run.json").read_text(encoding="utf-8"))
        gap = max(
            abs(per["ltr"][q]["ndcg10"] - saved["per_query"]["ltr_full"][str(q)]["ndcg10"])
            for q in ids
        )
        print("largest difference of reloaded LTR from the final run", gap)
        if gap > 1e-9:
            raise SystemExit("the reloaded models do not reproduce the final run")
    signals = query_signals(stage, res.student, res.slot_model)

    def both(row: dict[str, Any]) -> float:
        return (float(row["ndcg10"]) + float(row["condensed_ndcg10"])) / 2

    gain = {q: both(per["ltr_all_rewritten"][q]) - both(per["ltr"][q]) for q in ids}
    log: list[dict[str, Any]] = []
    if split == "dev":
        trigger, log = choose(candidate_triggers(signals), signals, gain)
        if not FROZEN.exists():
            body = {
                "signal": trigger.signal,
                "threshold": trigger.threshold,
                "prompt": VERSION,
                "rule": "largest mean gain of LTR over all dev queries (mean of standard and condensed nDCG@10); "
                "fires on at most half; a tie goes to fewer rewrites",
                "families": {
                    "rewriting": [
                        "ltr with triggered rewriting vs ltr (primary)",
                        "rrf with triggered rewriting vs rrf",
                    ]
                },
            }
            FROZEN.write_text(
                "# Written once by scripts/evaluate_rewrite.py --split dev. Never edited.\n"
                + yaml.safe_dump(body, sort_keys=False, width=100),
                encoding="utf-8",
                newline="\n",
            )
    frozen = load_config("frozen_rewrite")
    trigger = Trigger(frozen["signal"], frozen["threshold"])
    fired = {q for q in ids if trigger.fires(signals[q])}
    per["ltr_triggered"] = mixed(per, "ltr", "ltr_all_rewritten", fired)
    per["rrf_triggered"] = mixed(per, "rrf", "rrf_all_rewritten", fired)
    # Oracles use labels. They are bounds and not systems.
    worst = set(sorted(ids, key=lambda q: (both(per["ltr"][q]), q))[: len(fired)])
    per["oracle_worst"] = mixed(per, "ltr", "ltr_all_rewritten", worst)
    helped = {q for q in ids if gain[q] > 0}
    per["oracle_helped"] = mixed(per, "ltr", "ltr_all_rewritten", helped)

    def versus(a: str, b: str, part: list[int]) -> dict[str, Any]:
        return {
            "system": a,
            "baseline": b,
            "n": len(part),
            **compare_with_p({q: per[a][q] for q in part}, {q: per[b][q] for q in part}, **boot),
        }

    slice_ids = sorted(fired)
    deltas = {
        q: float(per["ltr_all_rewritten"][q]["ndcg10"]) - float(per["ltr"][q]["ndcg10"])
        for q in slice_ids
    }
    ledger = [
        json.loads(line)
        for line in (REPO_ROOT / "results/llm_spend.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    calls = [
        r
        for r in ledger
        if r["batch"] == f"rewrite_wands_{split}_{VERSION}" and r["status"] == "ok"
    ]
    latency = [
        json.loads(line)
        for line in (REPO_ROOT / "results/rewrite_latency.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    times = [r["ms"] for r in latency if r["split"] == split]
    cost = sum(r["cost_inr"] for r in calls) / max(len(calls), 1)
    levels = {}
    for name, scores in per.items():
        summary = summarise(scores)
        levels[name] = {
            m: summary[m]["mean"]
            for m in ("ndcg10", "condensed_ndcg10", "judged10", "exact10_share")
        }
        levels[name]["interval"] = asdict(
            bootstrap_mean([float(v["ndcg10"]) for v in scores.values()], **boot)
        )
    exact_slice = {
        name: summarise({q: per[name][q] for q in slice_ids})["exact10_share"]["mean"]
        for name in ("ltr", "ltr_all_rewritten")
    }
    payload = {
        "split": split,
        "n": len(ids),
        "trigger": {"signal": trigger.signal, "threshold": trigger.threshold},
        "triggered": len(fired),
        "with_rewrite_text": sum(bool(rewrites[q]) for q in ids),
        "levels": levels,
        "overall": [versus("ltr_triggered", "ltr", ids), versus("rrf_triggered", "rrf", ids)],
        "slice": [
            versus("ltr_all_rewritten", "ltr", slice_ids),
            versus("rrf_all_rewritten", "rrf", slice_ids),
        ],
        "always": versus("ltr_all_rewritten", "ltr", ids),
        "oracle": [versus("oracle_worst", "ltr", ids), versus("oracle_helped", "ltr", ids)],
        "exact10_share_on_slice": exact_slice,
        "harm": {
            "worse": sum(d < -1e-9 for d in deltas.values()),
            "better": sum(d > 1e-9 for d in deltas.values()),
            "same": sum(abs(d) <= 1e-9 for d in deltas.values()),
            "worst": sorted(deltas.items(), key=lambda x: x[1])[:10],
            "best": sorted(deltas.items(), key=lambda x: -x[1])[:10],
        },
        "cost": {
            "calls": len(calls),
            "inr_per_call": cost,
            "inr_per_1000_rewritten": 1000 * cost,
            "inr_per_1000_queries": 1000 * cost * len(fired) / len(ids),
        },
        "latency_ms": {
            "n": len(times),
            "p50": float(np.percentile(times, 50)) if times else None,
            "p95": float(np.percentile(times, 95)) if times else None,
        },
        "trigger_log": log,
    }
    path = REPO_ROOT / f"results/rewrite_{split}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in ("trigger", "triggered", "levels", "harm", "cost", "latency_ms")
            },
            indent=1,
        )
    )
    for row in (*payload["overall"], *payload["slice"], payload["always"], *payload["oracle"]):
        print(
            row["system"],
            "-",
            row["baseline"],
            row["n"],
            row["verdict"],
            row["ndcg10"]["mean"],
            row["condensed_ndcg10"]["mean"],
        )


if __name__ == "__main__":
    main()
