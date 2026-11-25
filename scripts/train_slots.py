"""Train the slot students, compare them with the lexicon, and select one by the pre-set rule.

Writes results/slots_dev.json, artifacts/slot_model.joblib and configs/slot_student.yaml.
Every reference here is the same LLM teacher: silver labels, not human ones.
"""

import json
import time
from dataclasses import asdict
from typing import Any

import joblib
import numpy as np

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.access import queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.qu.class_data import HELD_SHARE, SEED
from shelfrank.qu.class_student import split_seed
from shelfrank.qu.slot_lexicon import span_scores
from shelfrank.qu.slot_model import (
    CONFIG,
    LEXICON,
    TAGGER,
    SlotModel,
    TaggerConfig,
    WordTagger,
    labelled_slots,
    load_lexicon,
    load_lexicon_data,
)
from shelfrank.stats.bootstrap import bootstrap_mean

GRID = [TaggerConfig(x, w, c) for x in (False, True) for w in (1, 2) for c in (0.3, 1.0, 3.0, 10.0)]
MODEL = "artifacts/slot_model.joblib"


def f1_interval(truth: list[Any], predicted: list[Any]) -> dict[str, float]:
    """Bootstrap interval of the micro-F1, resampling queries."""
    counts = []
    for t, p in zip(truth, predicted, strict=True):
        s = span_scores([t], [p])["all"]
        hits = 0.0 if not s["support"] or not s["predicted"] else s["recall"] * s["support"]
        counts.append((hits, s["predicted"], s["support"]))
    table = np.asarray(counts, dtype=float)
    picks = np.random.default_rng(SEED).integers(0, len(table), (10_000, len(table)))
    sums = table[picks].sum(axis=1)
    scores = 2 * sums[:, 0] / np.maximum(sums[:, 1] + sums[:, 2], 1)
    low, high = np.percentile(scores, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def main() -> None:
    """Train, score, select, save."""
    cfg = load_config("data")
    ids = queries_for("dev", load_folds(REPO_ROOT / cfg["splits"]["path"]))
    frame = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id").loc[ids]
    dev = frame["query"].tolist()
    silver = labelled_slots("wands_dev")
    dev_truth = [silver[q] for q in ids]
    lines = (REPO_ROOT / "data/esci/esci_seed_queries.txt").read_text(encoding="utf-8").splitlines()
    seed = labelled_slots("esci_seed")
    train, held = split_seed(len(lines), HELD_SHARE, SEED)
    held_texts, held_truth = [lines[i] for i in held], [seed[i] for i in held]
    lexicon = load_lexicon_data()

    def score(model: SlotModel) -> dict[str, Any]:
        times, out = [], []
        for query in dev:
            start = time.perf_counter()
            out.append(model.slots(query))
            times.append(1000 * (time.perf_counter() - start))
        held_out = [model.slots(q) for q in held_texts]
        row: dict[str, Any] = {
            "silver": span_scores(dev_truth, out),
            "esci": span_scores(held_truth, held_out),
            "latency_ms": {
                "p50": float(np.percentile(times, 50)),
                "p95": float(np.percentile(times, 95)),
            },
        }
        row["silver_all_slots_right"] = float(
            np.mean(
                [
                    span_scores([t], [p])["all"]["f1"] == 1.0
                    or not any(t.values())
                    and not any(p.values())
                    for t, p in zip(dev_truth, out, strict=True)
                ]
            )
        )
        row["silver"]["all"] |= f1_interval(dev_truth, out)
        row["esci"]["all"] |= f1_interval(held_truth, held_out)
        row["selection"] = (row["silver"]["all"]["f1"] + row["esci"]["all"]["f1"]) / 2
        return row

    runs = [{"kind": LEXICON, "config": None, **score(load_lexicon())}]
    models: list[WordTagger | None] = [None]
    for config in GRID:
        model = WordTagger(config, lexicon)
        model.fit([lines[i] for i in train], [seed[i] for i in train])
        runs.append({"kind": TAGGER, "config": asdict(config), **score(model)})
        models.append(model)
    for run in runs:
        print(
            run["kind"],
            run["config"],
            f"silver {run['silver']['all']['f1']:.3f}",
            f"esci {run['esci']['all']['f1']:.3f}  selection {run['selection']:.4f}",
        )
    # max keeps the first on a tie, and the lexicon is first.
    best = max(range(len(runs)), key=lambda i: runs[i]["selection"])
    best_student = max(range(1, len(runs)), key=lambda i: runs[i]["selection"])
    chosen = models[best]
    if chosen is not None:
        joblib.dump(chosen.to_plain(), REPO_ROOT / MODEL)
    text = "# Written by scripts/train_slots.py, by the rule in docs/PROTOCOL.md section 6.\n"
    text += f"kind: {runs[best]['kind']}\n"
    if chosen is not None:
        text += f"path: {MODEL}\nconfig: {json.dumps(runs[best]['config'])}\n"
    CONFIG.write_text(text, encoding="utf-8", newline="\n")
    agree = []
    matcher = load_lexicon()
    for q in dev:
        a = matcher.slots(q)
        b = (chosen or matcher).slots(q)
        agree.append(float(a == b))
    payload = {
        "reference": "LLM silver labels; human verification pending",
        "selection_rule": "mean of micro-F1 against dev silver and held-out ESCI; a tie goes to the lexicon",
        "n_dev": len(dev),
        "n_esci_train": len(train),
        "n_esci_held": len(held),
        "queries_with_a_slot": {
            "dev": sum(any(t.values()) for t in dev_truth),
            "esci_held": sum(any(t.values()) for t in held_truth),
        },
        "selected": best,
        "best_student": best_student,
        "dev_queries_where_student_and_lexicon_agree": bootstrap_mean(agree, seed=SEED).mean,
        "runs": runs,
    }
    path = REPO_ROOT / "results/slots_dev.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("selected", runs[best]["kind"], runs[best]["config"])


if __name__ == "__main__":
    main()
