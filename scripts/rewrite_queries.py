"""Rewrite queries with Gemini, one call per query as at serving time. Offline unless --live.

Pilot:  .venv\\Scripts\\python.exe scripts\\rewrite_queries.py --split dev --pilot --live --projected X
"""

import argparse
import json
import time

from shelfrank.config import REPO_ROOT
from shelfrank.llm.budget import PILOT_ITEMS
from shelfrank.llm.setup import make_client, make_request
from shelfrank.llm.standing import reasons_to_ask
from shelfrank.ltr.pipeline import load_split
from shelfrank.rewrite.prompt import MAX_OUTPUT, SCHEMA, VERSION, parse, rewrite_prompt


def main() -> None:
    """Pilot or batch. Texts stay in the cache and the output file holds rewrites by query id."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "primary"], required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--projected", type=float)
    args = parser.parse_args()
    ids, texts, _ = load_split(args.split)
    total = len(ids)
    if args.pilot:
        ids, texts = ids[:PILOT_ITEMS], texts[:PILOT_ITEMS]
    client = make_client(live=args.live)
    name = f"rewrite_wands_{args.split}_{VERSION}"
    rows, timings = [], []
    for q, text in zip(ids, texts, strict=True):
        request = make_request(
            client.config, rewrite_prompt(text), SCHEMA, MAX_OUTPUT, rules_version=VERSION
        )
        paid = client.cache.get(request) is None
        start = time.perf_counter()
        response = client.generate(request, "rewriting", name, parse, args.pilot)
        if paid:
            timings.append({"id": q, "ms": 1000 * (time.perf_counter() - start)})
        rows.append({"id": q, "texts": parse(response.text).texts(text), "rules": VERSION})
    ledger = client.budget.ledger
    spent = ledger.spent(batch=name)
    log = REPO_ROOT / "results/rewrite_latency.jsonl"
    with log.open("a", encoding="utf-8", newline="\n") as handle:
        for row in timings:
            handle.write(json.dumps({"split": args.split, **row}) + "\n")
    print(f"{name}: {len(rows)} queries, batch spend Rs {spent:.4f}")
    if args.pilot:
        per_item = spent / len(rows)
        projection = per_item * total
        print(f"pilot Rs {per_item:.4f} per query; {total} project to Rs {projection:.2f}")
        if args.projected:
            reasons = reasons_to_ask(
                args.projected,
                per_item,
                projection,
                ledger.spent("rewriting"),
                client.budget.caps["rewriting"],
                ledger.spent(),
            )
            print("standing approval:", "STOP, " + "; ".join(reasons) if reasons else "holds")
    else:
        path = REPO_ROOT / f"data/labels/rewrites_wands_{args.split}_{VERSION}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("wrote", path.name)
    print(client.budget.summary())


if __name__ == "__main__":
    main()
