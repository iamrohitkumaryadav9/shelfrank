"""Label a set of queries with the teacher. Offline unless --live is given.

Pilot:  .venv\\Scripts\\python.exe scripts\\label.py slots esci_dev --pilot --live
Full:   .venv\\Scripts\\python.exe scripts\\label.py slots esci_dev --live   (needs an approval)
"""

import argparse
import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_queries
from shelfrank.eval.access import queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.llm.batch import project, run_batch
from shelfrank.llm.budget import PILOT_ITEMS
from shelfrank.llm.setup import make_client, make_request
from shelfrank.qu import teacher
from shelfrank.qu.guidelines import CLASS_VERSION, SLOT_VERSION, SLOTS, require_final
from shelfrank.qu.slot_vocab import in_vocabulary, load_vocabulary


def load_source(source: str) -> list[tuple[int, str]]:
    """Return (id, query). WANDS access goes through the split lock."""
    if source == "wands_dev":
        cfg = load_config("data")
        ids = queries_for("dev", load_folds(REPO_ROOT / cfg["splits"]["path"]))
        texts = load_queries(REPO_ROOT / cfg["wands"]["raw_dir"]).set_index("query_id")["query"]
        return [(q, str(texts[q])) for q in ids]
    name = {"esci_dev": "esci_prompt_dev.txt", "esci_seed": "esci_seed_queries.txt"}[source]
    lines = (REPO_ROOT / "data/esci" / name).read_text(encoding="utf-8").splitlines()
    return list(enumerate(lines))


def main() -> None:
    """Run a pilot or a batch, save the labels, and report cost."""
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["class", "slots"])
    parser.add_argument("source", choices=["esci_dev", "esci_seed", "wands_dev"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--live", action="store_true", help="allow paid calls")
    args = parser.parse_args()

    version = CLASS_VERSION if args.task == "class" else SLOT_VERSION
    require_final(version)
    rows = load_source(args.source)
    everything = len(rows)
    if args.pilot:
        rows = rows[:PILOT_ITEMS]
    taxonomy = json.loads((REPO_ROOT / "data/taxonomy.json").read_text(encoding="utf-8"))["classes"]
    raw = json.loads((REPO_ROOT / "data/slot_vocab.json").read_text(encoding="utf-8"))
    union, styles, _ = load_vocabulary(raw)
    client = make_client(live=args.live)

    def build(group: list[str]):  # noqa: ANN202
        if args.task == "class":
            prompt, schema = teacher.class_prompt(group, taxonomy), teacher.CLASS_SCHEMA
            limit = teacher.CLASS_MAX_OUTPUT
        else:
            prompt, schema = teacher.slot_prompt(group), teacher.SLOT_SCHEMA
            limit = teacher.slot_output_limit(args.source, len(group))
        return make_request(client.config, prompt, schema, limit, rules_version=version)

    def parse(text: str, group: list[str]) -> list:
        if args.task == "class":
            return teacher.parse_classes(text, group, taxonomy)
        return teacher.parse_slots(text, group)

    name = f"{args.task}_{args.source}_{version}"
    queries = [query for _, query in rows]

    def run(items: list[str], size: int) -> list:
        return run_batch(
            client,
            name,
            "labelling",
            items,
            size,
            build,
            lambda group: lambda text: parse(text, group),
            parse,
            pilot=args.pilot,
        )

    if args.pilot:
        labels = run(queries, PILOT_ITEMS)
    elif args.source == "esci_dev":
        labels = run(queries, teacher.CHUNK)
    else:
        # The pilot group is repeated exactly, so it comes from the cache and is not paid twice.
        labels = run(queries[:PILOT_ITEMS], PILOT_ITEMS) + run(queries[PILOT_ITEMS:], teacher.CHUNK)
    spent = client.budget.ledger.spent(batch=name)
    print(f"{name}: {len(rows)} items, batch spend Rs {spent:.4f}")
    if args.pilot:
        out = project(client, name, everything, teacher.CHUNK, PILOT_ITEMS)
        print(
            f"pilot Rs {out.inr_per_item:.4f} per item; {everything} items project to Rs {out.projected_inr:.2f}"
        )
        for (_, query), label in zip(rows, labels, strict=True):
            # ESCI only: a WANDS query is never printed, not even a dev one.
            if args.source != "wands_dev":
                print(
                    f"  {query!r:60s} {label if args.task == 'class' else {k: v for k, v in label.items() if v}}"
                )
    else:
        path = REPO_ROOT / "data/labels" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for (row_id, _), label in zip(rows, labels, strict=True):
                row = {"id": row_id, "label": label, "rules": version}
                if args.task == "slots":
                    row["label"] = {s: label[s] for s in SLOTS}
                    row["invented"] = label["invented"]
                    row["out_of_vocabulary"] = [
                        [s, span]
                        for s in SLOTS
                        for span in label[s]
                        if not in_vocabulary(s, span, union, styles)
                    ]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {path.name}")
    print(client.budget.summary())


if __name__ == "__main__":
    main()
