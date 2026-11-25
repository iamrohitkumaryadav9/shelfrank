"""Run a 10-item pilot of the teacher on the ESCI prompt-development slice."""

import argparse
import json

from shelfrank.config import REPO_ROOT
from shelfrank.data.taxonomy import OTHER
from shelfrank.llm.batch import project, run_batch
from shelfrank.llm.budget import PILOT_ITEMS
from shelfrank.llm.setup import make_client, make_request
from shelfrank.qu import teacher
from shelfrank.qu.guidelines import CLASS_VERSION, SLOT_VERSION, require_final

SIZES = {"prompt dev (100)": 100, "seed (2,000)": 2000, "WANDS (480)": 480}


def main() -> None:
    """Pilot one task, show its answers and project the batches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["class", "slots"])
    parser.add_argument("--version", default="v2")
    parser.add_argument("--live", action="store_true", help="allow paid calls")
    args = parser.parse_args()

    version = CLASS_VERSION if args.task == "class" else SLOT_VERSION
    if args.live:
        require_final(version)
    taxonomy = json.loads((REPO_ROOT / "data/taxonomy.json").read_text(encoding="utf-8"))["classes"]
    path = REPO_ROOT / "data/esci/esci_prompt_dev.txt"
    queries = path.read_text(encoding="utf-8").splitlines()[:PILOT_ITEMS]
    client = make_client(live=args.live, max_attempts=1)

    def build(group: list[str]):  # noqa: ANN202
        if args.task == "class":
            prompt, schema = teacher.class_prompt(group, taxonomy), teacher.CLASS_SCHEMA
            limit = teacher.CLASS_MAX_OUTPUT
            return make_request(client.config, prompt, schema, limit, rules_version=version)
        prompt, schema = teacher.slot_prompt(group), teacher.SLOT_SCHEMA
        limit = teacher.SLOT_MAX_OUTPUT
        return make_request(client.config, prompt, schema, limit, rules_version=version)

    def parse(text: str, group: list[str]) -> list:
        if args.task == "class":
            return teacher.parse_classes(text, group, taxonomy)
        return teacher.parse_slots(text, group)

    name = f"{args.task}-pilot-{args.version}"
    answers = run_batch(
        client,
        name,
        "labelling",
        queries,
        PILOT_ITEMS,
        build,
        lambda group: lambda text: parse(text, group),
        parse,
        pilot=True,
    )
    for query, answer in zip(queries, answers, strict=True):
        shown = answer if args.task == "class" else {k: v for k, v in answer.items() if v}
        print(f"  {query!r:62s} {shown}")
    calls = [c for c in client.budget.ledger.calls().values() if c.batch == name]
    print(
        "tokens in/out/thinking:",
        [(c.input_tokens, c.output_tokens, c.thinking_tokens) for c in calls],
    )
    for label, items in SIZES.items():
        out = project(client, name, items, teacher.CHUNK, PILOT_ITEMS)
        print(
            f"{label:18s} Rs {out.inr_per_item:.4f} per pilot item, projected Rs {out.projected_inr:.2f}"
        )
    if args.task == "class":
        print(
            "outside the taxonomy:", sum(a is None for a in answers), "other:", answers.count(OTHER)
        )
    print(client.budget.summary())


if __name__ == "__main__":
    main()
