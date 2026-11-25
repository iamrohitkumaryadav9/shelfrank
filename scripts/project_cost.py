"""Print the projected labelling cost. Makes no API call."""

from shelfrank.config import load_config
from shelfrank.llm.pricing import Batch, project

# Planning assumptions, to be replaced by counted tokens once the prompts exist. The class prompt
# lists 211 class names; its token figure is an estimate on the high side.
PLAN = [
    Batch("2a prompt iteration (100 ESCI dev, 3 rounds)", 100, 1800, 12, 20, repeats=3),
    Batch("2a class labels (2,000 ESCI seed)", 2000, 1800, 12, 20),
    Batch("2a zero-shot ceiling (480 WANDS)", 480, 1800, 12, 20),
    Batch("2b prompt iteration (100 ESCI dev, 3 rounds)", 100, 1200, 12, 60, repeats=3),
    Batch("2b slot labels (2,000 ESCI seed)", 2000, 1200, 12, 60),
    Batch("2b silver slots (480 WANDS)", 480, 1200, 12, 60),
]


def main() -> None:
    """Show each batch and the total, for each priced model."""
    config = load_config("llm")
    models = list(config["models"])
    print(
        f"{'batch':48s} {'calls':>5s} {'in':>8s} {'out':>8s} "
        + " ".join(f"{m[7:]:>16s}" for m in models)
    )
    totals = dict.fromkeys(models, 0.0)
    for batch in PLAN:
        costs = {m: project(config, batch, m) for m in models}
        for model, cost in costs.items():
            totals[model] += cost
        cells = " ".join(f"Rs {costs[m]:13.2f}" for m in models)
        print(
            f"{batch.name:48s} {batch.calls:5d} {batch.input_tokens:8,d} {batch.output_tokens:8,d} {cells}"
        )
    print(f"{'total':72s} " + " ".join(f"Rs {totals[m]:13.2f}" for m in models))
    print(f"cap: Rs {config['caps_inr']['labelling']}")


if __name__ == "__main__":
    main()
