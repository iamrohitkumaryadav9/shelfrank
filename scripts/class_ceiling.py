"""Gemini zero-shot product classes, the ceiling for the distilled student. No API call.

Reads the committed labels. Writes results/class_ceiling_<split>.json.
"""

import argparse
import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.departments import department_map
from shelfrank.data.wands import load_products, load_queries
from shelfrank.eval.access import queries_for
from shelfrank.eval.splits import load_folds
from shelfrank.ltr.class_for_ranking import class_model
from shelfrank.qu.class_student import evaluate_classes, ranked_labels, to_reference
from shelfrank.qu.guidelines import CLASS_VERSION
from shelfrank.stats.bootstrap import bootstrap_mean, paired_difference


def main() -> None:
    """Compare teacher and student on one split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "primary", "all"])
    split = parser.parse_args().split
    cfg = load_config("data")
    raw = REPO_ROOT / cfg["wands"]["raw_dir"]
    products = load_products(raw)
    taxonomy = json.loads((REPO_ROOT / cfg["taxonomy"]["path"]).read_text(encoding="utf-8"))[
        "classes"
    ]
    ids = queries_for(split, load_folds(REPO_ROOT / cfg["splits"]["path"]))
    frame = load_queries(raw).set_index("query_id").loc[ids]
    truth = [to_reference(c, taxonomy) for c in frame["query_class"]]
    labels: dict[int, str | None] = {}
    for name in ("dev", "primary"):
        path = REPO_ROOT / f"data/labels/class_wands_{name}_{CLASS_VERSION}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                labels[int(row["id"])] = row["label"]
    teacher = [[str(labels[q])] for q in ids]
    student = class_model(products, taxonomy)
    top = ranked_labels(student.probabilities(frame["query"].tolist()), student.classes, 3)
    departments = department_map(products)
    seeded_path = json.loads((REPO_ROOT / "results/class_dev.json").read_text(encoding="utf-8"))
    out = {"split": split, "n": len(ids), "reference": "Wayfair's assigned query class"}
    right = {}
    for name, predictions in (("teacher", teacher), ("student", top)):
        scores = evaluate_classes(truth, predictions, departments, None)
        if name == "teacher":
            scores.pop("top3")  # the teacher gives one label
        right[name] = [float(t == p[0]) for t, p in zip(truth, predictions, strict=True)]
        interval = bootstrap_mean(right[name])
        out[name] = scores | {
            "low": interval.low,
            "high": interval.high,
            "predicted_other": sum(p[0] == "other" for p in predictions),
        }
    gap = paired_difference(right["student"], right["teacher"])
    out["student_minus_teacher"] = {"mean": gap.mean, "low": gap.low, "high": gap.high}
    out["agreement_student_teacher"] = sum(
        a[0] == b[0] for a, b in zip(teacher, top, strict=True)
    ) / len(ids)
    out["reference_other"] = truth.count("other")
    out["selected_arm"] = seeded_path["selected"]
    path = REPO_ROOT / f"results/class_ceiling_{split}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
