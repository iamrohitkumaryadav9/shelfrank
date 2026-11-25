"""Primary queries where LTR loses most to BM25, with what each ranked. Writes results/failure_cases.json.

Run after the final evaluation. Written diagnoses live in docs/failure_diagnoses.json.
"""

import json

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.data.wands import load_products
from shelfrank.ltr.pipeline import load_split
from shelfrank.qu.slot_model import load_slot_model

N = 20
LABELS = {2: "Exact", 1: "Partial", 0: "Irrelevant", None: "unjudged"}


def main() -> None:
    """Collect the cases."""
    final = json.loads((REPO_ROOT / "results/final_run.json").read_text(encoding="utf-8"))
    ranks = json.loads((REPO_ROOT / "results/final_rankings.json").read_text(encoding="utf-8"))
    ids, texts, qrels = load_split("primary")
    text = dict(zip(ids, texts, strict=True))
    raw = REPO_ROOT / load_config("data")["wands"]["raw_dir"]
    products = load_products(raw).set_index("product_id")
    slots = load_slot_model()
    ltr, bm25 = final["per_query"]["ltr_full"], final["per_query"]["bm25_multi"]
    gap = {q: ltr[str(q)]["ndcg10"] - bm25[str(q)]["ndcg10"] for q in ids}
    cases = []
    for q in sorted(ids, key=lambda q: (gap[q], q))[:N]:
        case = {
            "query_id": q,
            "query": text[q],
            "department": final["departments"][str(q)],
            "query_class": final["query_class"][str(q)],
            "slots": {s: v for s, v in slots.slots(text[q]).items() if v},
            "judged": len(qrels[q]),
            "exact": sum(g == 2 for g in qrels[q].values()),
            "relevant": sum(g >= 1 for g in qrels[q].values()),
            "ndcg10": {
                n: final["per_query"][n][str(q)]["ndcg10"]
                for n in ("bm25_multi", "hybrid_rrf", "dense", "ltr_full")
            },
            "condensed": {
                n: final["per_query"][n][str(q)]["condensed_ndcg10"]
                for n in ("bm25_multi", "ltr_full")
            },
            "judged10": {
                n: final["per_query"][n][str(q)]["judged10"] for n in ("bm25_multi", "ltr_full")
            },
            "top": {},
        }
        for system in ("bm25_multi", "ltr_full"):
            case["top"][system] = [
                {
                    "name": str(products.loc[p, "product_name"]),
                    "class": str(products.loc[p, "product_class"]),
                    "label": LABELS[qrels[q].get(p)],
                }
                for p in ranks["top20"][system][str(q)][:10]
            ]
        cases.append(case)
    path = REPO_ROOT / "results/failure_cases.json"
    counts = {str(q): sum(len(v) for v in slots.slots(text[q]).values()) for q in ids}
    body = json.dumps({"n": N, "cases": cases, "slot_values": counts}, indent=1, ensure_ascii=False)
    path.write_text(body + "\n", encoding="utf-8", newline="\n")
    for c in cases:
        print(
            f"\n[{c['query_id']}] {c['query']!r} class={c['query_class']} slots={c['slots']} judged={c['judged']} exact={c['exact']} rel={c['relevant']}"
        )
        print(
            "   ndcg",
            {k: round(v, 2) for k, v in c["ndcg10"].items()},
            "cond",
            {k: round(v, 2) for k, v in c["condensed"].items()},
            "judged10",
            c["judged10"],
        )
        for system in ("bm25_multi", "ltr_full"):
            print(
                "  ",
                system,
                " | ".join(f"{r['name'][:32]}<{r['label'][0]}>" for r in c["top"][system][:7]),
            )


if __name__ == "__main__":
    main()
