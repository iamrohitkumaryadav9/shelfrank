"""README blocks that need the final results. Every number comes from results/*.json."""

from collections.abc import Mapping, Sequence
from typing import Any

from shelfrank.data.audit_report import table
from shelfrank.eval import final_report as fr
from shelfrank.eval.headline import earning_groups, headline, main_rows, secondary
from shelfrank.qu.slot_report import SILVER_LABEL

STAGES = {
    "dense_encode": "Dense encode (fp32 ONNX)",
    "bm25_and_dense_search": "BM25 and dense search, fusion",
    "query_understanding": "Query understanding",
    "rewrites": "Retrieval with cached rewrites",
    "feature_build": "Feature build (all candidates)",
    "ltr_score": "LTR score",
    "fuse": "Rank list",
    "fetch": "Fetch products (DuckDB)",
    "total": "**Total**",
}
RANKERS = {"ltr": "LTR", "rrf": "Hybrid RRF"}


def _gap(c: Mapping[str, float]) -> str:
    return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def pitch(data: Mapping[str, Any]) -> str:
    """The one-paragraph result, primary comparison first."""
    return f"{headline(data)}\n\n{secondary(data)}"


def key_results(data: Mapping[str, Any], checks: Mapping[str, Any]) -> str:
    """Main table, pre-registered comparisons and the caution that belongs to them."""
    row, _, _, _ = main_rows(data)
    dev, strong = checks["by_fold"]["0"], checks["against_fine_tuned_rrf"]
    return f"""{fr.main_table(data, "primary")}

{fr.comparison_table(data, "primary", "main", fr.NAMES)}

Primary queries are the {data["queries"]["primary"]} of folds 1 to 4. Settings were frozen before any of them was read. Read with it:

- **Quote the condensed number.** The standard gain is {row["ndcg10"]["mean"]:+.3f} and the gain that ignores unjudged products
  {row["condensed_ndcg10"]["mean"]:+.3f}. The difference is what unjudged handling can explain.
- **The dev fold does not show it:** {_gap(dev["ndcg10"])} standard and {_gap(dev["condensed_ndcg10"])} condensed there. Every other fold does. The result was
  checked for leaks before it was believed ([docs/RESULTS.md](docs/RESULTS.md), section 2).
- **Against a stronger baseline** that was not pre-registered, hybrid RRF with the fine-tuned encoder, LTR is {_gap(strong["ndcg10"])} standard and
  {_gap(strong["condensed_ndcg10"])} condensed."""


def ablation(data: Mapping[str, Any]) -> str:
    """Feature groups against the full model."""
    earned = ", ".join(earning_groups(data)) or "none"
    return f"""{fr.ablation_table(data, "primary")}

Groups whose removal is a loss after Holm: **{earned}**. The attribute features (G4) are not among them. G3, G4 and G6 have unadjusted
intervals below 0 that do not survive correction."""


def understanding(
    class_dev: Mapping[str, Any],
    ceilings: Mapping[str, Mapping[str, Any]],
    slots: Mapping[str, Any],
) -> str:
    """Class and slot results."""
    rows = []
    for split in ("dev", "primary", "all"):
        c = ceilings[split]
        f1 = (
            "n/a"
            if not c["student"]["macro_f1_classes"]
            else f"{c['student']['macro_f1']:.3f} ({c['student']['macro_f1_classes']} classes)"
        )
        rows.append(
            [
                split,
                c["n"],
                f"{c['student']['accuracy']:.3f} [{c['student']['low']:.3f}, {c['student']['high']:.3f}]",
                f"{c['student']['top3']:.3f}",
                f"{c['student']['department']:.3f}",
                f1,
                f"{c['teacher']['accuracy']:.3f} [{c['teacher']['low']:.3f}, {c['teacher']['high']:.3f}]",
            ]
        )
    header = [
        "Split",
        "n",
        "Student accuracy",
        "Top-3",
        "Department",
        "Macro-F1",
        "Gemini zero-shot accuracy",
    ]
    best = class_dev["best"][class_dev["selected"]]
    slot_rows = [
        [
            split,
            slots[split]["n"],
            slots[split]["queries_with_a_slot"],
            *(
                f"{slots[split][m]['all']['f1']:.3f} [{slots[split][m]['all']['low']:.3f}, {slots[split][m]['all']['high']:.3f}]"
                for m in ("student", "lexicon")
            ),
        ]
        for split in ("dev", "primary", "all")
    ]
    return f"""**Product class**, as agreement with Wayfair's assigned query class. The student is character n-grams with a linear model, trained on
product names and {class_dev["seed_training_queries"]:,} teacher-labelled ESCI queries, and takes {best["latency_ms"]["p50"]:.1f} ms per query on CPU.

{table(header, rows)}

**Attribute slots**, exact-span F1 over color, material_finish, style and room, **{SILVER_LABEL}**.

{table(["Split", "n", "With a slot", "Student (word tagger)", "Lexicon matcher"], slot_rows)}

Details: [docs/class_dev.md](docs/class_dev.md), [docs/slots_dev.md](docs/slots_dev.md)."""


def retrieval(data: Mapping[str, Any], dense: Mapping[str, Any]) -> str:
    """Retrieval comparisons and what the dev rule decided."""
    verdict = dense["differences"][0]["verdict"]
    return f"""{fr.comparison_table(data, "primary", "retrieval", fr.NAMES)}

On dev the fine-tuned encoder showed "{verdict}" as a dense retriever, so by a rule written before measuring, the ranker kept the off-the-shelf
encoder. On the primary queries it is better. int8 lost {abs(dense["serving"]["bge_small"]["formats"]["int8"]["ndcg10"] - dense["serving"]["bge_small"]["formats"]["pytorch"]["ndcg10"]):.3f} nDCG@10 on dev, more than the tolerance of 0.005, so the demo uses fp32 ONNX
([docs/dense_dev.md](docs/dense_dev.md), [docs/phase3_checks.md](docs/phase3_checks.md))."""


def rewriting(primary: Mapping[str, Any] | None) -> str:
    """Gain, harm and cost of rewriting."""
    if primary is None:
        return "Not evaluated yet: the paid batch is waiting for approval. The code, trigger search and tests are in place."
    overall, part = primary["overall"][0], primary["slice"][0]
    ps = fr.adjusted(primary["overall"])
    harm, cost, ms = primary["harm"], primary["cost"], primary["latency_ms"]
    return f"""Gemini rewrites a query when `{primary["trigger"]["signal"]}` looks hard. The trigger was chosen on dev and fires on {primary["triggered"]} of {primary["n"]} primary
queries.

- All queries, LTR with triggered rewriting against LTR: {_gap(overall["ndcg10"])} standard, {_gap(overall["condensed_ndcg10"])} condensed, Holm p {ps[0]:.4f}:
  **{fr.claimed(overall, ps[0])}**.
- Triggered queries only: {_gap(part["ndcg10"])} standard, {_gap(part["condensed_ndcg10"])} condensed. {harm["worse"]} got worse, {harm["better"]} better, {harm["same"]} stayed the same.
- Rs {cost["inr_per_1000_rewritten"]:.1f} per 1,000 rewritten queries. API time p50 {ms["p50"]:.0f} ms, p95 {ms["p95"]:.0f} ms.

Harm analysis and bounds: [docs/rewriting.md](docs/rewriting.md)."""


def suggestions(data: Mapping[str, Any]) -> str:
    """Autocomplete."""
    parts = data["prefix_words"]
    rows = [
        [
            f"{n} word{'s' if n != '1' else ''}",
            c["n"],
            f"{c['mrr']['frequency_only']['mean']:.3f}",
            f"{c['mrr']['ngram']['mean']:.3f}",
            _gap(c["ngram_minus_frequency"]),
        ]
        for n, c in parts.items()
    ]
    return f"""{table(["Typed", "n", "Most frequent words, MRR", "N-gram over the query trie, MRR", "Difference"], rows)}

There are no query counts in the data, so nothing here is popularity. A class prior was tried and its cross-validated weight is {data["class_weight"]:g}
([docs/suggest.md](docs/suggest.md))."""


def similar(data: Mapping[str, Any]) -> str:
    """Item-to-item substitutes."""
    from shelfrank.similar.report import NAMES, adjusted

    rows = [
        [
            NAMES[n],
            *(
                f"{s['query_mean'][m]['mean']:.3f}"
                for m in ("ndcg10", "condensed_ndcg10", "recall10", "mrr")
            ),
        ]
        for n, s in data["systems"].items()
    ]
    gaps = [
        [
            f"{NAMES[d['system']]} vs {NAMES[d['baseline']]}",
            _gap(d["ndcg10"]),
            _gap(d["condensed_ndcg10"]),
            f"{p:.4f}",
            d["verdict"],
        ]
        for d, p in zip(data["differences"], adjusted(data), strict=True)
    ]
    return f"""{data["n_seeds"]:,} seed products from {data["n_queries"]} primary queries. Products labelled Exact for the same query are substitutes of each other.

{table(["System", "nDCG@10", "Condensed", "Recall@10", "MRR"], rows)}

{table(["Comparison", "nDCG@10", "Condensed nDCG@10", "Holm p", "Verdict"], gaps)}

This measures substitutability, not engagement or complementarity ([docs/similar.md](docs/similar.md))."""


def latency(
    data: Mapping[str, Any],
    before: Mapping[str, Any] | None = None,
    fast: Mapping[str, Any] | None = None,
) -> str:
    """Latency budget, with the LTR timings before the product-side precomputation."""
    rows = []
    for stage, label in STAGES.items():
        cells = []
        if before is not None:
            c = before["stages"]["ltr"].get(stage)
            cells += [f"{c['p50']:.1f}", f"{c['p95']:.1f}"] if c else ["", ""]
        for ranker in RANKERS:
            c = data["stages"][ranker].get(stage)
            cells += [f"{c['p50']:.1f}", f"{c['p95']:.1f}"] if c else ["", ""]
        if any(cells):
            rows.append([label, *cells])
    names = (["LTR before"] if before is not None else []) + list(RANKERS.values())
    header = ["Stage (ms)", *(f"{name} {p}" for name in names for p in ("p50", "p95"))]
    agree = data["agreement_with_evaluation"]
    identity = ""
    if before is not None and fast is not None:
        identity = (
            f" Tokens and attribute values of every product are now computed when the data is built, not per search. Rankings stay "
            f"bit-identical, ids and scores, for {fast['identical']} of {fast['queries']} WANDS queries at depth {fast['depth']}, and peak memory went from "
            f"{before['memory_mb']:.0f} to {data['memory_mb']:.0f} MB."
        )
    return f"""{table(header, rows)}

{data["repeats"]} passes over {data["queries"]} dev queries, one at a time, on the development laptop ({data["machine"].split(",")[0]}, no GPU). Start-up takes {data["load_seconds"]:.1f} s once the
files are downloaded, the first search {data["first_query_seconds"]:.1f} s, and peak memory is {data["memory_mb"]:.0f} MB.{identity} The API and demo return the evaluated LTR
top 10 exactly for {agree["ltr"]["identical_top10"]:.1%} of the {agree["ltr"]["n"]} WANDS queries ({agree["ltr"]["mean_top10_overlap"]:.1%} of the products). The rest differ by rounding of half-precision
product vectors."""


def findings_final(
    data: Mapping[str, Any],
    checks: Mapping[str, Any],
    rewriting_primary: Mapping[str, Any] | None,
    speed: Mapping[str, Any],
) -> str:
    """What a reader should know first, all from results files."""
    dev = checks["by_fold"]["0"]
    curve = checks["training_size"]
    first, last = curve[min(curve, key=int)], curve[max(curve, key=int)]
    rows = {
        r["system"]: (r, p)
        for r, p in zip(
            data["comparisons"]["primary"]["ablation"],
            fr.adjusted(data["comparisons"]["primary"]["ablation"]),
            strict=True,
        )
    }
    g4, g4p = rows["without_G4"]
    point, pointp = rows["pointwise"]
    retrieval = {
        (r["system"], r["baseline"]): (r, p)
        for r, p in zip(
            data["comparisons"]["primary"]["retrieval"],
            fr.adjusted(data["comparisons"]["primary"]["retrieval"]),
            strict=True,
        )
    }
    tuned, tunedp = retrieval[("dense_esci", "dense")]
    shuffled = checks["shuffled_labels"]
    row, _, _, _ = main_rows(data)
    earned = " and ".join(earning_groups(data)) or "no group"
    ltr, rrf = speed["stages"]["ltr"]["total"], speed["stages"]["rrf"]["total"]
    lines = [
        f"- **Dev showed no gain; the primary queries do.** On the dev fold LTR is {_gap(dev['ndcg10'])} against hybrid RRF ({_gap(dev['condensed_ndcg10'])} condensed). "
        f"On the {data['queries']['primary']} primary queries it is {_gap(row['ndcg10'])}. Checks for a leak found none: the run repeats exactly, shuffled labels "
        f"drop LTR to {_gap(shuffled['ndcg10'])}, every other fold gains, and the gain does not grow with similarity to training queries. A known cause "
        "of the gap: BM25's field weights were tuned on dev, which makes the dev baselines optimistic.",
        f"- **More training queries, more gain.** {_gap(first['ndcg10'])} when trained on {min(curve, key=int)} queries, {_gap(last['ndcg10'])} on {max(curve, key=int)}. "
        "Dev experiments trained on about 72 queries, which may be part of why they did not show it.",
        f"- **{earned} earned it; the attribute features did not.** Without G4 the change is {_gap(g4['ndcg10'])}, Holm p {g4p:.3f}: no claimed effect.",
        f"- **LambdaMART against pointwise regression:** {_gap(point['ndcg10'])} for pointwise, Holm p {pointp:.3f}, {fr.claimed(point, pointp)}.",
        f"- **The price of pre-registration.** On the primary queries the fine-tuned encoder beats off-the-shelf by {_gap(tuned['ndcg10'])} (Holm p {tunedp:.3f}). "
        "On dev it did not, so the rule written before measuring kept it out of the ranker. It stays out: changing the ranker now would be choosing on the test set.",
        f"- **Latency.** LTR answers in {ltr['p50']:.0f} ms at the median ({ltr['p95']:.0f} ms p95) against {rrf['p50']:.0f} ms ({rrf['p95']:.0f} ms) for hybrid RRF.",
    ]
    if rewriting_primary is not None:
        overall = rewriting_primary["overall"][0]
        p = fr.adjusted(rewriting_primary["overall"])[0]
        lines.append(
            f"- **LLM rewriting did not help.** It gained on dev, where its trigger was chosen. On the primary queries LTR with triggered rewriting is "
            f"{_gap(overall['ndcg10'])} ({_gap(overall['condensed_ndcg10'])} condensed): {fr.claimed(overall, p)}. The demo shows rewrites but does not apply them by default."
        )
    return "\n".join(lines)


def spend(rows: Sequence[Mapping[str, Any]]) -> str:
    """Total LLM spend from the ledger. A reservation is replaced by the result of its call."""
    last: dict[str, float] = {}
    for r in rows:
        last[r["call_id"]] = float(r["cost_inr"])
    by: dict[str, float] = {}
    purpose = {r["call_id"]: r["purpose"] for r in rows}
    for call, cost in last.items():
        by[purpose[call]] = by.get(purpose[call], 0.0) + cost
    parts = ", ".join(f"{k} Rs {v:.2f}" for k, v in sorted(by.items()))
    return f"LLM spend, whole project: **Rs {sum(last.values()):.2f}** of a Rs 500 limit in code ({parts}). Costs include a 24% margin for tax and card fees. Every response is cached and committed, so results rebuild at no cost."


MAPPING = (
    (
        "Query understanding",
        "Class and slot students distilled from an LLM teacher",
        "#query-understanding",
    ),
    (
        "Query rewriting",
        "Serving-time trigger, harm analysis, cost per 1,000 queries",
        "#query-rewriting",
    ),
    (
        "Search suggestions",
        "Prefix completion, evaluated without popularity data",
        "#search-suggestions",
    ),
    ("Ranking", "LambdaMART over six feature groups, pre-registered ablation", "#ranking"),
    (
        "Retrieval",
        "BM25, dense, contrastive fine-tuning on ESCI, fusion",
        "#retrieval-and-fine-tuning",
    ),
    ("Content recommendations", "Item-to-item substitutes from human labels", "#more-like-this"),
    (
        "Evaluation with business metrics",
        "Exact@10 by department, pool bias, leak checks",
        "docs/RESULTS.md#4-in-business-terms",
    ),
    (
        "Experiment design",
        "A/B plan with guardrails and roll-back rules",
        "docs/RESULTS.md#7-plan-for-an-ab-test",
    ),
    ("Production", "FastAPI service, Docker image, latency budget", "#production-slice"),
    (
        "Personalization",
        "Design note: why it cannot be evaluated here, and what to build",
        "docs/PERSONALIZATION.md",
    ),
)


def mapping() -> str:
    """How the project maps to search work."""
    return table(
        ["Area", "What is here", "Results"], [[a, b, f"[link]({c})"] for a, b, c in MAPPING]
    )
