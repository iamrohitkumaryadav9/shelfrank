"""Generated blocks of the README, built only from committed results files."""

import re
from typing import Any

from shelfrank.data.audit_report import pct
from shelfrank.stats.bootstrap import paired_difference


class MissingBlockError(KeyError):
    """The README has no marker pair for a generated block."""


def replace_block(text: str, name: str, body: str) -> str:
    """Replace what lies between the markers for ``name``, failing if the markers are missing."""
    begin, end = f"<!-- begin:{name} -->", f"<!-- end:{name} -->"
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if len(pattern.findall(text)) != 1:
        raise MissingBlockError(name)
    return pattern.sub(lambda _: f"{begin}\n{body.strip()}\n{end}", text)


def limitations(audit: dict[str, Any]) -> str:
    """State what the benchmark cannot show."""
    bias, primary = audit["pool_bias"], audit["ceilings_primary"]
    return f"""- **The judgment pool favours class match.** Wayfair's own query classifier helped choose
  which products were judged. A product in the query's class is judged
  {pct(bias["judged_share_in_query_class_mean"])} of the time, against
  {pct(bias["judged_share_overall"])} for a random pair, so class features can gain by retrieving judged
  products. Condensed nDCG@10 and judged@10 are reported beside every such claim.
- **Some metrics have low ceilings.** A perfect ranker reaches Exact@10 of
  {primary["exact_at_10"]:.3f} on the {primary["n_exact"]} primary queries that have an Exact product, and
  Recall@100 of {primary["recall_at_100"]:.3f} on {primary["n_relevant"]}.
- **The benchmark is small.** It has {audit["rows"]["queries"]} queries, and the headline uses
  {primary["n"]}.
- **Patterns have no slot.** Floral, striped, geometric, plaid and solid are out of scope in this version. The
  catalogue keeps them under their own keys and lists none of them as a style.
- **Finishes share a slot with materials.** "Brushed nickel" and "oak" are both material_finish, because too few
  queries name a finish to train a separate slot. Matching looks a value up in colour, finish and material
  keys together, so the result does not depend on the slot.
- **The taxonomy has gaps.** No blanket class reaches the 30-product cut, so a query for a blanket is labelled
  "Bedding Sets".
- **`query_class` is itself a model prediction**, so classifier accuracy means agreement with Wayfair's
  classifier."""


def findings(retrieval: dict[str, Any], tuning: dict[str, Any]) -> str:
    """Development findings that shape how later results must be read."""
    dense = retrieval["systems"]["dense_bge_small"]["versus_baseline"]
    pool = retrieval["judged_by_class"]["dense_bge_small"]
    held = tuning["held_out_per_query"]
    title = retrieval["systems"]["bm25_title"]["per_query"]
    gain = paired_difference(
        [held[q] for q in sorted(held)], [title[q]["ndcg10"] for q in sorted(held)]
    )
    weights = tuning["weights"]
    zero = sum(w["class"] == 0 for w in tuning["inner_weights"])
    standard, condensed = dense["ndcg10"], dense["condensed_ndcg10"]
    return f"""All on the {retrieval["n_queries"]} dev queries, exploratory and uncorrected.

- **The order of dense and BM25 depends on unjudged handling.** Against tuned BM25, off-the-shelf bge-small
  is {standard["mean"]:+.3f} [{standard["low"]:+.3f}, {standard["high"]:+.3f}] on standard nDCG@10 and
  {condensed["mean"]:+.3f} [{condensed["low"]:+.3f}, {condensed["high"]:+.3f}] on condensed nDCG@10. Its
  top-10 results outside the query's class are {pct(pool["out_of_class_judged"])} judged, against
  {pct(pool["in_class_judged"])} inside it, which lowers its standard score.
- **Tuning is optimistic, which is why results are held out.** Tuned field weights score
  {tuning["dev_ndcg10_tuned_in_sample"]:.3f} on the queries they were tuned on and
  {tuning["dev_ndcg10_held_out"]:.3f} on queries they were not. The held-out gain over title-only BM25 is
  {gain.mean:+.3f} [{gain.low:+.3f}, {gain.high:+.3f}].
- **The product class field has weight {weights["class"]:g}** once the category path has weight
  {weights["hierarchy"]:g}, and {zero} of {len(tuning["inner_weights"])} held-out searches agree. The likely
  reason is that the path carries the same signal."""
