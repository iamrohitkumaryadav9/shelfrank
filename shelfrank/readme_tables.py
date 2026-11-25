"""Generated blocks of the README, built only from committed results files."""

import re
from typing import Any

from shelfrank.data.audit_report import pct


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
- **`query_class` is itself a model prediction**, so classifier accuracy means agreement with Wayfair's
  classifier."""
