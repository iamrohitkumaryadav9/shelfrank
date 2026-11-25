# Evaluation protocol

Nothing is pre-registered yet. The final version is `configs/frozen.yaml`, tagged `preregistered-v1`
before the one out-of-fold run.

## Labels

- Gains: Exact 2, Partial 1, Irrelevant 0.
- Repeated judgments are reduced to one per pair by majority vote, and a tie takes the lower grade.
- Primary metrics treat unjudged products as irrelevant. Condensed metrics (unjudged removed) and
  judged@10 are reported beside them for every system.
- Rankers train on judged pairs only.

## Splits

- `data/splits/wands_folds.json`: 5 folds of 96 queries, seed 20260. The script refuses to overwrite it.
- Stratified by `query_class` where a class has at least 5 queries. The rest are pooled.
- Queries whose stemmed token sets have Jaccard similarity of 0.75 or more share a fold (4 pairs).
- Fold 0 is the dev fold. Folds 1 to 4 (384 queries) give the primary result.
- `query_class` is used to stratify folds, to describe the data, and as the reference for evaluating
  our classifier. It is never a model input.

## Metrics

- Primary: nDCG@10. Secondary: Exact@10, Recall@100 (Exact or Partial), MRR of the first Exact.
- A query with no relevant product scores 0 for every system and stays in the mean, matching ranx and
  keeping the query count fixed.
- Exact@10, MRR and Recall@100 cannot reach 1 on this data. Their ceilings are in
  [data_audit.md](data_audit.md) section 2 and are reported next to the values.

## What has been looked at

This log records any exposure to queries outside the dev fold.

| Phase | Exposure |
| --- | --- |
| 0 | Aggregate statistics over all 480 queries (label counts, class counts, query lengths), as published in data_audit.md |
| 0 | Text and class of query ids 0 to 4 and 483 to 487, seen in the first rows of the raw file and the output of WANDS's own sample notebook |
| 0 | The example queries printed in the WANDS annotation guidelines and paper |
| 0 | Taxonomy coverage over all 480 queries was visible, but the threshold rule reads only catalogue counts |

No feature, prompt or hyperparameter has been designed yet.

## Guards

`tests/test_leakage.py` checks fold sizes and disjointness, near-duplicate grouping, that the CI fixture
holds dev queries only, that the taxonomy is rebuilt from the catalogue alone, and that no primary query
of four or more words appears in committed files. Each later phase adds its derived datasets to this file.
