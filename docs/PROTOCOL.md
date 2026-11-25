# Evaluation protocol

The rules below are fixed. The remaining choices (features, hyperparameters, candidate size, rewrite
trigger) go into `configs/frozen.yaml`, tagged `preregistered-v1` before the one out-of-fold run.

## 1. Labels

- Gains: Exact 2, Partial 1, Irrelevant 0, used linearly. The paper's 1, 0.5 and 0 give the same nDCG.
- One label per (query, product). Exact repeats are collapsed. Where repeats disagree, the majority
  label wins and a tie takes the lower label. All disagreements and their resolutions are listed in
  [data_audit.md](data_audit.md) section 1.
- Primary metrics treat unjudged products as irrelevant.
- Rankers train on judged pairs only.

## 2. Splits

- `data/splits/wands_folds.json`: 5 folds of 96 queries, seed 20260. The script refuses to overwrite it.
- Stratified by `query_class` where a class has at least 5 queries. The rest are pooled.
- Queries whose stemmed token sets have Jaccard similarity of 0.75 or more share a fold (4 pairs).
- Fold 0 is the dev fold. Folds 1 to 4 (384 queries) give the primary result.
- `query_class` stratifies folds, buckets results for analysis, and is the reference for the classifier.
  It is never a model input.

## 3. Metrics

| Metric | Averaged over | Primary n | All n |
| --- | --- | --- | --- |
| **nDCG@10** (primary metric) | all queries | 384 | 480 |
| Condensed nDCG@10 (unjudged removed) | all queries | 384 | 480 |
| Judged@10 | all queries | 384 | 480 |
| Recall@100, Exact or Partial | queries with a relevant product | 383 | 479 |
| Exact@10 | queries with an Exact product | 305 | 379 |
| Exact@10 as a share of its ceiling | queries with an Exact product | 305 | 379 |
| MRR of the first Exact | queries with an Exact product | 305 | 379 |
| Recall@100, Exact only | queries with an Exact product | 305 | 379 |

- **Every table states its n** and shows the perfect-ranker ceiling beside Exact@10, Recall@100 and
  Exact-only Recall@100.
- Exact@10 share of ceiling, for one query, is the number of Exact results in the top 10 divided by
  min(10, number of Exact products). It is averaged over queries.
- A query with no relevant product scores nDCG 0 for every system and stays in the mean. This matches
  ranx and keeps n fixed.
- The Exact@10 ceiling depends on which queries are averaged. Over all 384 primary queries it is 0.607.
  Over the 305 that have an Exact product, which is how the metric is now defined, it is 0.764. Only the
  second is used from here on.
- Metric code is our own, unit-tested by hand, and must agree with ranx to 1e-6.

## 4. Breakdowns

- **By department**, the main breakdown. A class belongs to the most common first level of
  `category_hierarchy` among its products. Only first levels with at least 100 catalogue products count
  as departments, which removes navigation paths such as "Sale". Everything else, including queries with
  no class or with a class that no product has, is "Other". The mapping reads the catalogue only.
- **By class**, only for classes with at least 10 primary queries.
- Departments with fewer than 10 queries are flagged as too small to interpret.

## 5. Query classifier

- Taxonomy: the 211 classes with at least 30 products, plus "other", from the catalogue alone.
- Headline: accuracy, worded as **agreement with Wayfair's assigned query class**, because that label is
  itself a model prediction.
- Also reported: top-3 accuracy and department-level accuracy.
- Secondary: macro-F1 over classes with at least 5 queries, with the class count stated.

## 6. Attribute slots

- Room values come from category paths plus a small hand list. The hand list is written from
  catalogue-side text only (category names, product names, feature values), never from queries.
- Room support (the number of room mentions in the 100 verified queries) is shown beside its F1. Below
  10, the F1 is reported as not interpretable.
- Whether the room slot stays is decided by the ablation.
- A product with several values for a slot is a family of variants. A conflict means that **none** of
  its listed values matches the query.

## 7. Pool bias

The judgment pool was partly selected with Wayfair's query classifier ([DATA.md](DATA.md),
[data_audit.md](data_audit.md) section 8). Class features can therefore gain by retrieving judged
products. These rules apply:

- Every claim about G3 (class features), and the overall LTR-versus-BM25 claim, shows condensed nDCG@10
  and judged@10 beside the standard numbers.
- To be copied into `configs/frozen.yaml` verbatim: "A G3 gain is claimed only if it is positive under
  BOTH standard and condensed evaluation with the 95% CI excluding 0. Otherwise it is reported as
  possibly pool-driven."
- Phase 4 reports the judged fraction of the LTR top 10 against the BM25 top 10, per department.

## 8. Ranking model

- `label_gain` defaults to [0, 1, 2], matching the metric. [0, 1, 3] is a dev-fold candidate only. The
  choice goes into `configs/frozen.yaml`.

## 9. Statistics

- Paired bootstrap over queries, 10,000 resamples, seeded. Mean difference with a 95% CI.
- The comparison family is declared in `configs/frozen.yaml` and Holm-corrected.

## 10. Published baseline

The paper plots BM25 at roughly 0.72 nDCG@10 with unstated settings. Ours is reported on the same queries
next to that sentence. It is a visual sanity check: BM25 is never tuned toward it, and no reproduction
is claimed.

## 11. Business metric

Exact@10 as a share of its achievable ceiling, over queries with at least one Exact product. Plain Exact@10
would be misleading, because 101 of 480 queries have no Exact product and a perfect ranker still falls
well short of 1.

## 12. Tool substitutions

| Planned | Used | Why |
| --- | --- | --- |
| pytrec_eval | ranx | pytrec_eval has no Windows wheel and needs a C++ compiler |
| shap | LightGBM `pred_contrib` (exact TreeSHAP) | Smart App Control blocks a DLL that shap imports |
| Local ONNX export | ONNX and optimum export inside the Kaggle notebook | Avoids more compiled wheels that Smart App Control may block |

Smart App Control is left on.

## 13. What has been looked at

This log records any exposure to queries outside the dev fold. If a non-dev query ever shapes a decision,
it is flagged at the next gate.

| Phase | Exposure | Shaped a decision? |
| --- | --- | --- |
| 0 | Aggregate statistics over all 480 queries, as published in data_audit.md | Yes: metric definitions and breakdown level (sections 3 and 4), which are the same for every system |
| 0 | Text and class of query ids 0 to 4 and 483 to 487, in the raw file preview and WANDS's sample notebook | No |
| 0 | The example queries printed in the WANDS annotation guidelines and paper | No |
| 0 | Taxonomy coverage over all 480 queries was visible; the rule reads only catalogue counts | No |

## 14. Guards

`tests/test_leakage.py` checks fold sizes and disjointness, near-duplicate grouping, that the CI fixture
holds dev queries only, that the taxonomy is rebuilt from the catalogue alone, and that no primary query
of four or more words appears in committed files. Each later phase adds its derived datasets to this file.
