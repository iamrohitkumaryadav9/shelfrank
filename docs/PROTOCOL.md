# Evaluation protocol

The rules below are fixed. The remaining choices (features, hyperparameters, candidate size, rewrite
trigger) go into `configs/frozen.yaml`, tagged `preregistered-v1` before the one out-of-fold run.

## 1. Labels

- Gains: Exact 2, Partial 1, Irrelevant 0, used linearly. The paper's 1, 0.5 and 0 give the same nDCG.
- One label per (query, product). Exact repeats are collapsed. Where repeats disagree, the majority
  label wins and a tie takes the lower label. All disagreements and their resolutions are listed in
  [data_audit.md](data_audit.md) section 1.
- Primary metrics treat unjudged products as irrelevant.
- What rankers train on is chosen on the dev fold between two variants (section 8).

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
- **Selection:** every class and slot student is chosen on the dev fold and on held-out ESCI seed queries.
  Before `preregistered-v1`, only dev numbers are reported. Results on all 480 queries come after it.
- The Gemini zero-shot ceiling runs on the 96 dev queries first and on the other 384 after the freeze.

## 6. Attribute slots

- Slots: color, material_finish, style, room. The rules are `shelfrank/qu/guidelines.py`, and every label
  carries their version.
- **Every slot value must exist in the catalogue's attribute vocabulary** (at least 3 products). G4 matching is
  key-agnostic: a color or material_finish value is looked up across the union of colour, finish and
  material keys. Finishes have no slot of their own because too few queries mention one.
- Room rule: a room word is part of the product type when it lies inside a category name and the name
  continues past it. Category names are classes with at least 30 products and path segments with at least
  100. Patio, garden, balcony and deck count as outdoor. **The threshold of 100 was chosen while looking at 7
  examples, so 7 of 7 is not independent evidence.** The rule stays as slots-v1 with **no human check before the
  freeze** (pre-freeze change below). Round B, if it is done, reports agreement with it.
- If the rules change, labels are paid for again under the new version. The old ones stay in the cache.
- Room values come from category paths plus a small hand list. The hand list is written from
  catalogue-side text only (category names, product names, feature values), never from queries.
- Room support (the number of room mentions in the reference labels) is shown beside its F1.
  Below 10, the F1 is reported as not interpretable.
- Whether the room slot stays is decided by the ablation.
- A product with several values for a slot is a family of variants. A conflict means that **none** of
  its listed values matches the query.

### Verification by hand, and the pre-freeze change of 2026-11-26

The wording of the first two points was clarified after the freeze, on 2026-11-26. No rule changed, and the frozen text is at the tag
`preregistered-v1`.

- The first human labelling pass was discarded because the rater misapplied the slot rules: product words were labelled as style.
  The tool was revised and the rater retrained.
- **Round A was then dropped before the freeze, to save rater time.** It could not have been blind anyway, because its tool showed the
  teacher's answers. This is a protocol change made before the freeze. Consequences:
  - No human slot label exists when the slot student is selected, when G4 is enabled, or at the freeze.
  - Gemini's slot labels on the 96 dev queries are called **silver** everywhere.
  - The room rule has no human check.
- **Slot selection rule, set before any student was trained:** the mean of two exact-span F1 scores over all slots, one
  against silver labels on the 96 dev queries and one against teacher labels on held-out ESCI seed queries (a fifth of
  the seed, not trained on). The candidates are the lexicon matcher and every student configuration. The highest
  score is used for G4, and an exact tie goes to the lexicon, which is simpler.
- Both references come from the same teacher, so the rule measures how well a student imitates Gemini.
- **Round B is a blind check, optional, and comes after the freeze.** It has 50 held-out queries from folds 1 to 4, a seeded draw committed as
  ids in `data/splits/round_b_draw.json`. Until it is complete, every slot F1 is labelled **"vs LLM silver labels;
  human verification pending"**. If it is completed, the human number becomes primary in the README. Nothing in the
  ranking pipeline depends on it.
- Round B labelling is unlocked by practice on 15 invented queries with rule-derived answers (at least 13 right). The
  tool refuses words that are not in the query and a word used in two slots. It shows no teacher answer and has no
  code path that could. Labels are stored with timestamps.
- Round B reports: agreement with silver overall and per slot with its n, student and lexicon F1 against the human
  labels, agreement with the room rule, and the count of "unsure".
- The slot student and the lexicons are frozen before Round B starts.

## 7. Pool bias

The judgment pool was partly selected with Wayfair's query classifier ([DATA.md](DATA.md),
[data_audit.md](data_audit.md) section 8). Class features can therefore gain by retrieving judged
products. These rules apply:

- **A gain is claimed only if it is positive under both standard and condensed nDCG@10 with the 95% CI
  excluding 0. Otherwise it is reported as "direction depends on unjudged handling".** Both intervals
  must exclude 0. This covers every claimed gain: dense, fine-tuning, hybrid, LTR, every feature group and
  rewriting. The sentence goes into `configs/frozen.yaml` verbatim.
- For G3 (class features), a gain that fails the rule is also described as possibly pool-driven.
- Every claim shows condensed nDCG@10 and judged@10 beside the standard numbers.
- **Hard rule:** judged or unjudged status, and anything derived from WANDS's pooling, is never a feature.
- Phase 4 reports the judged fraction of the LTR top 10 against the BM25 top 10, per department.

## 8. Ranking model

- `label_gain` defaults to [0, 1, 2], matching the metric. [0, 1, 3] is a dev-fold candidate only. The
  choice goes into `configs/frozen.yaml`.
- **Selection rule:** among the training options (variant, stopping rule, model size), the one with the highest
  mean of standard and condensed nDCG@10 on the dev fold. An exact tie goes to judged-only. The rule is kept in
  `configs/ltr.yaml` and copied into `configs/frozen.yaml` at the freeze, because the existence of that file is
  what unlocks the other folds. Dev numbers are optimistic, since options are chosen on the queries they are
  scored on.
- **Dev comparisons from here:** one INTERIM run (G4 off), then, with G4, one FINAL run. Nothing is
  added after either except logged bug fixes. No option is searched for until LTR wins. Groups are not dropped
  before the freeze: the pre-registered ablation decides.
- **Encoder of G2** (written before the fine-tuned encoder was measured on WANDS): the fine-tuned encoder replaces
  off-the-shelf bge-small in candidates and features only if, on dev, it is a gain over off-the-shelf as a dense
  retriever under the rule of section 7, and hybrid RRF built with it is not a loss against hybrid RRF built
  with off-the-shelf. Otherwise off-the-shelf stays and the fine-tuned encoder is an ablation. The hybrid RRF
  that LTR is compared with always uses the encoder that G2 uses.
- Candidates are the union of the tuned BM25 top 100 and the dense top 100.
- Two training variants are compared on the dev fold, using both standard and condensed nDCG@10: (a) judged
  pairs only, and (b) unjudged candidates added with label 0. The choice goes into `configs/frozen.yaml`
  and the other variant is reported as an ablation.
  In both, training rows are candidates of the union above; (a) keeps the judged ones. At test time every
  candidate is ranked, judged or not.
- Dev experiments run out-of-fold inside the dev fold and every configuration is appended to
  `results/ltr_dev_log.jsonl`. The number of trees is chosen on training queries only.
- G4 is refused by `shelfrank/ltr/gates.py` until a slot student is selected.
- The main BM25 baseline is tuned multi-field BM25. Its field weights are the ones tuned on fold 0 and are
  not re-tuned per fold. Title-only BM25 appears in every table.
- Reciprocal rank fusion uses k = 60, the published default. **It is not tuned** and is frozen now.

- **Serving encoder** (written before any measurement): the demo encodes queries with int8 ONNX only if its dense
  nDCG@10 on dev is within 0.005 of PyTorch's, and with fp32 ONNX otherwise. Product vectors come from PyTorch in
  both cases, as they would in the demo. The rule applies to whichever encoder the demo uses. No other use of
  int8 is made before that measurement.

## 9. Statistics

- **Primary comparison: LTR (full) against hybrid RRF**, the strongest untrained system. Secondary: LTR
  (full) against tuned multi-field BM25. Both are in the Holm family.

- Paired bootstrap over queries, 10,000 resamples, seeded. Mean difference with a 95% CI.
- The comparison families are declared in `configs/frozen.yaml` and Holm-corrected within each family. The p-value is
  a two-sided bootstrap p. A claim needs both metrics, so its p is the larger of the two. The verdict of section 7 and
  the Holm-adjusted p are reported side by side, and a claimed gain needs both.

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
| Local ONNX export | ONNX and optimum export inside the Colab notebook | Avoids more compiled wheels that Smart App Control may block |
| Kaggle GPU | Google Colab, free tier | Kaggle phone verification is unavailable. Notebooks keep every path in one config cell, so they also run on Kaggle |

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
| 1 | None. `shelfrank/eval/access.py` refuses any other split until `configs/frozen.yaml` exists. The only query texts displayed were the five dev queries of the CI fixture | No |
| 2, planned | Round B shows 50 primary queries, after the freeze. Their ids were drawn by code and no text was displayed | No: the student and lexicons are frozen first |
| 2 (so far) | None. The seed notebook reads all WANDS queries by code, to remove look-alikes, and never prints or saves them. Hashed token keys of all queries are committed for the leakage tests | No |
| 3 | None. The fine-tuning notebook reads all WANDS queries by code, to remove look-alikes from ESCI, and never prints or saves them | No |
| 3 (dev) | None. The encoder comparison used the 96 dev queries only. Five invented check queries are not WANDS queries | No |
| 2, after the freeze | Gemini labelled the 384 primary queries with classes and slots by code. No text was displayed | No |
| 4, after the final run | `scripts/failure_cases.py` displayed 20 primary queries with their rankings, for the failure analysis | No: every setting was frozen and the run was made |
| 4 (dev) | None. Features, training and ablations used the 96 dev queries only | No |
| 2 | The leakage test found that one style word in a draft of the slot rules is a dev query. It was removed before any call used that draft | No |

2 of 211 class names have the same words as a WANDS query. The class prompt lists them because they are the
catalogue's own vocabulary. Every other phrase in a prompt is checked against all 480 queries.

## 14. LLM spend

- Hard limits in code, all in: labelling Rs 100, rewriting Rs 200, planned total Rs 300, absolute Rs 500.
- **The absolute limit is a constant in the code.** No configuration can raise it, and any call whose worst
  case would cross it is refused, whatever the other caps say.
- The Rs 200 between the planned total and the absolute limit is for re-runs and fixes. It is usable only
  through a recorded approval.
- Before a call, its worst case is reserved in the ledger. Afterwards the reservation is replaced by the
  actual cost. A crash therefore leaves the worst case counted, not zero.
- The ledger counts every call: successes, failures, retries and thinking tokens. Failures are counted
  although the documentation says failed requests are not charged.
- Every paid batch starts with a 10-item pilot. Its measured cost per item gives the projection, and the rest
  runs only after a recorded approval.
- Standing approval of 2026-11-26: a batch may follow its pilot without a separate yes if the pilot's cost per item is
  within 20% of the projection made before it, the phase stays under its cap (labelling Rs 100, rewriting Rs 200) and
  cumulative spend stays under Rs 300. Each use is recorded in `configs/llm.yaml` with both figures.
- Cached responses hold a hash of the prompt and no query text.
- The teacher is `gemini-3.1-flash-lite`.

## 15. Guards

`tests/test_leakage.py` checks fold sizes and disjointness, near-duplicate grouping, that the CI fixture
holds dev queries only, that the taxonomy is rebuilt from the catalogue alone, and that no primary query
of four or more words appears in committed files. Each later phase adds its derived datasets to this file.

Leakage checks compare two token keys, because the key frozen with the folds misses some plurals. Under the
stronger key no additional pair of WANDS queries crosses folds, so the folds stand.

## 17. Rewriting

- The trigger is one signal and one threshold, chosen on fold 0 among thresholds at fixed quantiles that fire on at most half of the
  queries. The rule is the largest mean gain of LTR over all dev queries, by the mean of standard and condensed nDCG@10. A tie goes
  to fewer rewrites. The choice is kept even when the gain is negative. It is committed in `configs/frozen_rewrite.yaml` before the
  primary queries are scored.
- Aggregate primary results of the ranker were already known at that point. They are not used in the choice.
- Family for Holm: LTR with triggered rewriting against LTR (primary), and hybrid RRF likewise. Slices and oracles are exploratory.

## 16. "More like this" (item-to-item recommendations, added 2026-11-26 before the freeze)

Written before anything in this section was measured.

- **Task.** Given a seed product, rank the rest of the catalogue by how well each product could replace it.
- **Reference.** For each query, its Exact products form one substitute set. Sets with fewer than 2 products are skipped.
  Every other member of the set is relevant, gain 1. The seed itself is removed from the ranking.
- **Seeds.** At most 20 per set, a seeded draw, so that a few large sets do not decide the result.
- **Metrics.** nDCG@10, Recall@10 (with its ceiling, because most sets have more than 10 members) and MRR, depth 100.
  Condensed versions keep only products that were judged for the seed's query.
- **Averaging.** Within a query over its seeds, then over queries, so every query counts once. Intervals resample
  queries, not seeds. The seed-weighted mean is reported beside it with the same clustered bootstrap.
- **Folds.** Design and tuning on fold 0 only. Primary results on folds 1 to 4, once, after the freeze.
- **Systems.** BM25 (query: the seed's name and class; document: name, class and description; default k1 and b, not
  tuned), off-the-shelf bge-small, the fine-tuned encoder, hybrid RRF (k = 60, depth 100 each) of BM25 and one encoder,
  and a re-ranker of the RRF top 100.
- **Encoder rule:** RRF uses the encoder with the higher mean of standard and condensed nDCG@10 on dev as a dense
  retriever alone. An exact tie goes to off-the-shelf.
- **Re-ranker:** RRF score scaled to [0, 1] plus weighted terms for shared class, agreement of listed attributes (color,
  material_finish, style, room, read with the G4 code) and a rating prior. Weights come from a grid on dev by the same
  mean, and every grid point is logged. The grid contains 0 for every term.
- **Comparisons**, one Holm family of their own: re-ranker against RRF (primary), RRF against BM25, fine-tuned against
  off-the-shelf. The rule of section 7 applies to each.
- **Limits.** Sets are defined by relevance to one query, not by co-views or purchases. This measures substitutability,
  not complementarity or engagement. Other catalogue products that could replace the seed but were never judged
  Exact for that query count as wrong. Class features can gain from how the pool was built.
