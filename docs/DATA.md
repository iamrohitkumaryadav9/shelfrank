# Data

## WANDS

| | |
| --- | --- |
| Source | https://github.com/wayfair/WANDS |
| Commit | `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5` |
| License | MIT, "Copyright (c) 2021 ecir2022" |
| Files | `query.csv`, `label.csv`, `product.csv` (tab-separated), SHA-256 in `configs/data.yaml` |
| Paper | Chen, Liu, Liu, Sun, Baltrunas and Schroeder, "WANDS: Dataset for Product Search Relevance Assessment", ECIR 2022 ([preprint](https://easychair.org/publications/preprint/j2D4)) |

`tests/fixtures/wands` holds a 5-query, 30-product extract under the same license.

### What the paper says that shapes this project

- **Judgments were pooled in two steps.** Step 1 used click and cart logs, Solr, and a neural
  retriever. Step 2 mined the pool again with both retrievers and added products in the class that a
  proprietary query classifier predicted, stopping when Exact labels fell to 5% (section 4.3). The
  second step supplied 184,058 of the 233,448 labels (Table 5). The pool therefore favours lexical
  retrieval and class match. [data_audit.md](data_audit.md) section 8 measures the second effect.
- **`query_class` is a model prediction.** Table 1 calls it "Predicted Class". Agreement with it
  measures agreement with Wayfair's classifier, not correctness.
- **Gains.** Table 3 maps Exact, Partial and Irrelevant to 1.0, 0.5 and 0.0. Ours are 2, 1 and 0.
  nDCG does not change when all gains are scaled, so the two are identical under linear gain.
- **Labels are noisy.** Annotator agreement (Cohen's kappa) rose from 0.467 in the first two months to
  0.826 in the fourth (Table 2).

### Published baseline

No numeric table of nDCG exists in the paper or the repository. Figure 4 plots nDCG@10 for BM25 over
title and description (Solr). **Read off the plot, BM25 is about 0.72** and random ranking is about 0.04.
The BM25 parameters, analyser and treatment of repeated judgments are not given. Phase 1 reports ours
next to this approximate value as a sanity check and does not claim a reproduction.

## ESCI (Amazon Shopping Queries)

Not downloaded locally. Used only inside the GPU notebook, from Phase 2. Its license and source commit
are recorded here when it is first used.
