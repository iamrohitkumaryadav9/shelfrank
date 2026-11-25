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

No numeric table of nDCG exists in the paper or the repository. The paper plots BM25 (Solr, title and
description) at roughly 0.72 nDCG@10 with unstated settings (Figure 4, read by eye). Ours is reported on
the same queries beside it as a visual sanity check. BM25 is not tuned toward it and no reproduction is
claimed.

## ESCI (Amazon Shopping Queries)

| | |
| --- | --- |
| Source | https://github.com/amazon-science/esci-data |
| Commit | `7916cdf6ab75a462e77f20ab40428a10923998d5` |
| License | Apache-2.0 |
| `examples.parquet` | 51,286,808 bytes, SHA-256 `4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a` |
| `products.parquet` | 1,108,857,465 bytes, SHA-256 `25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265` |

Sizes and hashes come from the Git LFS pointers. The files are downloaded and processed only in Google
Colab, never on the development laptop. Only small derived files enter this repository.