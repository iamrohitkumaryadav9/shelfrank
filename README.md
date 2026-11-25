# ShelfRank

Query understanding and learning-to-rank for home-goods product search, evaluated on
[WANDS](https://github.com/wayfair/WANDS), Wayfair's public relevance benchmark (ECIR 2022).

**Status: Phase 1, retrieval baselines on the dev fold. No primary result exists yet.**

| Document | What it covers |
| --- | --- |
| [docs/data_audit.md](docs/data_audit.md) | What is in WANDS and what it means for evaluation (generated) |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Splits, metrics, label rules and what has been looked at so far |
| [docs/retrieval_dev.md](docs/retrieval_dev.md) | Retrieval baselines on the 96 dev queries (generated) |
| [docs/DATA.md](docs/DATA.md) | Source, commit, license, published baseline |
| [docs/slot_evidence.md](docs/slot_evidence.md) | What the catalogue says about attribute slots (generated) |
| [docs/COLAB.md](docs/COLAB.md) | Running the ESCI notebooks |
| [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) | Pinned versions and why |

## Findings so far

<!-- begin:findings -->
All on the 96 dev queries, exploratory and uncorrected.

- **The order of dense and BM25 depends on unjudged handling.** Against tuned BM25, off-the-shelf bge-small
  is -0.036 [-0.075, +0.005] on standard nDCG@10 and
  +0.033 [-0.004, +0.075] on condensed nDCG@10. Its
  top-10 results outside the query's class are 65.6% judged, against
  92.7% inside it, which lowers its standard score.
- **Tuning is optimistic, which is why results are held out.** Tuned field weights score
  0.759 on the queries they were tuned on and
  0.742 on queries they were not. The held-out gain over title-only BM25 is
  +0.015 [-0.015, +0.044].
- **The product class field has weight 0** once the category path has weight
  0.5, and 3 of 4 held-out searches agree. The likely
  reason is that the path carries the same signal.
<!-- end:findings -->

## Limitations known so far

<!-- begin:limitations -->
- **The judgment pool favours class match.** Wayfair's own query classifier helped choose
  which products were judged. A product in the query's class is judged
  55.4% of the time, against
  1.1% for a random pair, so class features can gain by retrieving judged
  products. Condensed nDCG@10 and judged@10 are reported beside every such claim.
- **Some metrics have low ceilings.** A perfect ranker reaches Exact@10 of
  0.764 on the 305 primary queries that have an Exact product, and
  Recall@100 of 0.575 on 383.
- **The benchmark is small.** It has 480 queries, and the headline uses
  384.
- **Patterns have no slot.** Floral, striped, geometric, plaid and solid are out of scope in this version. The
  catalogue keeps them under their own keys and lists none of them as a style.
- **The taxonomy has gaps.** No blanket class reaches the 30-product cut, so a query for a blanket is labelled
  "Bedding Sets".
- **`query_class` is itself a model prediction**, so classifier accuracy means agreement with Wayfair's
  classifier.
<!-- end:limitations -->

## Reproduce

```powershell
uv sync --extra dev --extra bench
uv run python scripts/get_wands.py      # about 96 MB, verified by SHA-256
uv run python scripts/data_audit.py     # writes results/data_audit.json and docs/data_audit.md
uv run python scripts/encode_products.py bge_small   # about 35 min on CPU, resumable
uv run python scripts/tune_bm25.py      # dev fold only
uv run python scripts/run_retrieval.py --split dev
uv run python scripts/make_tables.py    # fills every generated table
uv run pytest
```

Individual project by Rohit Kumar, M.Tech CSE, IIIT Delhi.
