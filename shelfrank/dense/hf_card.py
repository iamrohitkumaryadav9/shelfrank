"""Model card of the fine-tuned encoder for the Hugging Face Hub, built from committed results."""

from collections.abc import Mapping
from typing import Any

from shelfrank.data.audit_report import table


def _gap(c: Mapping[str, float]) -> str:
    return f"{c['mean']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}]"


def card(
    training: Mapping[str, Any],
    dense: Mapping[str, Any],
    final: Mapping[str, Any],
    repository: str,
    model_id: str = "iamrohitkumar9/bge-small-en-v1.5-esci",
) -> str:
    """Front matter and text."""
    source = training["provenance"]
    data = training["data"]
    settings = "\n".join(f"| {k} | {v} |" for k, v in training["config"].items())
    comparisons = {
        (r["system"], r["baseline"]): r for r in final["comparisons"]["primary"]["retrieval"]
    }
    primary = comparisons[("dense_esci", "dense")]
    fused = comparisons[("hybrid_rrf_esci", "hybrid_rrf")]
    dev = dense["differences"][0]
    rows = [
        [
            "Held-out ESCI (in-domain, 500 queries, no interval)",
            f"{training['held_out_ndcg_before']:.4f}",
            f"{training['held_out_ndcg_after']:.4f}",
            "",
            "",
        ],
        [
            "WANDS dev fold (96 queries)",
            f"{dense['systems']['dense_bge_small']['ndcg10']['mean']:.3f}",
            f"{dense['systems']['dense_bge_small_esci']['ndcg10']['mean']:.3f}",
            _gap(dev["ndcg10"]),
            _gap(dev["condensed_ndcg10"]),
        ],
        [
            "WANDS folds 1 to 4 (384 queries)",
            "",
            "",
            _gap(primary["ndcg10"]),
            _gap(primary["condensed_ndcg10"]),
        ],
        [
            "WANDS folds 1 to 4, inside hybrid RRF with BM25",
            "",
            "",
            _gap(fused["ndcg10"]),
            _gap(fused["condensed_ndcg10"]),
        ],
    ]
    return f"""---
license: mit
base_model: {source["base_model"]}
library_name: sentence-transformers
pipeline_tag: sentence-similarity
language:
- en
tags:
- product-search
- e-commerce
- sentence-transformers
---

# bge-small-en-v1.5 fine-tuned on Amazon ESCI

A bi-encoder for product search: [{source["base_model"]}](https://huggingface.co/{source["base_model"]}) at revision `{source["base_revision"][:12]}`, trained
for one epoch on the US small version of the Amazon Shopping Queries Dataset (ESCI). Part of
[ShelfRank](https://github.com/{repository}), which evaluates it on Wayfair's WANDS benchmark.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{model_id}")
queries = model.encode(["Represent this sentence for searching relevant passages: grey velvet sofa"], normalize_embeddings=True)
products = model.encode(["Tincher 74'' velvet round arm curved sofa . Sofas . ..."], normalize_embeddings=True)
```

Queries take the instruction prefix of the base model. Vectors are the first token, normalized. The file has no pooler layer, which
this pooling never reads.

## Training data

{training["triplets"]:,} triplets from {data["train_queries"]:,} ESCI training queries: an Exact product as positive and, for
{training["hard_negative_share"]:.0%} of them, an Irrelevant product of the same query as hard negative. Other Exact products of a query are
masked out of the in-batch negatives. {data["held_out_queries"]} queries were held out. ESCI queries that resemble any WANDS query were removed
({data["dropped"]["resembles a protected query"]} of them). **No WANDS data was used for training.**

## Training

Multiple-negatives ranking loss, plain PyTorch, fp16 on a Colab T4, code at commit `{source["git_commit"][:7]}` of the repository.

| Setting | Value |
| --- | --- |
{settings}

Wall time and resume history of the run were not logged.

## Evaluation: in-domain and transfer to home goods

{table(["Evaluation", "nDCG before", "nDCG after", "Difference, nDCG@10", "Difference, condensed"], rows)}

Differences are paired, with 95% bootstrap intervals over queries. On the WANDS dev fold the gain was not clear, so the ShelfRank ranker
kept the off-the-shelf encoder by a rule written before the test folds were read. On the 384 held-out WANDS queries it is better under
both standard and condensed nDCG@10. An Amazon catalogue transfers to Wayfair's home-goods catalogue, but the gain is small.

## Limitations

- int8 ONNX export lost about 0.01 nDCG@10 on WANDS dev, so only fp32 is recommended.
- One training run with one seed. The held-out ESCI result has no interval because per-query scores were not saved.
- English, US locale only.

## License

MIT, as the base model. ESCI is Apache-2.0 (Amazon). WANDS is MIT (Wayfair). Not affiliated with either.
"""
