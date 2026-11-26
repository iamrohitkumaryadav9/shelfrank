---
license: mit
base_model: BAAI/bge-small-en-v1.5
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

A bi-encoder for product search: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) at revision `5c38ec7c405e`, trained
for one epoch on the US small version of the Amazon Shopping Queries Dataset (ESCI). Part of
[ShelfRank](https://github.com/iamrohitkumaryadav9/shelfrank), which evaluates it on Wayfair's WANDS benchmark.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("<this repository>")
queries = model.encode(["Represent this sentence for searching relevant passages: grey velvet sofa"], normalize_embeddings=True)
products = model.encode(["Tincher 74'' velvet round arm curved sofa . Sofas . ..."], normalize_embeddings=True)
```

Queries take the instruction prefix of the base model. Vectors are the first token, normalized. The file has no pooler layer, which
this pooling never reads.

## Training data

173,440 triplets from 20,058 ESCI training queries: an Exact product as positive and, for
58% of them, an Irrelevant product of the same query as hard negative. Other Exact products of a query are
masked out of the in-batch negatives. 500 queries were held out. ESCI queries that resemble any WANDS query were removed
(11 of them). **No WANDS data was used for training.**

## Training

Multiple-negatives ranking loss, plain PyTorch, fp16 on a Colab T4, code at commit `8a9fc4f` of the repository.

| Setting | Value |
| --- | --- |
| batch_size | 64 |
| learning_rate | 2e-05 |
| warmup_share | 0.1 |
| weight_decay | 0.01 |
| max_grad_norm | 1.0 |
| scale | 20.0 |
| max_query_tokens | 32 |
| max_passage_tokens | 128 |
| checkpoint_every | 500 |
| log_every | 50 |
| max_minutes | 100 |
| seed | 20260 |

Wall time and resume history of the run were not logged.

## Evaluation: in-domain and transfer to home goods

| Evaluation | nDCG before | nDCG after | Difference, nDCG@10 | Difference, condensed |
| --- | --- | --- | --- | --- |
| Held-out ESCI (in-domain, 500 queries, no interval) | 0.8280 | 0.8491 |  |  |
| WANDS dev fold (96 queries) | 0.724 | 0.724 | -0.000 [-0.038, +0.034] | +0.009 [-0.017, +0.031] |
| WANDS folds 1 to 4 (384 queries) |  |  | +0.021 [+0.007, +0.035] | +0.014 [+0.003, +0.025] |
| WANDS folds 1 to 4, inside hybrid RRF with BM25 |  |  | +0.012 [+0.004, +0.022] | +0.009 [+0.002, +0.017] |

Differences are paired, with 95% bootstrap intervals over queries. On the WANDS dev fold the gain was not clear, so the ShelfRank ranker
kept the off-the-shelf encoder by a rule written before the test folds were read. On the 384 held-out WANDS queries it is better under
both standard and condensed nDCG@10. An Amazon catalogue transfers to Wayfair's home-goods catalogue, but the gain is small.

## Limitations

- int8 ONNX export lost about 0.01 nDCG@10 on WANDS dev, so only fp32 is recommended.
- One training run with one seed. The held-out ESCI result has no interval because per-query scores were not saved.
- English, US locale only.

## License

MIT, as the base model. ESCI is Apache-2.0 (Amazon). WANDS is MIT (Wayfair). Not affiliated with either.
