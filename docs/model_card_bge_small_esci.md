# bge-small-en-v1.5 fine-tuned on ESCI (US, small version)

Base: BAAI/bge-small-en-v1.5 at 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a.
Code: commit 8a9fc4f0801a5fe1a3356fc6b5c3662126836d07.

Trained on 173,440 triplets from 20,058 ESCI training queries
(Exact as positive, Irrelevant of the same query as hard negative for
58% of them). Queries that resemble a WANDS query were removed.
No WANDS data was used.

Held-out ESCI nDCG over judged products, 500 queries set aside, of which those
without an Exact product are skipped: 0.8280 before, 0.8491 after. Per-query scores were not saved, so there is no confidence interval.
This is an in-domain check. It says nothing about WANDS, which is measured separately.

Steps: 2,710 of 2,710. Wall time: not logged. Resume history: not logged.

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

ESCI is licensed Apache-2.0 by Amazon. bge-small-en-v1.5 is licensed MIT by BAAI.
