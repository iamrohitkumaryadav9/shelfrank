# How the README will present the main result

Written on 2026-11-26, before the interim dev run, so that the result cannot shape the framing. The primary
comparison is LTR (full) against hybrid RRF on 384 queries. Its verdict has four values (PROTOCOL.md section 7), and
the text for each is fixed here. Numbers are filled from `results/*.json`. The four versions share one structure: the
same table, the same order, and the same limitations.

## The same in every version

1. First table: title-only BM25, tuned BM25, dense, hybrid RRF, LTR, with standard nDCG@10, condensed nDCG@10, judged@10
   and intervals. Every row appears whatever the outcome.
2. Directly below: the primary comparison (LTR against hybrid RRF), then the secondary one (LTR against tuned BM25),
   both Holm-corrected. **The secondary is never shown without the primary.**
3. Then the ablation table, all groups, including those that earned nothing.
4. Then where it fails, and the limitations.
5. The fine-tuned encoder is reported with its dev verdict in every version.

The first sentence of the README and of any resume bullet states the primary comparison. A gain over BM25 alone is not
the headline.

## Version A: gain (LTR > RRF)

> On WANDS, a LambdaMART ranker raised nDCG@10 from {rrf} (hybrid RRF) to {ltr}, a difference of {d} [{low}, {high}] on 384
> held-out queries, and the gain holds under condensed evaluation ({dc} [{lowc}, {highc}]). BM25 scores {bm25}. The groups
> that earned it are {groups}.

## Version B: no clear difference (LTR is about equal to RRF)

> On WANDS, a LambdaMART ranker with {n} features did not clearly beat hybrid RRF, which needs no training: {ltr} against
> {rrf}, a difference of {d} [{low}, {high}] on 384 held-out queries. Both beat tuned BM25 ({bm25}) [only if the rule says so].
> With this much training data the fusion of BM25 and a dense encoder is hard to improve on, and this report shows which
> features were tried and what each one added.

What B may claim: the size of effect the interval rules out, the cost of LTR compared with RRF, and the ablation. What B
may not do: lead with the BM25 gain, report only the better of the two metrics, or call a positive point estimate a trend.

## Version C: direction depends on unjudged handling

> LTR is {d} [{low}, {high}] against hybrid RRF on standard nDCG@10 and {dc} [{lowc}, {highc}] on condensed nDCG@10. The
> direction depends on how unjudged products are treated, so no gain is claimed.

## Version D: loss

> A LambdaMART ranker scored below hybrid RRF, {ltr} against {rrf}, {d} [{low}, {high}]. The simpler system is the better
> one here, and the failure analysis below shows why.

The demo serves the system that the primary comparison supports: LTR in version A, hybrid RRF otherwise, with LTR
selectable and labelled.
