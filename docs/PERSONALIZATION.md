# Personalization: a design note

Nothing in this note was built or measured. It says what is missing, what I would build, and how I would find out whether it works.

## Why it cannot be evaluated here

WANDS and ESCI have no users and no sessions. A row is a query, a product and a relevance label given by an annotator. There is no
click, no cart, no purchase, no time and no identity. A personalization result on this data would need invented users, and an invented
user agrees with whoever invented them. So this repository has no personalized ranker and no simulated one.

## What I would add at Wayfair

Re-rank the top results of the current ranker with a small number of session features. Retrieval stays shared and cacheable.

| Feature | Built from | Why |
| --- | --- | --- |
| Share of recently viewed products in the candidate's class | product views in the session | a shopper looking at sofas means the sofa sense of an ambiguous query |
| Style affinity: dot product of the shopper's style histogram and the product's styles | the same attribute parsing as G4 (`shelfrank/features/attributes.py`) | style is stable across a room, and a query rarely states it |
| Color and material affinity | the same | furniture has to match what is already bought |
| Price-band distance | price of viewed and bought products, per class | a budget is per class, not per shopper |
| Already viewed, already bought | events | a bought sofa should drop out, a viewed lamp may return |
| Household signals: rooms seen, project in progress | room slots and category paths of viewed products | a bathroom project makes "vanity" unambiguous |

The affinity vectors decay with time, and the session part outweighs history: people furnish one room and then another.

## Data needed

Query, impression, position, click, add-to-cart and purchase events with a pseudonymous shopper id and session id, product price at the
time of the impression, and logged ranking scores. Position and impression logs are the part that is easiest to lack and hardest to
replace: without them, clicks cannot be corrected for position.

## Cold start

- New shopper: the features are missing, not zero, and LightGBM routes missing values separately. The ranker then equals today's.
- New session of a known shopper: history only, down-weighted.
- New product: no interaction features are used about products, only attributes, so a new product has the same features as an old one.

## Offline

Counterfactual evaluation on logged impressions, with inverse propensity weights from a small randomized swap experiment on adjacent
positions. Metrics: weighted nDCG with purchases and carts as gains, by shopper tenure and by session length. Guardrail: nDCG against human
labels on a fixed query set, because a personalized ranker must not show irrelevant products. WANDS-style judgments are still useful for
that. Offline numbers choose candidates for a test. They are not evidence of impact.

## Online

| | |
| --- | --- |
| Hypothesis | Session-aware re-ranking raises add-to-cart per search for shoppers with at least 3 views in the session |
| Unit | Shopper, hashed, so one person sees one system |
| Primary metric | Add-to-cart per search |
| Guardrails | Search exit rate, zero-result rate, p95 latency, diversity of classes in the top 10, return rate after 30 days |
| Novelty check | Effect by week, because a first-week effect can fade |
| Roll back if | any guardrail's interval excludes its margin on the wrong side, or the effect is limited to repeat views of products already seen |

## Risks

A style affinity can narrow what a shopper ever sees. I would keep one or two positions for results that the non-personalized ranker would
show, and measure class and style diversity per session. Affinities would use catalogue attributes only, with no inferred personal traits.
