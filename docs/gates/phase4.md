# Gate report: Phase 4, learning to rank

- **Pre-registered result:** version A of [README_FRAMING.md](../README_FRAMING.md). Numbers: [final_results.md](../final_results.md).
- **Surprise:** the primary gain is larger than any dev comparison showed. It was checked for leaks before it was believed
  ([RESULTS.md](../RESULTS.md), section 2): exact repetition, shuffled labels, training size, folds, similar training queries, a stronger baseline.
- **Earned by** G1 and G2 after Holm. G4, the attribute match, is not a claimed effect.
- **Null and negative:** pointwise is not clearly worse, the fine-tuned encoder inside LTR changes nothing, and training on judged pairs
  only is worse under standard evaluation and better under condensed.
- **Exposure:** 20 primary queries were displayed after the final run for the failure analysis.
