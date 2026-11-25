# Gate report: Phase 2, query understanding

- **Class student:** three arms, 36 configurations, selected by the pre-set rule ([class_dev.md](../class_dev.md)). The rule chose
  "both" although names-only has the better dev accuracy, and it was kept. Gemini zero-shot is on the same page.
- **Slots:** word tagger against the lexicon, selected by the pre-set rule ([slots_dev.md](../slots_dev.md)). Every F1 is vs LLM
  silver labels; human verification pending.
- **Protocol change before the freeze:** Round A was dropped (rater time). Round B is drawn (50 ids) and optional
  ([PROTOCOL.md](../PROTOCOL.md) section 6).
- **Surprises:** the teacher itself agrees with Wayfair's class on about two thirds of dev queries, so the student's gap to it is
  smaller than its gap to 1. A third of the ESCI reference is "other", which pulls the selection rule towards saying "other".
- **Spend:** see `results/llm_spend.jsonl`. No new non-dev exposure.
