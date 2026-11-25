# Gate report: Phases 6 and 6b

- **Suggestions:** [suggest.md](../suggest.md). No popularity data exists. The n-gram model beats the frequency baseline. The class prior got weight 0.
- **More like this:** [similar.md](../similar.md). RRF beats BM25, and the fine-tuned encoder beats off-the-shelf. The re-ranker with attributes does
  not beat plain RRF on the primary queries.
- The suggestion corpus is the 1,600 local ESCI queries, because ESCI is processed only in Colab.
