"""Next-word suggestions from a corpus of unique queries.

The corpus has no frequencies of whole queries: every query occurs once. Counts are therefore
counts of word sequences across different queries, not popularity.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

BOS = "<s>"
ClassPrior = Callable[[str], dict[str, float]]


def words_of(query: str) -> list[str]:
    """Lower-cased words, kept as typed: a suggestion is shown to a person."""
    return query.lower().split()


@dataclass
class TrieNode:
    """One word of the prefix tree."""

    children: dict[str, "TrieNode"]
    count: int = 0
    ends: int = 0


class QueryTrie:
    """Completions of a typed prefix, by how many different queries continue that way."""

    def __init__(self, queries: Sequence[str]) -> None:
        """Insert every query."""
        self.root = TrieNode({})
        for query in queries:
            node = self.root
            for word in words_of(query):
                node = node.children.setdefault(word, TrieNode({}))
                node.count += 1
            node.ends += 1

    def node(self, words: Sequence[str]) -> TrieNode | None:
        """Node of a sequence of whole words."""
        node: TrieNode | None = self.root
        for word in words:
            node = node.children.get(word) if node else None
        return node

    def next_words(self, words: Sequence[str]) -> Counter[str]:
        """Words that followed this exact prefix."""
        node = self.node(words)
        return Counter({w: c.count for w, c in node.children.items()}) if node else Counter()


class SuggestModel:
    """Back-off over word sequences, optionally reweighted by the predicted product class.

    Order: the exact prefix in the trie, then the last two words, the last word, and all words.
    """

    def __init__(
        self,
        queries: Sequence[str],
        class_of: Sequence[str] | None = None,
        class_prior: ClassPrior | None = None,
        class_weight: float = 0.0,
    ) -> None:
        """Count. ``class_of`` gives the class of each training query, for the prior."""
        self.trie = QueryTrie(queries)
        self.bigram: dict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.unigram: Counter[str] = Counter()
        self.word_class: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_prior, self.class_weight = class_prior, class_weight
        for i, query in enumerate(queries):
            words = [BOS, *words_of(query)]
            for j in range(1, len(words)):
                self.unigram[words[j]] += 1
                self.bigram[words[j - 1]][words[j]] += 1
                if j >= 2:
                    self.trigram[(words[j - 2], words[j - 1])][words[j]] += 1
                if class_of is not None:
                    self.word_class[words[j]][class_of[i]] += 1

    def levels(self, words: Sequence[str]) -> list[Counter[str]]:
        """Candidate counts, most specific first."""
        padded = [BOS, *words]
        out = [self.trie.next_words(words)]
        if len(words) >= 2:
            out.append(self.trigram.get((padded[-2], padded[-1]), Counter()))
        out.append(self.bigram.get(padded[-1], Counter()))
        out.append(self.unigram)
        return out

    def _affinity(self, word: str, prior: dict[str, float]) -> float:
        counts = self.word_class.get(word)
        if not counts:
            return 0.0
        total = sum(counts.values())
        return sum(prior.get(c, 0.0) * n / total for c, n in counts.items())

    def rank(self, words: Sequence[str], k: int = 10, typed: str = "") -> list[str]:
        """Next words, best first. ``typed`` is the unfinished word, if any."""
        prior: dict[str, float] = {}
        if self.class_prior is not None and self.class_weight and words:
            prior = self.class_prior(" ".join(words))
        scores: dict[str, float] = {}
        scale = 1.0
        for counts in self.levels(words):
            total = sum(counts.values())
            for word, n in counts.items():
                if word.startswith(typed):
                    scores[word] = scores.get(word, 0.0) + scale * n / total
            scale *= 0.1
        if prior:
            for word in scores:
                scores[word] *= 1 + self.class_weight * self._affinity(word, prior)
        return sorted(scores, key=lambda w: (-scores[w], w))[:k]

    def complete(self, text: str, k: int = 5, depth: int = 2) -> list[str]:
        """Full suggestions for what has been typed, for the demo."""
        finished = text.endswith(" ") or not text
        words = words_of(text)
        typed = "" if finished or not words else words.pop()
        out: list[str] = []
        for word in self.rank(words, k, typed):
            sequence = [*words, word]
            for _ in range(depth - 1):
                following = self.trie.next_words(sequence)
                if not following:
                    break
                sequence.append(min(following, key=lambda w: (-following[w], w)))
            out.append(" ".join(sequence))
        return list(dict.fromkeys(out))


class FrequencyBaseline:
    """The most frequent words of the corpus, whatever was typed."""

    def __init__(self, queries: Sequence[str]) -> None:
        """Count words."""
        counts = Counter(w for q in queries for w in words_of(q))
        self.order = sorted(counts, key=lambda w: (-counts[w], w))

    def rank(self, words: Sequence[str], k: int = 10, typed: str = "") -> list[str]:
        """The same list for every prefix."""
        return [w for w in self.order if w.startswith(typed)][:k]


def reciprocal_rank(ranked: Sequence[str], truth: str) -> float:
    """1 / rank, and 0 when the word is not in the list."""
    return 1 / (ranked.index(truth) + 1) if truth in ranked else 0.0


def prefix_cases(queries: Sequence[str], length: int) -> list[tuple[int, list[str], str]]:
    """(query number, the first ``length`` words, the word that follows)."""
    out = []
    for i, query in enumerate(queries):
        words = words_of(query)
        if len(words) > length:
            out.append((i, words[:length], words[length]))
    return out
