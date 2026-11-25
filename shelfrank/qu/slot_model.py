"""Slot student: a linear model over single words, trained on teacher-labelled ESCI queries.

Word classification and not a sequence model: queries are short, spans are mostly one or two words,
and 1,600 queries are too few to learn transitions that matter.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml

from shelfrank.config import REPO_ROOT
from shelfrank.qu.guidelines import SLOT_VERSION, SLOTS
from shelfrank.qu.slot_lexicon import LexiconMatcher
from shelfrank.text.normalize import tokenize

NONE = "none"
PUNCTUATION = ".,;:!?()\"'"
CONFIG = REPO_ROOT / "configs/slot_student.yaml"
LEXICON, TAGGER = "lexicon", "tagger"

Slots = dict[str, list[str]]


class SlotModel(Protocol):
    """Anything that reads slots from a query."""

    def slots(self, query: str) -> Slots:
        """Slots of a query, as the query's own words."""
        ...


@dataclass(frozen=True)
class TaggerConfig:
    """One student configuration."""

    lexicon_features: bool
    window: int
    c: float


def words_of(query: str) -> list[str]:
    """Words without surrounding punctuation."""
    return [w.strip(PUNCTUATION) for w in query.split()]


def word_labels(query: str, slots: Mapping[str, Sequence[str]]) -> list[str]:
    """Slot of each word. A span that is not found in the words is skipped."""
    words = [w.casefold() for w in words_of(query)]
    out = [NONE] * len(words)
    for slot in SLOTS:
        for span in slots.get(slot, []):
            part = [w.strip(PUNCTUATION) for w in span.casefold().split()]
            for start in range(len(words) - len(part) + 1):
                free = all(x == NONE for x in out[start : start + len(part)])
                if part and words[start : start + len(part)] == part and free:
                    out[start : start + len(part)] = [slot] * len(part)
                    break
    return out


def spans_from_labels(query: str, labels: Sequence[str]) -> Slots:
    """Join neighbouring words of one slot."""
    words = words_of(query)
    out: Slots = {slot: [] for slot in SLOTS}
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != NONE:
                out[labels[start]].append(" ".join(words[start:i]))
            start = i
    return out


def word_features(
    query: str, config: TaggerConfig, lexicon: LexiconMatcher
) -> list[dict[str, float]]:
    """Features of every word of a query."""
    words = [" ".join(tokenize(w)) or w.casefold() for w in words_of(query)]
    found = word_labels(query, lexicon.slots(query)) if config.lexicon_features else []
    known = {slot: {t for t in terms} for slot, terms in lexicon.terms.items()}
    rows = []
    for i, word in enumerate(words):
        row: dict[str, float] = {f"w={word}": 1, "bias": 1, "last": float(i == len(words) - 1)}
        row["first"] = float(i == 0)
        for n in (2, 3, 4):
            row[f"suffix={word[-n:]}"] = 1
        row[f"prefix={word[:3]}"] = 1
        row["digit"] = float(any(c.isdigit() for c in word))
        for step in range(1, config.window + 1):
            before = words[i - step] if i >= step else "<s>"
            after = words[i + step] if i + step < len(words) else "</s>"
            row[f"-{step}={before}"] = 1
            row[f"+{step}={after}"] = 1
        if config.lexicon_features:
            row[f"lexicon={found[i]}"] = 1
            for slot, terms in known.items():
                row[f"in_{slot}"] = float(tuple(word.split()) in terms)
            for step in (-1, 1):
                if 0 <= i + step < len(words):
                    row[f"lexicon{step:+d}={found[i + step]}"] = 1
        rows.append(row)
    return rows


class WordTagger:
    """Multinomial logistic regression over word features."""

    def __init__(self, config: TaggerConfig, lexicon: Mapping[str, Any]) -> None:
        """Keep the settings and the lexicon the features use."""
        self.config = config
        self.lexicon = dict(lexicon)
        self.matcher = LexiconMatcher(lexicon)

    def fit(self, queries: Sequence[str], labels: Sequence[Mapping[str, Sequence[str]]]) -> Any:  # noqa: ANN401
        """Train on teacher labels."""
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression

        rows: list[dict[str, float]] = []
        targets: list[str] = []
        for query, slots in zip(queries, labels, strict=True):
            rows += word_features(query, self.config, self.matcher)
            targets += word_labels(query, slots)
        vectorizer = DictVectorizer(sparse=True)
        model = LogisticRegression(C=self.config.c, max_iter=2000)
        model.fit(vectorizer.fit_transform(rows), targets)
        # Plain numbers: the demo then needs no scikit-learn to read slots.
        self.vocabulary = {str(k): int(v) for k, v in vectorizer.vocabulary_.items()}
        self.classes = [str(c) for c in model.classes_]
        self.coef = model.coef_.astype("float32")
        self.bias = model.intercept_.astype("float32")
        return self

    def labels(self, query: str) -> list[str]:
        """Slot of each word."""
        out = []
        for row in word_features(query, self.config, self.matcher):
            score = self.bias.copy()
            for name, value in row.items():
                column = self.vocabulary.get(name)
                if column is not None and value:
                    score = score + value * self.coef[:, column]
            out.append(self.classes[int(score.argmax())])
        return out

    def slots(self, query: str) -> Slots:
        """Slots of a query."""
        return spans_from_labels(query, self.labels(query)) if query.split() else empty()

    def to_plain(self) -> dict[str, Any]:
        """Everything needed to rebuild it, without scikit-learn objects."""
        return {
            "config": asdict(self.config),
            "lexicon": self.lexicon,
            "vocabulary": self.vocabulary,
            "classes": self.classes,
            "coef": self.coef,
            "bias": self.bias,
        }

    @classmethod
    def from_plain(cls: type["WordTagger"], body: Mapping[str, Any]) -> "WordTagger":
        """Rebuild."""
        out = cls(TaggerConfig(**body["config"]), body["lexicon"])
        out.vocabulary, out.classes = body["vocabulary"], body["classes"]
        out.coef, out.bias = body["coef"], body["bias"]
        return out


def empty() -> Slots:
    """No slot."""
    return {slot: [] for slot in SLOTS}


def labelled_slots(source: str, root: Path = REPO_ROOT) -> dict[int, Slots]:
    """Teacher slot labels by query id."""
    path = root / f"data/labels/slots_{source}_{SLOT_VERSION}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {int(row["id"]): row["label"] for row in rows}


def load_lexicon_data(root: Path = REPO_ROOT) -> dict[str, Any]:
    """The committed lexicon."""
    return json.loads((root / "data/slot_lexicon.json").read_text(encoding="utf-8"))


def load_lexicon(root: Path = REPO_ROOT) -> LexiconMatcher:
    """The baseline."""
    return LexiconMatcher(load_lexicon_data(root))


def load_slot_model(root: Path = REPO_ROOT) -> SlotModel:
    """The selected slot model."""
    chosen = yaml.safe_load((root / "configs/slot_student.yaml").read_text(encoding="utf-8"))
    if chosen["kind"] == LEXICON:
        return load_lexicon(root)
    import joblib

    return WordTagger.from_plain(joblib.load(root / chosen["path"]))
