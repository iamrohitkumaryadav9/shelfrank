"""Which queries to rewrite, from signals that exist at serving time. No label is used."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from shelfrank.ltr.first_stage import FirstStage
from shelfrank.qu.class_student import LinearClassifier
from shelfrank.qu.slot_model import SlotModel
from shelfrank.text.normalize import tokenize

LOW, HIGH = "at most", "at least"
# Signals where a small value means a hard query, and where a large one does.
SIGNALS = {
    "bm25_top": LOW,
    "dense_top": LOW,
    "overlap10": LOW,
    "class_confidence": LOW,
    "slot_values": HIGH,
    "words": HIGH,
}
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5)
MAX_SHARE = 0.5


@dataclass(frozen=True)
class Trigger:
    """Rewrite when the signal is on the hard side of the threshold."""

    signal: str
    threshold: float

    def fires(self, signals: Mapping[str, float]) -> bool:
        """Whether to rewrite a query with these signals."""
        value = signals[self.signal]
        return value <= self.threshold if SIGNALS[self.signal] == LOW else value >= self.threshold


def query_signals(
    stage: FirstStage, student: LinearClassifier, slot_model: SlotModel
) -> dict[int, dict[str, float]]:
    """Signals of every query of a stage."""
    confidence = student.probabilities(stage.texts).max(axis=1)
    out = {}
    for i, (q, text) in enumerate(zip(stage.query_ids, stage.texts, strict=True)):
        bm25, dense = stage.runs["bm25_multi"][q][:10], stage.runs["dense"][q][:10]
        out[q] = {
            "bm25_top": float(stage.tuned[i].max()),
            "dense_top": float(stage.cosine[i].max()),
            "overlap10": len(set(bm25) & set(dense)) / 10,
            "class_confidence": float(confidence[i]),
            "slot_values": float(sum(len(v) for v in slot_model.slots(text).values())),
            "words": float(len(tokenize(text))),
        }
    return out


def candidate_triggers(signals: Mapping[int, Mapping[str, float]]) -> list[Trigger]:
    """Thresholds at quantiles of the observed values that fire on at most half of the queries."""
    out = []
    for name, side in SIGNALS.items():
        values = np.array([s[name] for s in signals.values()])
        for quantile in QUANTILES:
            level = quantile if side == LOW else 1 - quantile
            trigger = Trigger(name, float(np.quantile(values, level)))
            share = np.mean([trigger.fires(s) for s in signals.values()])
            if 0 < share <= MAX_SHARE and trigger not in out:
                out.append(trigger)
    return out


def choose(
    triggers: Sequence[Trigger],
    signals: Mapping[int, Mapping[str, float]],
    gain: Mapping[int, float],
) -> tuple[Trigger, list[dict[str, float | str]]]:
    """Trigger with the largest mean gain over all queries; a tie goes to fewer rewrites."""
    log = []
    for trigger in triggers:
        fired = [q for q in signals if trigger.fires(signals[q])]
        log.append(
            {
                "signal": trigger.signal,
                "threshold": trigger.threshold,
                "share": len(fired) / len(signals),
                "gain": sum(gain[q] for q in fired) / len(signals),
            }
        )
    best = max(range(len(log)), key=lambda i: (log[i]["gain"], -float(log[i]["share"])))
    return triggers[best], log
