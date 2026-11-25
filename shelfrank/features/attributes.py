"""G4: attributes the query asks for against attributes the product lists.

Needs a selected slot student (shelfrank/ltr/gates.py). Matching is key-agnostic: a colour or
material is looked up in the union of the colour, finish and material keys. A product with several
values is a family of variants, so a conflict means that none of its values matches.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from shelfrank.data.product_features import ROOM_KEYS, parse_features
from shelfrank.qu.guidelines import SLOTS
from shelfrank.qu.slot_lexicon import canonical
from shelfrank.qu.slot_vocab import SAME_PLACE, Term, key_group, room_spans
from shelfrank.text.normalize import tokenize

MATCH, CONFLICT, NOT_MENTIONED, NOT_ASKED = "match", "conflict", "not_mentioned", "not_asked"
STATES = (MATCH, CONFLICT, NOT_MENTIONED)


@dataclass
class Listed:
    """What a product lists: values to match, and which slots it says anything about."""

    union: set[Term] = field(default_factory=set)
    style: set[Term] = field(default_factory=set)
    room: set[Term] = field(default_factory=set)
    mentioned: set[str] = field(default_factory=set)


def _places(term: Term) -> Term:
    return tuple(SAME_PLACE.get(t, t) for t in term)


def listed_values(raw_features: object, hierarchy: object, slot_of: Mapping[Term, str]) -> Listed:
    """Read a product. ``slot_of`` is the lexicon's slot for each known colour or material.

    Whether the product mentions colour or material is decided by what its values mean and not by
    their key, because the catalogue lists walnut as a colour.
    """
    out = Listed()
    for key, value in parse_features(raw_features).pairs:
        term = canonical(tuple(tokenize(value)))
        if not term or value in ("yes", "no"):
            continue
        group = key_group(key)
        if group in ("color", "finish", "material"):
            out.union.add(term)
            fallback = "color" if group == "color" else "material_finish"
            out.mentioned.add(slot_of.get(term, fallback))
        elif group == "style":
            out.style.add(term)
            out.mentioned.add("style")
        elif key in ROOM_KEYS:
            out.room.add(_places(term))
    if isinstance(hierarchy, str):
        tokens = tokenize(hierarchy)
        out.room |= {_places(tuple(tokens[a:b])) for a, b in room_spans(hierarchy)}
    if out.room:
        out.mentioned.add("room")
    return out


def contains(value: Term, wanted: Term) -> bool:
    """Whether the wanted words occur together in a value: "gray" matches "dark gray"."""
    n = len(wanted)
    return n > 0 and any(value[i : i + n] == wanted for i in range(len(value) - n + 1))


def state(slot: str, span: str, listed: Listed) -> str:
    """Match, conflict or not mentioned, for one requested value."""
    wanted = canonical(tuple(tokenize(span)))
    if slot == "room":
        wanted, values = _places(wanted), listed.room
    else:
        values = listed.style if slot == "style" else listed.union
    if any(contains(value, wanted) for value in values):
        return MATCH
    return CONFLICT if slot in listed.mentioned else NOT_MENTIONED


def slot_state(slot: str, spans: Sequence[str], listed: Listed) -> str:
    """One state per slot. A conflict on any requested value is a conflict."""
    if not spans:
        return NOT_ASKED
    states = [state(slot, span, listed) for span in spans]
    for worst in (CONFLICT, NOT_MENTIONED):
        if worst in states:
            return worst
    return MATCH


def attribute_features(
    pairs: pd.DataFrame,
    query_slots: Sequence[Mapping[str, Sequence[str]]],
    listed: Mapping[int, Listed],
) -> pd.DataFrame:
    """Columns g4_*: three flags per slot, and totals. ``listed`` is keyed by ``pi``."""
    rows = []
    for q, p in zip(pairs["qi"].tolist(), pairs["pi"].tolist(), strict=True):
        states = [slot_state(s, query_slots[q].get(s, []), listed[p]) for s in SLOTS]
        flags = [float(s == kind) for s in states for kind in STATES]
        asked = sum(s != NOT_ASKED for s in states)
        matches, conflicts = states.count(MATCH), states.count(CONFLICT)
        share = matches / asked if asked else 0.0
        rows.append((*flags, float(asked), float(matches), float(conflicts), share))
    columns = [f"{s}_{kind}" for s in SLOTS for kind in STATES]
    columns += ["asked", "matches", "conflicts", "match_share"]
    return pd.DataFrame(rows, columns=[f"g4_{c}" for c in columns])
