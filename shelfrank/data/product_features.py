"""Parse the ``product_features`` string into key/value pairs."""

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedFeatures:
    """Pairs plus the parts that carry no usable key, kept so the audit can count them."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    keyless: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)

    def by_key(self) -> dict[str, list[str]]:
        """Group distinct values per key in first-seen order.

        Values are lists because many products carry several values for one key: they are
        families of purchasable variants, so "color" is a set of available options.
        """
        grouped: dict[str, list[str]] = defaultdict(list)
        for key, value in self.pairs:
            if value not in grouped[key]:
                grouped[key].append(value)
        return dict(grouped)


def parse_features(raw: object) -> ParsedFeatures:
    """Split on ``|`` and then on the first colon only, because values may contain colons.

    Numeric values appear as ``key:64.7`` and text values as ``key : oak``, so both spacings are
    accepted.
    """
    parsed = ParsedFeatures()
    if not isinstance(raw, str):
        return parsed
    for part in raw.split("|"):
        if not part.strip():
            continue
        key, colon, value = part.partition(":")
        key, value = key.strip(), value.strip()
        if not colon:
            parsed.malformed.append(part.strip())
        elif not key:
            parsed.keyless.append(value)
        else:
            parsed.pairs.append((key, value))
    return parsed


STYLE_KEYS = frozenset({"style", "dsprimaryproductstyle", "dssecondaryproductstyle"})
ROOM_KEYS = frozenset({"roomuse", "internalonly-roomuse"})


def slot_of_key(key: str) -> str | None:
    """Map a feature key to the slot it describes, for coverage counting.

    Style is an explicit set because a suffix rule would also accept mechanical keys such as
    ``faucethandlestyle``. Colour and material use the suffix because the part-level keys
    (``legcolor``, ``framematerial``) do describe the product.
    """
    if key in STYLE_KEYS:
        return "style"
    if key in ROOM_KEYS:
        return "room"
    if key.endswith("color"):
        return "color"
    if key.endswith("material") and not key.startswith("contains"):
        return "material"
    return None
