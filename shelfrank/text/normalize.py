"""One text pipeline for queries and products, so that both sides always agree.

Every rule is a general property of English or of catalogue typography. None was derived from WANDS
queries.
"""

import re
import unicodedata

_PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "″": '"', "×": "x", "–": "-"})
# A quote mark or "in" counts as inches only straight after a number, since "in" alone is a
# preposition, and not before another number, since "3 in 1" is not a length.
_INCH = re.compile(r"(?<=\d)\s*(?:''|\"|inch(?:es)?\b|in\b(?!\s*\d)\.?)")
_FOOT = re.compile(r"(?<=\d)\s*(?:'(?!')|feet\b|foot\b|ft\b\.?)")
_POSSESSIVE = re.compile(r"(?<=[a-z])\s*'s\b")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+")

STOPWORDS = frozenset({"a", "an", "and", "the", "of", "for", "with", "in", "on", "to", "by", "or"})

# British to American, limited to words that occur in home goods.
SPELLING = {
    "aluminium": "aluminum",
    "armour": "armor",
    "centre": "center",
    "centrepiece": "centerpiece",
    "colour": "color",
    "coloured": "colored",
    "fibre": "fiber",
    "grey": "gray",
    "jewellery": "jewelry",
    "metre": "meter",
    "mould": "mold",
    "moulding": "molding",
    "organiser": "organizer",
    "plough": "plow",
    "storey": "story",
}
IRREGULAR_PLURALS = {
    "children": "child",
    "knives": "knife",
    "leaves": "leaf",
    "loaves": "loaf",
    "men": "man",
    "mice": "mouse",
    "shelves": "shelf",
    "wolves": "wolf",
    "women": "woman",
}
# Singular already, although they end in a plural-looking s.
INVARIANT = frozenset({"bus", "canvas", "christmas", "gas", "lens", "plus", "series", "stainless"})


def normalize(text: str) -> str:
    """Lowercase, strip accents, unify quotes and spell out inch and foot marks."""
    folded = unicodedata.normalize("NFKD", text.translate(_PUNCTUATION).lower())
    plain = "".join(char for char in folded if not unicodedata.combining(char))
    plain = _POSSESSIVE.sub("", plain)
    plain = _INCH.sub(" inch ", plain)
    plain = _FOOT.sub(" ft ", plain)
    return re.sub(r"\s+", " ", plain).strip()


def singular(token: str) -> str:
    """Strip a regular plural ending.

    A conservative rule rather than a Porter stemmer: conflating "curtained" with "curtain" or
    "bedding" with "bed" would merge different product types.
    """
    if token in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[token]
    if token in INVARIANT or len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    return token[:-1]


def tokenize(text: object) -> list[str]:
    """Split into index terms; non-text (a missing cell) gives no terms.

    Hyphens split, so "mid-century" matches "mid century". Digits and letters split, so "54in" and
    "54 in" agree.
    """
    if not isinstance(text, str):
        return []
    tokens = []
    for raw in _TOKEN.findall(normalize(text)):
        token = singular(SPELLING.get(raw, raw))
        token = SPELLING.get(token, token)
        if token not in STOPWORDS:
            tokens.append(token)
    return tokens
