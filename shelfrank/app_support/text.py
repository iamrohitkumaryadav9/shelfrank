"""Showing catalogue text in Markdown."""

import re

_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-!|>~$:<&])")


def plain(text: str) -> str:
    """Escape Markdown, so a product name is shown as written and can never become markup or a link.

    Streamlit already escapes HTML unless it is told not to, and this project never tells it.
    """
    return _SPECIAL.sub(r"\\\1", text)
