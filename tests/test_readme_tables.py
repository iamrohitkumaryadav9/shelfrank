import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.readme_tables import MissingBlockError, limitations, replace_block

TEXT = "a\n<!-- begin:x -->\nold\n<!-- end:x -->\nb"


def test_replaces_only_the_named_block():
    assert replace_block(TEXT, "x", "new") == "a\n<!-- begin:x -->\nnew\n<!-- end:x -->\nb"


def test_replacement_is_idempotent():
    once = replace_block(TEXT, "x", "new")
    assert replace_block(once, "x", "new") == once


def test_backslashes_in_the_body_are_literal():
    assert "a\\1|b" in replace_block(TEXT, "x", "a\\1|b")


@pytest.mark.parametrize("text", ["no markers", TEXT + TEXT])
def test_missing_or_repeated_markers_fail(text):
    with pytest.raises(MissingBlockError):
        replace_block(text, "x", "new")


def test_committed_readme_is_up_to_date():
    audit = json.loads((REPO_ROOT / "results/data_audit.json").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert replace_block(readme, "limitations", limitations(audit)) == readme
