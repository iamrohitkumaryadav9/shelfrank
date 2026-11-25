import json

import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.readme_tables import MissingBlockError, findings, limitations, replace_block

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


def load(name):
    return json.loads((REPO_ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def test_committed_readme_is_up_to_date():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert replace_block(readme, "limitations", limitations(load("data_audit"))) == readme
    body = findings(load("retrieval_dev"), load("bm25_tuning"))
    assert replace_block(readme, "findings", body) == readme


def test_prose_in_the_limitations_holds():
    path = REPO_ROOT / "data/taxonomy.json"
    classes = json.loads(path.read_text(encoding="utf-8"))["classes"]
    assert not [c for c in classes if "blanket" in c.lower()] and "Bedding Sets" in classes
    assert set(load("slot_evidence")["patterns"].values()) == {0}


def test_prose_in_the_findings_holds():
    data, tuning = load("retrieval_dev"), load("bm25_tuning")
    dense = data["systems"]["dense_bge_small"]["versus_baseline"]
    assert dense["ndcg10"]["mean"] < 0 < dense["condensed_ndcg10"]["mean"]
    pool = data["judged_by_class"]["dense_bge_small"]
    assert pool["out_of_class_judged"] < pool["in_class_judged"]
    assert tuning["dev_ndcg10_held_out"] < tuning["dev_ndcg10_tuned_in_sample"]
    assert tuning["weights"]["class"] == 0 < tuning["weights"]["hierarchy"]
