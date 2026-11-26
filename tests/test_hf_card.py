import json

from shelfrank.config import REPO_ROOT
from shelfrank.dense.hf_card import card


def load(name):
    return json.loads((REPO_ROOT / f"results/{name}.json").read_text(encoding="utf-8"))


def test_committed_card_is_generated_and_states_what_it_must():
    text = card(
        load("finetune_esci"), load("dense_dev"), load("final_run"), "iamrohitkumaryadav9/shelfrank"
    )
    assert (REPO_ROOT / "docs/hf_model_card.md").read_text(encoding="utf-8") == text
    assert text.startswith("---\nlicense: mit\n") and "datasets:" not in text
    for phrase in (
        "No WANDS data was used for training",
        "8a9fc4f",
        "were not logged",
        "384 held-out WANDS",
    ):
        assert phrase in text
