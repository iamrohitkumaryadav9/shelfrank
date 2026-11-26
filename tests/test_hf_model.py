import json

from shelfrank.config import REPO_ROOT


def test_published_encoder_is_recorded_and_linked():
    record = json.loads((REPO_ROOT / "results/hf_model.json").read_text(encoding="utf-8"))
    assert record["public"] and not record["gated"] and len(record["revision"]) == 40
    assert "model.safetensors" in record["matches_local_artifact"]
    assert all(len(h) == 64 for h in record["sha256"].values())
    assert record["url"] in (REPO_ROOT / "README.md").read_text(encoding="utf-8")
