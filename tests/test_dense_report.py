import json

import numpy as np
import pytest

from shelfrank.config import REPO_ROOT
from shelfrank.eval.claims import GAIN, LOSS, NONE, UNCLEAR
from shelfrank.eval.dense_report import NAMES, encoder_choice, render, serving_choice


@pytest.mark.parametrize(
    ("dense", "hybrid", "expected"),
    [
        (GAIN, GAIN, "bge_small_esci"),
        (GAIN, NONE, "bge_small_esci"),
        (GAIN, UNCLEAR, "bge_small_esci"),
        (GAIN, LOSS, "bge_small"),
        (NONE, GAIN, "bge_small"),
        (UNCLEAR, GAIN, "bge_small"),
        (LOSS, LOSS, "bge_small"),
    ],
)
def test_encoder_rule(dense, hybrid, expected):
    assert encoder_choice(dense, hybrid) == expected


@pytest.mark.parametrize(
    ("int8", "expected"), [(0.704, "int8"), (0.696, "int8"), (0.694, "fp32"), (0.706, "fp32")]
)
def test_serving_rule_is_symmetric(int8, expected):
    assert serving_choice(int8, 0.7) == expected


def test_rules_are_in_the_protocol():
    text = " ".join((REPO_ROOT / "docs/PROTOCOL.md").read_text(encoding="utf-8").split())
    assert "within 0.005 of PyTorch's" in text
    assert "is not a loss against hybrid RRF built with off-the-shelf" in text


@pytest.fixture(scope="module")
def data():
    return json.loads((REPO_ROOT / "results/dense_dev.json").read_text(encoding="utf-8"))


def test_committed_page_is_up_to_date(data):
    assert (REPO_ROOT / "docs/dense_dev.md").read_text(encoding="utf-8") == render(data)


def test_committed_choices_follow_the_rules(data):
    first, second = data["differences"][:2]
    assert (first["system"], first["baseline"]) == ("dense_bge_small_esci", "dense_bge_small")
    assert (second["system"], second["baseline"]) == ("rrf_bge_small_esci", "rrf_bge_small")
    assert data["encoder"] == encoder_choice(first["verdict"], second["verdict"])
    for kinds in data["serving"].values():
        formats = kinds["formats"]
        pair = formats["int8"]["ndcg10"], formats["pytorch"]["ndcg10"]
        assert kinds["choice"] == serving_choice(*pair)


def test_off_the_shelf_numbers_match_phase_1(data):
    old = json.loads((REPO_ROOT / "results/retrieval_dev.json").read_text(encoding="utf-8"))
    for new, name in (("dense_bge_small", "dense_bge_small"), ("rrf_bge_small", "hybrid_rrf")):
        expected = old["systems"][name]["summary"]["ndcg10"]["mean"]
        assert data["systems"][new]["ndcg10"]["mean"] == pytest.approx(expected, abs=1e-9)
    assert data["split"] == "dev" and data["n_queries"] == 96 and set(data["systems"]) == set(NAMES)


def test_onnx_encoder_returns_the_unit_vector_of_one_text(monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    from shelfrank.dense import serving

    class Session:
        def __init__(self, path, providers):
            assert providers == ["CPUExecutionProvider"]

        def run(self, outputs, feed):
            assert outputs == ["embedding"] and set(feed) == set(serving.INPUTS)
            assert all(v.dtype == np.int64 for v in feed.values())
            return [np.array([[3.0, 4.0]])]

    def tokenizer(texts, **options):
        assert options["max_length"] == 8
        return {name: np.ones((1, 2), dtype=np.int32) for name in serving.INPUTS}

    monkeypatch.setattr(ort, "InferenceSession", Session)
    assert serving.OnnxEncoder("m.onnx", tokenizer, 8)("x").tolist() == [0.6, 0.8]


def test_phase3_page_is_generated_and_every_check_passed():
    from shelfrank.dense.checks_report import render as render_checks

    def load(name):
        return json.loads((REPO_ROOT / f"results/{name}.json").read_text(encoding="utf-8"))

    data, training = load("phase3_checks"), load("finetune_esci")
    page = (REPO_ROOT / "docs/phase3_checks.md").read_text(encoding="utf-8")
    assert page == render_checks(data, training)
    for model in data["pooling"].values():
        assert model["identical_with_layer"] and model["identical_noised"]
        assert model["identical_removed"] and model["max_difference_from_first_token"] < 1e-6
    for model in data["embeddings"].values():
        assert model["manifest_matches"] and model["reencoded_min_cosine"] > 0.999
        assert model["shape"] == [42994, 384]
    source = data["provenance"]
    assert all(source["code_files_match"].values()) and source["every_embedded_file_recorded"]
    assert source["training_settings_match"] and source["data_settings_match"]
