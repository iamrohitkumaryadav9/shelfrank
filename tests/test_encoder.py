import numpy as np
import pytest

from shelfrank.dense.encoder import (
    StaleCacheError,
    encode_cached,
    fingerprint,
    model_identity,
    product_text,
    product_texts,
)

IDENTITY = {"model": "m", "revision": "r", "max_seq_length": 8}
TEXTS = ["ccc", "a", "bb", "dddd", "ee"]


def lengths(texts):
    return np.array([[len(t), ord(t[0])] for t in texts], dtype=np.float32)


def run(texts, directory, encode=lengths, identity=IDENTITY, chunk=2):
    return encode_cached(texts, directory, identity, encode, chunk=chunk, log=lambda _: None)


def test_rows_come_back_in_input_order(tmp_path):
    np.testing.assert_array_equal(run(TEXTS, tmp_path), lengths(TEXTS))


def test_second_call_does_not_encode(tmp_path):
    first = run(TEXTS, tmp_path)

    def explode(_):
        raise AssertionError("should have used the cache")

    np.testing.assert_array_equal(run(TEXTS, tmp_path, explode), first)


def test_interrupted_run_resumes_without_repeating_chunks(tmp_path):
    seen = []

    def flaky(texts):
        if len(seen) == 1:
            raise RuntimeError("power cut")
        seen.append(texts)
        return lengths(texts)

    with pytest.raises(RuntimeError, match="power cut"):
        run(TEXTS, tmp_path, flaky)
    resumed = []

    def record(texts):
        resumed.extend(texts)
        return lengths(texts)

    np.testing.assert_array_equal(run(TEXTS, tmp_path, record), lengths(TEXTS))
    assert len(resumed) == len(TEXTS) - len(seen[0])


def test_part_files_are_removed(tmp_path):
    run(TEXTS, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["embeddings.npy", "manifest.json"]


def test_changed_text_is_refused(tmp_path):
    run(TEXTS, tmp_path)
    with pytest.raises(StaleCacheError):
        run([*TEXTS[:-1], "changed"], tmp_path)


def test_changed_model_is_refused(tmp_path):
    run(TEXTS, tmp_path)
    with pytest.raises(StaleCacheError):
        run(TEXTS, tmp_path, identity={**IDENTITY, "revision": "other"})


def test_fingerprint_sees_boundaries_and_order():
    assert fingerprint(["ab", "c"]) != fingerprint(["a", "bc"])
    assert fingerprint(["a", "b"]) != fingerprint(["b", "a"])


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (("bed", "Beds", "soft"), "bed . Beds . soft"),
        (("bed", float("nan"), None), "bed"),
        (("chair", "A|B", " x "), "chair . A , B . x"),
    ],
)
def test_product_text(row, expected):
    assert product_text(*row) == expected


def test_every_fixture_product_has_text_starting_with_its_name(sample):
    texts = product_texts(sample["products"])
    names = sample["products"]["product_name"]
    assert all(t.startswith(n) for t, n in zip(texts, names, strict=True))


def test_local_model_is_pinned_by_the_hash_of_its_weights(tmp_path):
    cfg = {"model": "m", "revision": "local", "max_seq_length": 8}
    with pytest.raises(FileNotFoundError, match="COLAB"):
        model_identity(cfg, tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m/model.safetensors").write_bytes(b"one")
    first = model_identity(cfg, tmp_path)["revision"]
    (tmp_path / "m/model.safetensors").write_bytes(b"two")
    assert first.startswith("sha256:") and model_identity(cfg, tmp_path)["revision"] != first


def test_published_model_keeps_its_revision(tmp_path):
    cfg = {"model": "a/b", "revision": "abc", "max_seq_length": 8, "query_prefix": "x"}
    assert model_identity(cfg, tmp_path) == {"model": "a/b", "revision": "abc", "max_seq_length": 8}


@pytest.fixture
def tiny_model(tmp_path):
    """A random BERT saved without a pooler, in the layout of our fine-tuned model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizerFast

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "red", "lamp", "blue", "sofa"]
    (tmp_path / "vocab.txt").write_text("\n".join(words), encoding="utf-8")
    BertTokenizerFast(str(tmp_path / "vocab.txt")).save_pretrained(tmp_path / "bert")
    torch.manual_seed(0)
    shape = {"hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2}
    config = BertConfig(vocab_size=len(words), intermediate_size=16, **shape)
    BertModel(config, add_pooling_layer=False).save_pretrained(tmp_path / "bert")
    body = models.Transformer(
        str(tmp_path / "bert"), model_args={"add_pooling_layer": False}, max_seq_length=8
    )
    pooling = models.Pooling(16, pooling_mode="cls")
    SentenceTransformer(modules=[body, pooling, models.Normalize()], device="cpu").save(
        str(tmp_path / "m")
    )
    return tmp_path


def test_local_model_loads_without_a_pooler_and_without_a_warning(tiny_model, monkeypatch, caplog):
    import logging

    from shelfrank.dense import encoder as en

    monkeypatch.setattr(en, "REPO_ROOT", tiny_model)
    logging.getLogger("transformers").propagate = True
    encode = en.load_sentence_transformer("m", "sha256:x", 8)
    body = encode.__closure__[0].cell_contents[0].auto_model
    assert body.pooler is None and "newly initialized" not in caplog.text


def test_embeddings_do_not_depend_on_the_pooler(tiny_model, monkeypatch):
    """Bit-identical with the layer removed, freshly random, and overwritten with noise."""
    import torch
    from sentence_transformers import SentenceTransformer

    from shelfrank.dense import encoder as en

    monkeypatch.setattr(en, "REPO_ROOT", tiny_model)
    texts = ["red lamp", "blue sofa", "lamp"]
    removed = en.load_sentence_transformer("m", "sha256:x", 8)(texts)
    with_layer = SentenceTransformer(str(tiny_model / "m"), device="cpu")
    pooler = with_layer[0].auto_model.pooler
    assert pooler is not None
    np.testing.assert_array_equal(with_layer.encode(texts, normalize_embeddings=True), removed)
    with torch.no_grad():
        pooler.dense.weight.normal_(0, 100)
        pooler.dense.bias.normal_(0, 100)
    np.testing.assert_array_equal(with_layer.encode(texts, normalize_embeddings=True), removed)
