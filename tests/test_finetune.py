"""The whole fine-tuning job on synthetic ESCI files and a tiny random model. No download."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from shelfrank.config import REPO_ROOT, load_config
from shelfrank.dense import esci_data as ed
from shelfrank.dense.finetune_notebook import EMBEDDED, build, commit_in
from shelfrank.qu.seed_notebook import dumps

SETTINGS = {
    "locale": "us",
    "split": "train",
    "n_held_out": 4,
    "max_bullet_chars": 20,
    "near_duplicate_jaccard": 0.75,
    "seed": 1,
}
COLORS = "red blue green black white pink brown gray teal navy".split()
ITEMS = "lamp sofa rug desk shelf clock".split()


@pytest.fixture
def esci(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    examples, products = [], []
    for i, (color, item) in enumerate((c, t) for t in ITEMS for c in COLORS):
        for j, label in enumerate("EESI"):
            product = f"P{i}_{j}"
            row = {
                "query_id": i,
                "query": f"{color} {item} number{chr(97 + i % 26)}{chr(97 + i // 26)}",
                "product_id": product,
                "product_locale": "us",
                "esci_label": label,
                "small_version": 1,
                "split": "train",
            }
            examples.append(row)
            title = f"{color} {item}" if label == "E" else "other thing"
            products.append(
                {
                    "product_id": product,
                    "product_locale": "us",
                    "product_title": title,
                    "product_brand": "Acme",
                    "product_color": None,
                    "product_bullet_point": "x" * 100,
                }
            )
    base = {"product_id": "Z", "product_locale": "us", "esci_label": "E"}
    examples += [
        base | {"query_id": 900, "query": "large one", "small_version": 0, "split": "train"},
        base | {"query_id": 901, "query": "test one", "small_version": 1, "split": "test"},
        base
        | {
            "query_id": 902,
            "query": "spanish one",
            "small_version": 1,
            "split": "train",
            "product_locale": "es",
        },
    ]
    pq.write_table(pa.Table.from_pylist(examples), tmp_path / "examples.parquet")
    pq.write_table(pa.Table.from_pylist(products), tmp_path / "products.parquet")
    return tmp_path / "examples.parquet", tmp_path / "products.parquet"


@pytest.fixture
def data(esci):
    return ed.prepare(*esci, ["red lamp numberaa"], SETTINGS)


def test_only_small_us_training_queries_are_read(esci):
    queries, judged = ed.read_judgments(esci[0], "us", "train")
    assert len(queries) == len(judged) == 60
    assert not {"900", "901", "902"} & set(queries)


def test_lookalikes_of_protected_queries_are_removed_everywhere(data):
    assert "red lamp numberaa" not in data.queries.values()
    assert data.report["dropped"] == {"resembles a protected query": 1}


def test_held_out_queries_are_not_trained_on(data):
    assert len(data.held_out) == 4 and len(data.train) == 55
    assert not set(data.held_out) & set(data.train)


def test_passage_order_and_truncation():
    row = {
        "product_title": " A  lamp ",
        "product_brand": "Acme",
        "product_color": None,
        "product_bullet_point": "abcdef",
    }
    assert ed.passage(row, 3) == "A lamp . Acme . abc"


def test_data_survives_a_round_trip(data, tmp_path):
    path = tmp_path / "d.json"
    data.save(path)
    assert ed.TrainingData.load(path) == data and not path.with_suffix(".partial").exists()


def test_triplets_pair_exact_with_irrelevant_of_the_same_query(data):
    for group in ed.make_triplets(data.train, data.passages, 1):
        assert len({t.query for t in group}) == len(group)
        for t in group:
            assert data.train[t.query][t.positive] == "E"
            assert data.train[t.query][t.negative] == "I" and t.hard


def test_query_without_irrelevant_gets_a_marked_random_negative():
    rounds = ed.make_triplets({"q": {"a": "E"}}, {"a": "x", "b": "y"}, 1)
    assert rounds[0][0].hard is False


def test_batches_are_full_have_distinct_queries_and_are_reproducible(data):
    rounds = ed.make_triplets(data.train, data.passages, 1)
    batches = ed.make_batches(rounds, 8, 1)
    assert batches == ed.make_batches(rounds, 8, 1) and len(batches) == 12
    assert all(len({t.query for t in b}) == len(b) == 8 for b in batches)


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from scripts import finetune_dense as ft  # noqa: E402
from shelfrank.dense import finetune as tune  # noqa: E402

CONFIG = tune.TrainConfig(batch_size=8, learning_rate=1e-2, checkpoint_every=4, log_every=3)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A tiny random BERT with the same file layout as the real base model."""
    from transformers import BertConfig, BertModel, BertTokenizerFast

    folder = tmp_path_factory.mktemp("base")
    letters = [chr(97 + i) for i in range(26)]
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *COLORS, *ITEMS, "other", "thing", "."]
    (folder / "vocab.txt").write_text(
        "\n".join([*words, *letters, *[f"##{x}" for x in letters]]), encoding="utf-8"
    )
    BertTokenizerFast(str(folder / "vocab.txt")).save_pretrained(folder)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
    )
    BertModel(config, add_pooling_layer=False).save_pretrained(folder)
    return folder


def test_loss_by_hand_and_blocked_candidates_do_not_count():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    none = torch.zeros((2, 4), dtype=torch.bool)
    loss = tune.contrastive_loss(q, q, -q, none, 1.0)
    row = math.exp(1) / (math.exp(1) + 1 + math.exp(-1) + 1)
    assert loss.item() == pytest.approx(-math.log(row))
    blocked = none.clone()
    blocked[0, 1] = True
    assert tune.contrastive_loss(q, q, -q, blocked, 1.0) < loss


def test_other_exact_products_of_the_query_are_blocked_but_not_its_own_positive():
    batch = [ed.Triplet("q1", "a", "n", True), ed.Triplet("q2", "b", "m", True)]
    judged = {"q1": {"a": "E", "b": "E", "n": "I"}, "q2": {"b": "E", "m": "I", "n": "S"}}
    out = tune.blocked_candidates(batch, judged)
    assert out.tolist() == [[False, True, False, False], [False, False, False, False]]


def test_learning_rate_warms_up_and_decays_to_zero():
    values = [tune.learning_rate_factor(s, 100, 0.1) for s in range(101)]
    assert values[0] == pytest.approx(0.1) and max(values) == 1.0 and values[100] == 0.0
    assert values[:10] == sorted(values[:10]) and values[10:] == sorted(values[10:], reverse=True)


def test_ndcg_by_hand():
    assert tune.ndcg([1.0, 0.0]) == 1.0 and tune.ndcg([0.0]) == 0.0
    assert tune.ndcg([0.0, 1.0]) == pytest.approx(1 / math.log2(3))


def run(base, data, folder, config=CONFIG, batches=None):
    torch.manual_seed(5)
    model, tokenizer = ft.load_base(base)
    if batches is None:
        rounds = ed.make_triplets(data.train, data.passages, config.seed)
        batches = ed.make_batches(rounds, config.batch_size, config.seed)
    checkpoints = tune.Checkpoints(folder, {"run": 1})
    logs = []
    out = tune.train(
        model, tokenizer, data, batches, config, checkpoints, "find: ", "cpu", logs.append
    )
    return model, out, logs


def test_resumed_training_equals_uninterrupted_training(base, data, tmp_path, monkeypatch):
    whole, out, _ = run(base, data, tmp_path / "whole")
    assert out["finished"] and out["total"] == 12

    real, calls = tune.contrastive_loss, []

    def crash(*args):
        calls.append(1)
        if len(calls) == 7:
            raise RuntimeError("session ended")
        return real(*args)

    monkeypatch.setattr(tune, "contrastive_loss", crash)
    with pytest.raises(RuntimeError, match="session ended"):
        run(base, data, tmp_path / "parts")
    monkeypatch.setattr(tune, "contrastive_loss", real)
    parts, out, logs = run(base, data, tmp_path / "parts")
    assert out["finished"] and "resumed at step 4 of 12" in logs[0]
    for a, b in zip(whole.parameters(), parts.parameters(), strict=True):
        assert torch.equal(a, b)


def test_time_limit_saves_and_stops(base, data, tmp_path):
    config = tune.TrainConfig(batch_size=8, max_minutes=0.0)
    _, out, logs = run(base, data, tmp_path, config)
    assert out["step"] == 0 and not out["finished"] and "time limit" in logs[-1]


def test_training_again_after_the_end_does_nothing(base, data, tmp_path):
    run(base, data, tmp_path)
    _, out, _ = run(base, data, tmp_path)
    assert out["finished"] and out["step"] == 12


def test_checkpoints_of_another_run_are_refused(tmp_path):
    tune.Checkpoints(tmp_path, {"lr": 1}).save(3, {"step": 3})
    with pytest.raises(ValueError, match="other settings"):
        tune.Checkpoints(tmp_path, {"lr": 2}).latest()


def test_a_crash_while_saving_leaves_the_previous_checkpoint(tmp_path, monkeypatch):
    saved = tune.Checkpoints(tmp_path, {})
    saved.save(1, {"step": 1})
    saved.save(2, {"step": 2})
    monkeypatch.setattr(torch, "save", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        saved.save(3, {"step": 3})
    assert saved.latest()["step"] == 2


def test_only_two_checkpoint_files_ever_exist(tmp_path):
    saved = tune.Checkpoints(tmp_path, {})
    for step in range(1, 6):
        saved.save(step, {"step": step})
    assert len(list(tmp_path.glob("*.pt"))) == 2 and saved.latest()["step"] == 5


def test_encode_returns_unit_vectors_in_input_order(base):
    model, tokenizer = ft.load_base(base)
    texts = ["red lamp with a long name", "sofa", "blue rug"]
    together = tune.encode(model, tokenizer, texts, 32, "cpu", batch_size=2)
    alone = tune.encode(model, tokenizer, ["sofa"], 32, "cpu")
    assert torch.allclose(together[1], alone[0], atol=1e-5)
    assert torch.allclose(together.norm(dim=1), torch.ones(3))


@pytest.fixture
def paths(tmp_path, esci, monkeypatch):
    queries = tmp_path / "wands.csv"
    queries.write_text("query_id\tquery\n1\tred lamp numberaa\n", encoding="utf-8")
    files = {"examples": esci[0], "products": esci[1], "wands_queries": queries}

    def fetch(url, target, sha256):
        target.write_bytes(files[url].read_bytes())
        return True

    monkeypatch.setattr(ft, "ensure_file", fetch)
    return ft.Paths(tmp_path / "work", tmp_path / "keep")


SOURCES = {n: {"url": n, "sha256": ""} for n in ("examples", "products", "wands_queries")}


def test_whole_job_trains_improves_and_packages(base, paths, monkeypatch):
    pytest.importorskip("sentence_transformers")
    data = ft.stage_data(paths, SOURCES, SETTINGS)
    monkeypatch.setattr(ft, "ensure_file", lambda *a: pytest.fail("downloaded twice"))
    assert ft.stage_data(paths, SOURCES, SETTINGS) == data

    with pytest.raises(RuntimeError, match="not finished"):
        ft.trained_model(paths, base)
    out = ft.stage_train(paths, data, base, CONFIG, "find: ", "cpu")
    assert out["finished"] and out["history"][-1][1] < out["history"][0][1]
    ft.stage_evaluate(paths, data, base, CONFIG, "find: ", "cpu")
    monkeypatch.setattr(ft, "export_int8", lambda *a: (_ for _ in ()).throw(OSError("no onnx")))
    provenance = {"git_commit": "abc", "base_model": "m", "base_revision": "r"}
    archive = ft.stage_package(paths, data, base, CONFIG, "find: ", provenance)
    assert archive.exists()

    folder = paths.output / "bge_small_esci"
    results = json.loads((folder / "training.json").read_text(encoding="utf-8"))
    assert (
        results["hard_negative_share"] == 1.0
        and "no onnx" in results["onnx"]["bge_small_esci"]["error"]
    )
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert "commit abc" in card and "No WANDS data" in card
    model, tokenizer = ft.load_base(folder)
    trained, _ = ft.trained_model(paths, base)
    a = tune.encode(model, tokenizer, ["red lamp"], 32, "cpu")
    b = tune.encode(trained, tokenizer, ["red lamp"], 32, "cpu")
    assert torch.allclose(a, b, atol=1e-6)


def test_exported_graph_matches_pytorch(base, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from shelfrank.dense import onnx_export as ox

    model, tokenizer = ft.load_base(base)
    ox.export_fp32(model, tokenizer, tmp_path / "m.onnx")
    ours = ox.onnx_encode(tmp_path / "m.onnx", tokenizer, ["red lamp", "a sofa"], 32)
    theirs = tune.encode(model, tokenizer, ["red lamp", "a sofa"], 32, "cpu").numpy()
    assert ox.agreement(theirs, ours)["min"] > 0.9999


def test_agreement_is_cosine():
    from shelfrank.dense.onnx_export import agreement

    a = np.array([[1.0, 0.0]])
    assert agreement(a, a * 3) == {"mean": 1.0, "min": 1.0}


NOTEBOOK = REPO_ROOT / "notebooks/finetune_bge_esci.ipynb"


def rebuilt():
    model = load_config("retrieval")["dense"]["bge_small"]
    committed = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return build(REPO_ROOT, load_config("esci"), model, commit_in(committed))


def test_committed_notebook_matches_the_tested_modules():
    assert NOTEBOOK.read_text(encoding="utf-8") == dumps(rebuilt())


def test_notebook_cells_compile_and_hold_no_secret_or_output():
    for cell in rebuilt()["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            if source.startswith("%%writefile"):
                source = source.split("\n", 1)[1]
            compile(source, "cell", "exec")
        assert "API_KEY" not in source


def test_notebook_cells_run_outside_the_repository(base, esci, tmp_path):
    """Run the cells as Colab would, in a process that cannot see this repository's package."""
    queries = tmp_path / "wands.csv"
    queries.write_text("query_id\tquery\n1\tred lamp numberaa\n", encoding="utf-8")
    lines = [f"import shutil\nBASE = {str(base)!r}\nFILES = {{'examples': {str(esci[0])!r},"]
    lines.append(f"'products': {str(esci[1])!r}, 'wands_queries': {str(queries)!r}}}")
    for cell in rebuilt()["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            continue
        if source.startswith("%%writefile"):
            name, body = source.split("\n", 1)
            path = name.split()[1]
            lines.append(f"Path({path!r}).write_text({body!r}, encoding='utf-8')")
            continue
        source = source.replace("subprocess.run([sys.executable", "print([sys.executable").replace(
            ", *PACKAGES], check=True)", ", *PACKAGES])"
        )
        if "stage_data" in source:
            lines.append(
                f"SETTINGS.update({SETTINGS!r})\nTRAINING.update(batch_size=8, checkpoint_every=4)\n"
                "SOURCES = {n: {'url': n, 'sha256': ''} for n in FILES}\n"
                "from scripts import finetune_dense as ft\n"
                "ft.ensure_file = lambda u, t, s: shutil.copyfile(FILES[u], t)\n"
                "ft.download_base = lambda *a: Path(BASE)"
            )
        lines.append(source)
    script = tmp_path / "cells.py"
    script.write_text("\n".join(lines), encoding="utf-8")
    env = os.environ | {
        "SHELFRANK_ALLOW_CPU": "1",
        "SHELFRANK_WORK": str(tmp_path / "w"),
        "SHELFRANK_KEEP": str(tmp_path / "k"),
        "PYTHONPATH": "",
    }
    done = subprocess.run(
        [sys.executable, "-P", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "finished: True" in done.stdout and (tmp_path / "k/bge_small_esci.zip").exists()
    assert len(EMBEDDED) == 8
