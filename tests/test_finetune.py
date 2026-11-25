"""The whole fine-tuning job on synthetic ESCI files and a tiny random model. No download."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from esci_fixtures import COLORS, ITEMS, SETTINGS, data, esci, rebuilt  # noqa: F401

from shelfrank.dense import esci_data as ed

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


def test_run_log_keeps_sessions_resumes_and_every_loss(base, data, tmp_path, monkeypatch):
    torch.manual_seed(5)
    rounds = ed.make_triplets(data.train, data.passages, CONFIG.seed)
    batches = ed.make_batches(rounds, CONFIG.batch_size, CONFIG.seed)

    def go(folder):
        model, tokenizer = ft.load_base(base)
        saved = tune.Checkpoints(folder, {})
        record = tune.RunLog(folder / "run_log.json")
        args = (model, tokenizer, data, batches, CONFIG, saved, "find: ", "cpu", lambda m: None)
        return tune.train(*args, record)

    go(tmp_path / "whole")
    real, calls = tune.contrastive_loss, []

    def crash(*args):
        calls.append(1)
        if len(calls) == 7:
            raise RuntimeError("session ended")
        return real(*args)

    monkeypatch.setattr(tune, "contrastive_loss", crash)
    with pytest.raises(RuntimeError, match="session ended"):
        go(tmp_path / "parts")
    monkeypatch.setattr(tune, "contrastive_loss", real)
    go(tmp_path / "parts")
    go(tmp_path / "parts")  # already finished: not a session

    whole = json.loads((tmp_path / "whole/run_log.json").read_text(encoding="utf-8"))
    parts = json.loads((tmp_path / "parts/run_log.json").read_text(encoding="utf-8"))
    assert [s["ended"] for s in whole["sessions"]] == ["finished"] and whole["resumes"] == []
    assert [s["ended"] for s in parts["sessions"]] == ["still running or lost", "finished"]
    assert [s["first_step"] for s in parts["sessions"]] == [0, 4]
    assert [r["message"] for r in parts["resumes"]] == ["resumed at step 4"]
    assert len(whole["loss_per_step"]) == 12
    assert parts["loss_per_step"] == pytest.approx(whole["loss_per_step"], abs=1e-6)
    assert all(s["seconds"] >= 0 for s in parts["sessions"])


def test_run_log_records_the_time_limit(base, data, tmp_path):
    config = tune.TrainConfig(batch_size=8, max_minutes=0.0)
    model, tokenizer = ft.load_base(base)
    rounds = ed.make_triplets(data.train, data.passages, config.seed)
    batches = ed.make_batches(rounds, 8, config.seed)
    record = tune.RunLog(tmp_path / "run_log.json")
    saved = tune.Checkpoints(tmp_path / "c", {})
    args = (model, tokenizer, data, batches, config, saved, "", "cpu", lambda m: None)
    tune.train(*args, record)
    assert tune.RunLog(record.path).body["sessions"][0]["ended"] == "time limit"


def test_model_card_says_not_logged_and_never_estimates():
    assert ft.run_lines({}) == "Wall time: not logged. Resume history: not logged."
    run = {"sessions": [{"seconds": 90.0}, {"seconds": 30.0}], "resumes": [{"message": "m"}]}
    assert ft.run_lines({"run": run}) == "Training minutes per session: 1.5, 0.5. Resumes: m."


def test_held_out_interval_needs_per_query_scores():
    means = {"held_out_ndcg_before": 0.5, "held_out_ndcg_after": 0.6}
    assert "no confidence interval" in ft.held_out_lines(means)
    rows = {
        "held_out_before_per_query": {"a": 0.4, "b": 0.6},
        "held_out_after_per_query": {"a": 0.5, "b": 0.7},
    }
    assert "+0.1000 [+0.1000, +0.1000]" in ft.held_out_lines(means | rows)


def test_reference_texts_are_twenty_and_resemble_no_wands_query():
    from shelfrank.config import REPO_ROOT
    from shelfrank.eval.leakage import exact_matches, load_key_hashes

    assert len(set(ft.REFERENCE_TEXTS)) == 20
    keys = load_key_hashes(REPO_ROOT / "data/splits/wands_query_keys.json")
    assert exact_matches(ft.REFERENCE_TEXTS, keys) == []


@pytest.mark.full_data
def test_reference_texts_are_not_near_duplicates_of_wands_queries(raw_dir, data_cfg):
    from shelfrank.data.wands import load_queries
    from shelfrank.eval.leakage import near_matches

    threshold = data_cfg["splits"]["near_duplicate_jaccard"]
    assert near_matches(ft.REFERENCE_TEXTS, load_queries(raw_dir)["query"], threshold) == []


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
    assert results["run"]["sessions"][0]["ended"] == "finished"
    assert len(results["run"]["loss_per_step"]) == 12
    assert results["held_out_before_per_query"].keys() == results["held_out_after_per_query"].keys()
    reference = json.loads((folder / "reference_embeddings.json").read_text(encoding="utf-8"))
    assert np.array(reference["vectors"]).shape == (20, 32)
    card = (folder / "README.md").read_text(encoding="utf-8")
    assert "Training minutes per session" in card and "Paired difference" in card
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
                f"SETTINGS.update({SETTINGS!r})\n"
                "TRAINING.update(batch_size=8, checkpoint_every=4)\n"
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
