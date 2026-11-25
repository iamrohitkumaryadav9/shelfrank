from esci_fixtures import NOTEBOOK, data, esci, rebuilt  # noqa: F401

from shelfrank.dense import esci_data as ed
from shelfrank.dense.finetune_notebook import EMBEDDED
from shelfrank.qu.seed_notebook import dumps


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


def test_every_embedded_module_exists_and_is_listed_once():
    assert len(set(EMBEDDED)) == len(EMBEDDED) == 8
