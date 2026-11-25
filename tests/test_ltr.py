import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from shelfrank.config import load_config
from shelfrank.eval.access import ProtocolError
from shelfrank.eval.claims import GAIN, LOSS, NONE, UNCLEAR, compare, holm, verdict
from shelfrank.ltr.ablation import ablations, alternatives, training_choices, variants
from shelfrank.ltr.cross_validation import grouped_folds, out_of_fold
from shelfrank.ltr.dataset import (
    JUDGED_ONLY,
    UNJUDGED_ZERO,
    RankingData,
    build_data,
    group_sizes,
)
from shelfrank.ltr.gates import allowed_groups, slot_blockers
from shelfrank.ltr.model import (
    CV,
    POINTWISE,
    LtrConfig,
    Ranker,
    lightgbm_params,
    split_queries,
    to_run,
)
from shelfrank.ltr.selection import bug_fixes, output_path, select
from shelfrank.stats.bootstrap import Interval


def interval(low, high):
    return Interval((low + high) / 2, low, high, 10)


@pytest.mark.parametrize(
    ("standard", "condensed", "expected"),
    [
        ((0.01, 0.03), (0.01, 0.03), GAIN),
        ((0.01, 0.03), (-0.01, 0.03), UNCLEAR),
        ((-0.01, 0.03), (0.01, 0.03), UNCLEAR),
        ((0.01, 0.03), (-0.03, -0.01), UNCLEAR),
        ((0.0, 0.03), (0.01, 0.03), UNCLEAR),
        ((-0.03, -0.01), (-0.03, -0.01), LOSS),
        ((-0.01, 0.01), (-0.01, 0.01), NONE),
    ],
)
def test_a_gain_needs_both_metrics(standard, condensed, expected):
    assert verdict(interval(*standard), interval(*condensed)) == expected


def scores(values):
    return {q: {"ndcg10": v, "condensed_ndcg10": v} for q, v in enumerate(values)}


def test_compare_is_paired_and_signed():
    out = compare(scores([0.5, 0.6, 0.7]), scores([0.4, 0.5, 0.6]), n_resamples=200)
    assert out["ndcg10"]["mean"] == pytest.approx(0.1) and out["verdict"] == GAIN
    assert compare(scores([0.1] * 3), scores([0.2] * 3), n_resamples=200)["verdict"] == LOSS
    with pytest.raises(ValueError, match="same queries"):
        compare(scores([0.1]), scores([0.1, 0.2]))


def test_holm_by_hand_and_monotone():
    out = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"a": 0.9, "b": 0.8})["b"] == 1.0


def make_data(n_queries=24, n_products=20, seed=0):
    """Grade follows g1_signal; a third of the products are unjudged."""
    rng = np.random.default_rng(seed)
    rows, pairs = [], []
    qrels = {q: {} for q in range(n_queries)}
    for q in range(n_queries):
        for p in range(n_products):
            grade = int(rng.integers(0, 3))
            pairs.append((q, p))
            rows.append((grade + rng.normal(0, 0.3), rng.normal()))
            if p % 3:
                qrels[q][p] = grade
    frame = pd.DataFrame(pairs, columns=["query_id", "product_id"])
    features = pd.DataFrame(rows, columns=["g1_signal", "g2_noise"])
    return build_data(frame, features, qrels)


def config(**changes):
    base = LtrConfig(
        ("g1_signal", "g2_noise"),
        JUDGED_ONLY,
        params={"min_data_in_leaf": 5, "num_leaves": 7, "learning_rate": 0.1},
        rounds=30,
        patience=10,
        seeds=(1, 2),
    )
    return replace(base, **changes)


def test_grades_are_missing_when_unjudged_and_zero_only_as_labels():
    data = make_data()
    rows = np.arange(3)
    assert np.isnan(data.grade[0]) and data.labels(rows)[0] == 0


def test_variants_differ_only_in_unjudged_training_rows():
    data = make_data()
    judged, every = data.rows([0, 1], JUDGED_ONLY), data.rows([0, 1], UNJUDGED_ZERO)
    assert len(every) == 40 and set(judged) < set(every)
    assert np.isnan(data.grade[sorted(set(every) - set(judged))]).all()
    assert len(data.rows([0, 1])) == 40
    with pytest.raises(ValueError, match="unknown variant"):
        data.rows([0], "other")


def test_rows_must_be_grouped_by_query():
    with pytest.raises(ValueError, match="sorted"):
        RankingData(np.array([2, 1]), np.array([1, 1]), pd.DataFrame(), np.zeros(2))


def test_features_and_pairs_must_align():
    with pytest.raises(ValueError, match="length"):
        build_data(
            pd.DataFrame({"query_id": [1], "product_id": [1]}), pd.DataFrame({"g1_a": []}), {1: {}}
        )


@pytest.mark.parametrize(
    ("ids", "expected"), [([1, 1, 2], [2, 1]), ([], []), ([5], [1]), ([1, 2, 3], [1, 1, 1])]
)
def test_group_sizes(ids, expected):
    assert group_sizes(np.array(ids)) == expected


def test_validation_queries_come_from_the_training_queries_and_do_not_overlap():
    train, valid = split_queries(range(10), 0.2, 1)
    assert len(valid) == 2 and not set(train) & set(valid)
    assert sorted(train + valid) == list(range(10))
    assert split_queries(range(10), 0.2, 1) == (train, valid)


def test_run_breaks_ties_by_product_id():
    data = make_data(1, 4)
    run = to_run(data, np.arange(4), np.array([1.0, 2.0, 2.0, 0.0]))
    assert run == {0: [1, 2, 0, 3]}


def test_params_for_each_objective():
    ranking = lightgbm_params(config(label_gain=(0, 1, 3)), 5)
    assert ranking["label_gain"] == [0, 1, 3] and ranking["objective"] == "lambdarank"
    assert lightgbm_params(config(objective=POINTWISE), 5)["objective"] == "regression"
    with pytest.raises(ValueError, match="objective"):
        lightgbm_params(config(objective="x"), 5)


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"variant": UNJUDGED_ZERO},
        {"objective": POINTWISE},
        {"stopping": CV},
        {"stopping": CV, "objective": POINTWISE},
    ],
)
def test_ranker_learns_the_signal_and_is_deterministic(changes):
    pytest.importorskip("lightgbm")
    data = make_data()
    ranker = Ranker(config(**changes)).fit(data, range(16))
    rows = data.rows(range(16, 24))
    found = ranker.scores(data, rows)
    judged = ~np.isnan(data.grade[rows])
    assert np.corrcoef(found[judged], data.grade[rows][judged])[0, 1] > 0.6
    gain = ranker.gain_importance()
    assert gain["g1_signal"] > gain["g2_noise"] and sum(gain.values()) == pytest.approx(1)
    again = Ranker(config(**changes)).fit(data, range(16)).scores(data, rows)
    assert np.array_equal(found, again) and len(ranker.trees) == 2


def test_out_of_fold_never_trains_on_the_query_it_ranks(monkeypatch):
    data = make_data(6, 5)
    seen = []

    class Fake:
        trees = (1,)

        def __init__(self, config):
            pass

        def fit(self, data, query_ids):
            seen.append(set(query_ids))
            return self

        def scores(self, data, rows):
            assert not seen[-1] & set(data.query_id[rows])
            return np.zeros(len(rows))

        def gain_importance(self):
            return {}

    monkeypatch.setattr("shelfrank.ltr.cross_validation.Ranker", Fake)
    out = out_of_fold(data, [[0, 1], [2, 3], [4, 5]], config())
    assert set(out.run) == set(range(6)) and all(len(r) == 5 for r in out.run.values())
    assert seen == [{2, 3, 4, 5}, {0, 1, 4, 5}, {0, 1, 2, 3}]
    with pytest.raises(ValueError, match="overlap"):
        out_of_fold(data, [[0, 1], [1, 2]], config())


def test_every_candidate_is_ranked_including_unjudged_ones(monkeypatch):
    pytest.importorskip("lightgbm")
    data = make_data(12, 9)
    out = out_of_fold(data, [list(range(6)), list(range(6, 12))], config())
    assert all(sorted(r) == list(range(9)) for r in out.run.values())


def test_near_duplicates_share_an_inner_fold():
    texts = {i: f"unique{chr(97 + i)} thing{chr(97 + i)}" for i in range(18)}
    texts |= {18: "red lamp", 19: "lamp red"}
    folds = grouped_folds(texts, 4, 1, 0.75)
    assert sorted(len(f) for f in folds) == [5, 5, 5, 5]
    assert sorted(q for f in folds for q in f) == sorted(texts)
    assert any({18, 19} <= set(f) for f in folds)
    assert folds == grouped_folds(texts, 4, 1, 0.75)


def test_cv_stopping_uses_one_tree_count_for_every_seed_and_rejects_unknown_rules():
    pytest.importorskip("lightgbm")
    data = make_data()
    trees = Ranker(config(stopping=CV)).fit(data, range(16)).trees
    assert len(set(trees)) == 1 and 1 <= trees[0] <= 30
    with pytest.raises(ValueError, match="stopping"):
        Ranker(config(stopping="x")).fit(data, range(16))


def test_learning_curve_trains_on_exactly_that_many_queries(monkeypatch):
    data = make_data(12, 3)
    sizes = []

    class Fake:
        trees = ()

        def __init__(self, config):
            pass

        def fit(self, data, query_ids):
            sizes.append(len(query_ids))
            return self

        def scores(self, data, rows):
            return np.zeros(len(rows))

        def gain_importance(self):
            return {}

    monkeypatch.setattr("shelfrank.ltr.cross_validation.Ranker", Fake)
    out = out_of_fold(data, [list(range(6)), list(range(6, 12))], config(), train_limit=4)
    assert sizes == [4, 4] and len(out.run) == 12


def test_training_choices_cross_stopping_and_size():
    out = training_choices(config())
    assert out["stop_cv"].stopping == CV and out["stop_cv"].params == config().params
    assert out["stop_cv_smaller"].params["num_leaves"] == 7
    assert out["stop_holdout_smaller"].stopping != CV


COLUMNS = ["g1_a", "g2_b", "g3_c"]


def test_ablations_remove_and_add_whole_groups():
    out = ablations(config(), COLUMNS, ["G1", "G2", "G3"])
    assert out["without_G2"].columns == ("g1_a", "g3_c")
    assert out["only_G1"].columns == ("g1_a",)
    assert out["only_G1_G2"].columns == ("g1_a", "g2_b")
    assert len(out) == 5


def test_variants_and_alternatives_change_one_thing_each():
    assert {c.variant for c in variants(config()).values()} == {JUDGED_ONLY, UNJUDGED_ZERO}
    other = alternatives(config())
    assert other["pointwise"].objective == POINTWISE
    assert other["label_gain_013"].label_gain == (0, 1, 3)
    assert all(c.columns == config().columns for c in other.values())


def test_log_entry_describes_the_whole_configuration():
    body = json.loads(json.dumps(config().describe()))
    assert body["columns"] == ["g1_signal", "g2_noise"] and body["groups"] == ["G1", "G2"]
    assert set(body) >= {"variant", "objective", "label_gain", "params", "seeds", "valid_share"}


def round_a(root, disagreements):
    (root / "results").mkdir(exist_ok=True)
    body = {"room_rule": {"cases": 9, "disagreements": disagreements}}
    (root / "results/slots_round_a.json").write_text(json.dumps(body), encoding="utf-8")


def test_g4_is_refused_until_round_a_the_room_rule_and_a_student(tmp_path):
    with pytest.raises(ProtocolError, match="Round A has no summary"):
        allowed_groups(["G1", "G4"], tmp_path)
    round_a(tmp_path, 3)
    with pytest.raises(ProtocolError, match="3 disagreements"):
        allowed_groups(["G4"], tmp_path)
    round_a(tmp_path, 2)
    assert slot_blockers(tmp_path) == []
    with pytest.raises(ProtocolError, match="no slot student"):
        allowed_groups(["G4"], tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/slot_student.yaml").write_text("x: 1", encoding="utf-8")
    assert allowed_groups(["G4"], tmp_path) == ["G4"]


def test_other_groups_need_nothing(tmp_path):
    assert allowed_groups(["G1", "G6"], tmp_path) == ["G1", "G6"]


def test_committed_config_is_allowed_now():
    assert allowed_groups(load_config("ltr")["groups"])


@pytest.mark.parametrize("extra", [[], ["--pilot"]])
def test_slot_seed_batch_is_refused_before_any_client_exists(monkeypatch, extra):
    from scripts import label

    monkeypatch.setattr("sys.argv", ["label.py", "slots", "esci_seed", *extra])
    monkeypatch.setattr(label, "slot_blockers", lambda: ["Round A has no summary"])
    monkeypatch.setattr(label, "make_client", lambda **k: pytest.fail("a client was made"))
    with pytest.raises(SystemExit, match="must wait"):
        label.main()


def test_committed_dev_page_is_generated_from_the_committed_results():
    from shelfrank.config import REPO_ROOT
    from shelfrank.ltr.report import render

    results = json.loads((REPO_ROOT / "results/ltr_dev.json").read_text(encoding="utf-8"))
    page = (REPO_ROOT / "docs/ltr_dev.md").read_text(encoding="utf-8")
    assert page == render(results)
    for path in (REPO_ROOT / "results").glob("ltr_interim.json"):
        interim = json.loads(path.read_text(encoding="utf-8"))
        page = (REPO_ROOT / "docs/ltr_interim.md").read_text(encoding="utf-8")
        assert page == render(interim, "ltr_interim") and "INTERIM" in page
        assert "G4" not in interim["groups"]
    assert "G4" not in results["groups"] and results["split"] == "dev"


def test_every_reported_configuration_is_in_the_log():
    from shelfrank.config import REPO_ROOT

    results = json.loads((REPO_ROOT / "results/ltr_dev.json").read_text(encoding="utf-8"))
    lines = (REPO_ROOT / "results/ltr_dev_log.jsonl").read_text(encoding="utf-8").splitlines()
    logged = {json.loads(line)["name"] for line in lines}
    assert set(results["systems"]) - {"bm25_multi", "dense", "hybrid_rrf"} <= logged


def system(standard, condensed):
    return {"ndcg10": {"mean": standard}, "condensed_ndcg10": {"mean": condensed}}


def test_selection_is_the_mean_of_both_metrics_and_ties_go_to_judged_only():
    options = {
        "zero": LtrConfig(("g1_a",), UNJUDGED_ZERO),
        "judged": LtrConfig(("g1_a",), JUDGED_ONLY),
    }
    assert select({"zero": system(0.9, 0.5), "judged": system(0.6, 0.7)}, options) == "zero"
    assert select({"zero": system(0.5, 0.75), "judged": system(0.75, 0.5)}, options) == "judged"


def test_each_dev_stage_runs_once_unless_a_bug_fix_is_logged(tmp_path):
    (tmp_path / "results").mkdir()
    path = output_path("interim", ["G1"], "", tmp_path)
    path.write_text('{"bug_fixes": []}', encoding="utf-8")
    with pytest.raises(ProtocolError, match="bug-fix"):
        output_path("interim", ["G1"], "", tmp_path)
    assert output_path("interim", ["G1"], "wrong sign", tmp_path) == path
    assert bug_fixes(path, "wrong sign") == ["wrong sign"] and bug_fixes(path, "") == []


@pytest.mark.parametrize(("stage", "groups"), [("interim", ["G1", "G4"]), ("final", ["G1"])])
def test_g4_is_off_in_the_interim_run_and_on_in_the_final_one(tmp_path, stage, groups):
    with pytest.raises(ProtocolError, match="G4"):
        output_path(stage, groups, "", tmp_path)


def test_no_interim_run_after_the_final_one(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results/ltr_final.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolError, match="final"):
        output_path("interim", ["G1"], "x", tmp_path)


def test_selection_rule_and_encoder_are_recorded_before_the_freeze():
    from shelfrank.config import REPO_ROOT

    cfg = load_config("ltr")
    protocol = " ".join((REPO_ROOT / "docs/PROTOCOL.md").read_text(encoding="utf-8").split())
    assert "mean of standard and condensed nDCG@10 on the dev fold" in protocol
    assert "mean of standard and condensed nDCG@10" in cfg["selection_rule"]
    dense = json.loads((REPO_ROOT / "results/dense_dev.json").read_text(encoding="utf-8"))
    assert cfg["encoder"] == dense["encoder"]
