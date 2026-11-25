import random
from collections import Counter

import pytest

from shelfrank.eval.splits import RARE, Folds, assign_folds, load_folds, save_folds, strata


def make_classes(n=100, seed=0):
    rng = random.Random(seed)
    names = ["A"] * 30 + ["B"] * 20 + ["C"] * 3 + [None] * 2
    return {i: rng.choice(names) for i in range(n)}


def test_small_and_missing_classes_are_pooled():
    out = strata({0: "A", 1: "A", 2: "B", 3: None}, n_folds=2)
    assert out == {0: "A", 1: "A", 2: RARE, 3: RARE}


def test_every_query_is_assigned_once_and_sizes_are_equal():
    out = assign_folds(make_classes(), [], 5, seed=1)
    assert sorted(out) == list(range(100))
    assert set(Counter(out.values()).values()) == {20}


def test_uneven_totals_differ_by_at_most_the_rounding():
    sizes = Counter(assign_folds(make_classes(103), [], 5, seed=1).values()).values()
    assert max(sizes) <= 21 and sum(sizes) == 103


@pytest.mark.parametrize("seed", range(5))
def test_large_strata_are_balanced_within_one(seed):
    classes = {i: "A" if i < 33 else "B" for i in range(60)}
    out = assign_folds(classes, [], 5, seed=seed)
    for name in "AB":
        counts = Counter(out[q] for q, c in classes.items() if c == name)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_groups_stay_together():
    groups = [[0, 50], [1, 2, 3]]
    out = assign_folds(make_classes(), groups, 5, seed=3)
    assert all(len({out[q] for q in group}) == 1 for group in groups)
    assert set(Counter(out.values()).values()) == {20}


def test_same_seed_same_folds_regardless_of_input_order():
    classes = make_classes()
    shuffled = dict(reversed(classes.items()))
    assert assign_folds(classes, [[4, 9]], 5, 11) == assign_folds(shuffled, [[9, 4]], 5, 11)


def test_different_seed_changes_the_assignment():
    assert assign_folds(make_classes(), [], 5, 1) != assign_folds(make_classes(), [], 5, 2)


@pytest.mark.parametrize("groups", [[[0, 1], [1, 2]], [[0, 999]]])
def test_invalid_groups_are_rejected(groups):
    with pytest.raises(ValueError, match="disjoint"):
        assign_folds(make_classes(), groups, 5, 0)


def test_oversized_group_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        assign_folds(make_classes(10), [[0, 1, 2, 3]], 5, 0)


@pytest.mark.parametrize("n_folds", [0, 1, 11])
def test_invalid_fold_count_is_rejected(n_folds):
    with pytest.raises(ValueError, match="n_folds"):
        assign_folds(make_classes(10), [], n_folds, 0)


def test_dev_and_primary_partition_the_queries():
    folds = Folds({1: 0, 2: 1, 3: 2, 4: 0}, n_folds=3, dev_fold=0)
    assert folds.dev == [1, 4] and folds.primary == [2, 3]


def test_round_trip(tmp_path):
    folds = Folds({3: 1, 1: 0}, n_folds=2, dev_fold=1)
    save_folds(folds, tmp_path / "x" / "f.json", {"seed": 5})
    assert load_folds(tmp_path / "x" / "f.json") == folds


@pytest.mark.parametrize(
    "folds", [Folds({1: 0, 2: 0}, 2, 0), Folds({1: 0, 2: 5}, 2, 0), Folds({1: 0, 2: 1}, 2, 7)]
)
def test_corrupt_file_is_rejected(tmp_path, folds):
    save_folds(folds, tmp_path / "f.json", {})
    with pytest.raises(ValueError, match="not a valid"):
        load_folds(tmp_path / "f.json")
