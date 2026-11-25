import numpy as np
import pytest

from shelfrank.stats.bootstrap import Interval, bootstrap_mean, paired_difference


def test_same_seed_same_interval():
    values = np.random.default_rng(0).random(50)
    assert bootstrap_mean(values, 500, seed=3) == bootstrap_mean(values, 500, seed=3)


def test_constant_sample_has_no_width():
    out = bootstrap_mean([0.4] * 20, 200)
    assert out.low == out.high == pytest.approx(0.4)


def test_interval_contains_the_mean():
    out = bootstrap_mean(np.random.default_rng(1).random(80), 2000)
    assert out.low < out.mean < out.high and out.n == 80


def test_width_matches_the_normal_approximation():
    values = np.random.default_rng(2).normal(size=400)
    out = bootstrap_mean(values, 10_000)
    expected = 2 * 1.96 * values.std() / np.sqrt(400)
    assert out.high - out.low == pytest.approx(expected, rel=0.05)


def test_resample_count_need_not_be_a_multiple_of_the_chunk():
    assert np.isfinite(bootstrap_mean([0.0, 1.0], 1500).high)


@pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
def test_bad_samples_are_rejected(values):
    with pytest.raises(ValueError, match="at least one value"):
        bootstrap_mean(values)


def test_pairing_detects_a_small_consistent_gain():
    base = np.random.default_rng(4).random(100)
    out = paired_difference(base + 0.01, base, 2000)
    assert out.mean == pytest.approx(0.01) and out.excludes_zero()


def test_difference_is_antisymmetric():
    rng = np.random.default_rng(5)
    a, b = rng.random(60), rng.random(60)
    forward, backward = paired_difference(a, b, 1000), paired_difference(b, a, 1000)
    assert forward.mean == pytest.approx(-backward.mean)
    assert forward.low == pytest.approx(-backward.high, abs=0.01)


def test_unequal_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        paired_difference([1.0], [1.0, 2.0])


def test_no_difference_does_not_exclude_zero():
    assert not paired_difference([0.1, 0.2], [0.1, 0.2], 100).excludes_zero()


def test_an_interval_touching_zero_does_not_exclude_it():
    assert not Interval(0.1, 0.0, 0.2, 5).excludes_zero()
