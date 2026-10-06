import numpy as np
import pytest

from lbpface.weights import symmetrise, weights_from_rates, window_recognition_rates


def test_symmetrise_averages_mirror_windows():
    r = np.arange(12, dtype=float).reshape(3, 4)
    s = symmetrise(r)
    np.testing.assert_allclose(s, s[:, ::-1])
    assert s[0, 0] == pytest.approx((r[0, 0] + r[0, 3]) / 2)


def test_percentile_rule_on_a_7x7_map():
    # 49 distinct, left/right symmetric rates: 1..25 mirrored so that mirror pairs are equal
    rng = np.random.default_rng(0)
    half = rng.permutation(np.linspace(0.05, 0.95, 28))  # 7 rows x 4 columns incl. centre column
    rates = np.zeros((7, 7))
    rates[:, :4] = half.reshape(7, 4)
    rates[:, 4:] = rates[:, :3][:, ::-1]
    w = weights_from_rates(rates)
    assert set(np.unique(w)) <= {0.0, 1.0, 2.0, 4.0}
    lo, hi2, hi4 = np.percentile(rates, [20, 80, 90])
    assert np.all(w[rates < lo] == 0) and np.all(w[rates > hi4] == 4)
    assert np.all(w[(rates > hi2) & (rates <= hi4)] == 2)
    assert np.all(w[(rates >= lo) & (rates <= hi2)] == 1)
    np.testing.assert_array_equal(w, w[:, ::-1])  # weights are left/right symmetric
    # the paper's final vector has 39 of 49 windows (10 zero weights); here: about a fifth
    assert 8 <= (w == 0).sum() <= 12
    assert 4 <= (w == 4).sum() <= 8


def test_uniform_rates_give_weight_one_everywhere():
    assert np.all(weights_from_rates(np.full((7, 7), 0.5)) == 1.0)


def test_weights_do_not_depend_on_left_right_flip_of_input():
    rng = np.random.default_rng(3)
    r = rng.random((7, 7))
    np.testing.assert_array_equal(weights_from_rates(r), weights_from_rates(r[:, ::-1]))


def test_window_rates_find_the_informative_window():
    rng = np.random.default_rng(0)
    n_bins, n_reg, n_subj = 6, 3, 15
    ids = np.arange(n_subj)
    proto = rng.integers(5, 20, size=(n_subj, n_bins)).astype(float)

    def make(noise):
        informative = proto + rng.integers(-noise, noise + 1, size=proto.shape)
        junk = [rng.integers(0, 30, size=proto.shape).astype(float) for _ in range(n_reg - 1)]
        return np.concatenate([np.clip(informative, 0, None)] + junk, axis=1)

    gal, prb = make(1), make(1)
    rates = window_recognition_rates(gal, ids, prb, ids, n_reg, n_bins)
    assert rates.shape == (3,)
    assert rates[0] > 0.9 and rates[1] < 0.5 and rates[2] < 0.5
