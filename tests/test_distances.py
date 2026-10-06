"""Dissimilarity measures (Eqs. 3-6)."""
import numpy as np
import pytest

from lbpface.distances import (chi_square, dissimilarity, expand_region_weights, histogram_intersection,
                               log_likelihood, pairwise_distances)


def naive_chi2(s, m, w=None):
    tot = 0.0
    for i in range(len(s)):
        if s[i] + m[i] > 0:
            tot += (1.0 if w is None else w[i]) * (s[i] - m[i]) ** 2 / (s[i] + m[i])
    return tot


@pytest.fixture
def hists():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 6, size=(7, 4 * 5)).astype(float)  # 4 regions x 5 bins
    p = rng.integers(0, 6, size=(3, 4 * 5)).astype(float)
    g[:, 3] = 0.0
    p[:, 3] = 0.0  # a bin that is empty everywhere -> 0/0
    return p, g


def test_chi_square_matches_eq5_and_handles_empty_bins(hists):
    p, g = hists
    for i in range(p.shape[0]):
        for j in range(g.shape[0]):
            assert chi_square(p[i], g[j]) == pytest.approx(naive_chi2(p[i], g[j]))
    assert np.isfinite(chi_square(p[0], g[0]))


def test_chi_square_basic_properties(hists):
    p, g = hists
    assert chi_square(p[0], p[0]) == 0.0
    assert chi_square(p[0], g[0]) == pytest.approx(chi_square(g[0], p[0]))  # symmetric
    assert chi_square(p[0], g[0]) >= 0.0
    # hand computed: S=[1,0], M=[0,1] -> 1/1 + 1/1
    assert chi_square(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 2.0


def test_weighted_chi_square_matches_eq6(hists):
    p, g = hists
    rw = np.array([0.0, 1.0, 4.0, 2.0])
    w = expand_region_weights(rw, 5)
    for i in range(p.shape[0]):
        for j in range(g.shape[0]):
            assert chi_square(p[i], g[j], w) == pytest.approx(naive_chi2(p[i], g[j], w))


def test_intersection_and_log_likelihood_match_eq3_eq4():
    s = np.array([3.0, 1.0, 0.0, 2.0])
    m = np.array([1.0, 1.0, 4.0, 0.0])
    assert histogram_intersection(s, m) == 2.0  # min: 1 + 1 + 0 + 0
    eps = 1e-10
    expected = -(3 * np.log(1 + eps) + 1 * np.log(1 + eps) + 0 * np.log(4 + eps) + 2 * np.log(0 + eps))
    assert log_likelihood(s, m) == pytest.approx(expected)
    # log-likelihood is asymmetric and explodes when the model has an empty bin the sample uses
    assert log_likelihood(s, m) > 40


def test_dissimilarity_orientation_smaller_is_more_similar():
    a = np.array([5.0, 0.0, 1.0])
    near = np.array([4.0, 1.0, 1.0])
    far = np.array([0.0, 6.0, 0.0])
    for metric in ("chi2", "intersection", "log_likelihood"):
        assert dissimilarity(a, near, metric) < dissimilarity(a, far, metric), metric
    with pytest.raises(ValueError):
        dissimilarity(a, near, "cosine")


@pytest.mark.parametrize("metric", ["chi2", "intersection", "log_likelihood"])
def test_pairwise_matches_single_pair_and_is_block_size_independent(hists, metric):
    p, g = hists
    d = pairwise_distances(p, g, metric)
    assert d.shape == (3, 7)
    for i in range(3):
        for j in range(7):
            assert d[i, j] == pytest.approx(dissimilarity(p[i], g[j], metric), rel=1e-12)
    tiny = pairwise_distances(p, g, metric, max_block_elems=1)  # one probe per block
    np.testing.assert_array_equal(d, tiny)


@pytest.mark.parametrize("metric", ["chi2", "intersection", "log_likelihood"])
def test_zero_weight_regions_are_dropped_and_equal_to_weighting(hists, metric):
    p, g = hists
    rw = np.array([0.0, 1.0, 4.0, 2.0])
    d = pairwise_distances(p, g, metric, rw, 5)
    w = expand_region_weights(rw, 5)
    for i in range(3):
        for j in range(7):
            assert d[i, j] == pytest.approx(dissimilarity(p[i], g[j], metric, w), rel=1e-12)


def test_weight_validation(hists):
    p, g = hists
    with pytest.raises(ValueError):
        pairwise_distances(p, g, "chi2", np.ones(3), 5)  # wrong number of regions
    with pytest.raises(ValueError):
        pairwise_distances(p, g, "chi2", np.array([1, 1, -1, 1.0]), 5)
    with pytest.raises(ValueError):
        pairwise_distances(p, g, "chi2", np.ones(4))  # n_bins missing
