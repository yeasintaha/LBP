import numpy as np
import pytest

from lbpface.stats import (draw_splits, permutation_rates, pairwise_outperform_table, prob_outperforms,
                           summarise)


def make_problem(n_subj=12, per=4, sep=3.0, seed=0):
    """Images = noisy copies of subject centres; distance = euclidean."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_subj, 5)) * sep
    labels = np.repeat(np.arange(n_subj), per)
    x = centres[labels] + rng.normal(size=(labels.size, 5))
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    return d, labels


def test_splits_are_reproducible_disjoint_and_cover_each_subject():
    labels = np.repeat(np.arange(6), 4)
    a = list(draw_splits(labels, 20, seed=7))
    b = list(draw_splits(labels, 20, seed=7))
    c = list(draw_splits(labels, 20, seed=8))
    for (g1, p1), (g2, p2) in zip(a, b):
        np.testing.assert_array_equal(g1, g2)
        np.testing.assert_array_equal(p1, p2)
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(a, c))
    for g, p in a:
        assert g.size == p.size == 6
        assert not set(g.tolist()) & set(p.tolist())
        np.testing.assert_array_equal(labels[g], np.arange(6))
        np.testing.assert_array_equal(labels[p], np.arange(6))


def test_split_stream_is_pinned():
    labels = np.repeat(np.arange(3), 4)
    g, p = next(draw_splits(labels, 1, seed=123))
    assert g.tolist() == [1, 4, 11]
    assert p.tolist() == [3, 7, 10]


def test_unequal_images_per_subject_is_an_error():
    with pytest.raises(ValueError):
        list(draw_splits(np.array([0, 0, 0, 1, 1]), 1, seed=0))


def test_multi_image_gallery_split():
    labels = np.repeat(np.arange(5), 10)
    (g, p), = list(draw_splits(labels, 1, 0, 5, 5))
    assert g.size == p.size == 25 and not set(g.tolist()) & set(p.tolist())


def test_rates_on_well_separated_data_are_high_and_deterministic():
    d, labels = make_problem(sep=6.0)
    r1 = permutation_rates({"a": d}, labels, 50, seed=1)["a"]
    r2 = permutation_rates({"a": d}, labels, 50, seed=1)["a"]
    np.testing.assert_array_equal(r1, r2)
    assert r1.mean() > 0.95


def test_better_algorithm_outperforms_worse_one():
    d_good, labels = make_problem(sep=4.0, seed=0)
    rng = np.random.default_rng(9)
    d_bad = d_good + rng.random(d_good.shape) * 40  # very noisy distances
    d_bad = 0.5 * (d_bad + d_bad.T)
    rates = permutation_rates({"good": d_good, "bad": d_bad}, labels, 200, seed=2)
    assert prob_outperforms(rates["good"], rates["bad"]) > 0.9
    assert prob_outperforms(rates["bad"], rates["good"]) < 0.05
    tbl = pairwise_outperform_table(rates)
    assert set(tbl) == {("good", "bad"), ("bad", "good")}


def test_same_permutations_for_every_algorithm():
    d, labels = make_problem()
    r = permutation_rates({"x": d, "y": d.copy()}, labels, 30, seed=4)
    np.testing.assert_array_equal(r["x"], r["y"])
    assert prob_outperforms(r["x"], r["y"]) == 0.0  # ties are not wins


def test_rate_matches_brute_force_on_one_split():
    d, labels = make_problem(sep=1.0, seed=3)  # hard problem -> mixed outcomes
    (g, p), = list(draw_splits(labels, 1, 5))
    correct = 0
    for i, pi in enumerate(p):
        nearest = g[np.argmin(d[pi, g])]
        correct += labels[nearest] == labels[pi]
    rate = permutation_rates({"a": d}, labels, 1, seed=5)["a"][0]
    assert rate == pytest.approx(correct / p.size)


def test_summarise():
    s = summarise(np.linspace(0.5, 1.0, 101))
    assert s["mean"] == pytest.approx(0.75)
    assert s["lower"] < s["mean"] < s["upper"]
    assert s["lower"] == pytest.approx(np.percentile(np.linspace(0.5, 1.0, 101), 2.5))
    assert s["n_permutations"] == 101


def test_shape_checks():
    d, labels = make_problem()
    with pytest.raises(ValueError):
        permutation_rates({"a": d[:-1, :-1]}, labels, 2, 0)
    with pytest.raises(ValueError):
        prob_outperforms(np.zeros(3), np.zeros(4))

