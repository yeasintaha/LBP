import numpy as np
import pytest

from lbpface.classify import cumulative_match_curve, nearest_neighbour, recognition_rate, true_ranks


def test_rank1_equals_argmin_accuracy_without_ties():
    rng = np.random.default_rng(0)
    dist = rng.random((50, 30))
    gal = np.arange(30)  # one image per subject
    prb = rng.integers(0, 30, size=50)
    acc = (nearest_neighbour(dist, gal) == prb).mean()
    assert recognition_rate(dist, gal, prb) == pytest.approx(acc)


def test_rank_and_cmc_hand_example():
    # 3 gallery subjects 10, 20, 30
    dist = np.array([[0.1, 0.5, 0.9],   # probe of 10 -> rank 1
                     [0.3, 0.2, 0.8],   # probe of 10 -> rank 2
                     [0.4, 0.5, 0.6]])  # probe of 30 -> rank 3
    gal = np.array([10, 20, 30])
    prb = np.array([10, 10, 30])
    np.testing.assert_array_equal(true_ranks(dist, gal, prb), [1, 2, 3])
    np.testing.assert_allclose(cumulative_match_curve(dist, gal, prb, 4), [1 / 3, 2 / 3, 1.0, 1.0])


def test_cmccd_is_monotone_and_reaches_one():
    rng = np.random.default_rng(1)
    dist = rng.random((40, 25))
    cmc = cumulative_match_curve(dist, np.arange(25), rng.integers(0, 25, 40), max_rank=25)
    assert np.all(np.diff(cmc) >= 0) and cmc[-1] == 1.0


def test_ties_never_help():
    dist = np.zeros((5, 4))  # degenerate: everything equally close
    gal, prb = np.arange(4), np.array([0, 1, 2, 3, 0])
    assert recognition_rate(dist, gal, prb) == 0.0


def test_multi_image_gallery_ranks_subjects_by_closest_image():
    # subject 1 has two gallery images, subject 2 has one
    dist = np.array([[0.9, 0.1, 0.5]])
    gal = np.array([1, 1, 2])
    assert recognition_rate(dist, gal, np.array([1])) == 1.0
    assert true_ranks(dist, gal, np.array([2]))[0] == 2


def test_probe_subject_missing_from_gallery_is_an_error():
    with pytest.raises(ValueError, match="missing"):
        recognition_rate(np.zeros((1, 2)), np.array([1, 2]), np.array([3]))
