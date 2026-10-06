"""Spatial histograms: geometry and feature lengths quoted in the paper."""
import numpy as np
import pytest

from lbpface.features import LBPDescriptor, k_by_k_window, spatial_histogram, window_grid
from lbpface.lbp import n_labels

FERET_SHAPE = (150, 130)  # (h, w) of the preprocessed FERET images
ORL_SHAPE = (112, 92)


def test_paper_window_sizes_for_k_by_k_division():
    # x-axis of Fig. 4: 8x9 10x11 11x13 13x15 14x16 16x18 18x21 21x25 26x30 32x37 (+43x50 in Table 1)
    ks = [16, 13, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    sizes = [k_by_k_window(FERET_SHAPE, k) for k in ks]
    assert sizes == [(8, 9), (10, 11), (11, 13), (13, 15), (14, 16), (16, 18), (18, 21), (21, 25),
                     (26, 30), (32, 37), (43, 50)]
    # and the k x k division is recovered from the window size
    for k in ks:
        rows, cols, _, _ = window_grid(FERET_SHAPE, window=k_by_k_window(FERET_SHAPE, k))
        assert (rows, cols) == (k, k)


def test_feature_lengths_quoted_in_the_paper():
    assert LBPDescriptor(P=16, R=2, window=(18, 21)).feature_length(FERET_SHAPE) == 11907
    assert LBPDescriptor(P=8, R=2, window=(21, 25)).feature_length(FERET_SHAPE) == 2124
    feret = LBPDescriptor.feret()
    assert feret.layout(FERET_SHAPE)[:2] == (7, 7)
    assert feret.n_bins == 59
    # 49 windows minus the 10 that get weight 0 -> 39 windows * 59 bins = 2301 (Conclusion)
    assert (49 - 10) * feret.n_bins == 2301


def test_orl_window_grid():
    assert LBPDescriptor.orl().layout(ORL_SHAPE)[:2] == (3, 3)
    assert LBPDescriptor.orl().n_bins == 243
    assert LBPDescriptor.orl().feature_length(ORL_SHAPE) == 9 * 243


def test_exactly_one_of_grid_or_window():
    with pytest.raises(ValueError):
        LBPDescriptor(P=8, R=2)
    with pytest.raises(ValueError):
        LBPDescriptor(P=8, R=2, grid=(7, 7), window=(18, 21))
    with pytest.raises(ValueError):
        window_grid((10, 10), window=(20, 20))


def _naive_hist(labels, n_bins, rows, cols, win_h, win_w, y0, x0):
    out = np.zeros(rows * cols * n_bins)
    for r in range(rows):
        for c in range(cols):
            block = labels[y0 + r * win_h : y0 + (r + 1) * win_h, x0 + c * win_w : x0 + (c + 1) * win_w]
            for v in block.ravel():
                out[(r * cols + c) * n_bins + v] += 1
    return out


@pytest.mark.parametrize("anchor", ["center", "topleft"])
def test_spatial_histogram_matches_naive_loops(anchor):
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 59, size=(150, 130))
    h = spatial_histogram(lab, 59, window=(18, 21), anchor=anchor)
    y0, x0 = ((150 - 7 * 21) // 2, (130 - 7 * 18) // 2) if anchor == "center" else (0, 0)
    np.testing.assert_array_equal(h, _naive_hist(lab, 59, 7, 7, 21, 18, y0, x0))
    assert h.sum() == 7 * 7 * 21 * 18  # every covered pixel counted once (Eq. 2)


def test_spatial_histogram_rejects_bad_labels():
    with pytest.raises(ValueError):
        spatial_histogram(np.full((10, 10), 5), 5, grid=(2, 2))


def test_descriptor_is_deterministic_and_window_sums_are_equal():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=FERET_SHAPE).astype(np.uint8)
    d = LBPDescriptor.feret()
    a, b = d.transform_one(img), d.transform_one(img)
    np.testing.assert_array_equal(a, b)
    per_window = a.reshape(49, 59).sum(axis=1)
    assert np.all(per_window == 18 * 21)


def test_descriptor_depends_on_image_content():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, size=FERET_SHAPE).astype(np.uint8)
    b = rng.integers(0, 256, size=FERET_SHAPE).astype(np.uint8)
    d = LBPDescriptor.feret()
    assert not np.array_equal(d.transform_one(a), d.transform_one(b))
