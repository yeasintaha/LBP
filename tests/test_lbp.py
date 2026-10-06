"""LBP operators against the paper's own examples and an independent implementation."""
import numpy as np
import pytest

from lbpface.lbp import (circular_transitions, is_uniform, lbp_3x3, lbp_codes, lbp_u2, n_labels,
                         n_uniform_patterns, sampling_offsets, uniform_mapping)


def bits(s):
    return int(s, 2)


def test_fig1_basic_operator_gives_211():
    patch = np.array([[5, 9, 1], [4, 4, 6], [7, 2, 3]])
    out = lbp_3x3(patch)
    assert out.shape == (1, 1)
    assert out[0, 0] == 0b11010011 == 211


def test_basic_operator_is_invariant_to_any_monotone_grey_transform():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(20, 25))
    for f in (lambda x: x ** 3 + 7, lambda x: np.log1p(x), lambda x: 2 * x + 5):
        np.testing.assert_array_equal(lbp_3x3(img), lbp_3x3(f(img.astype(float))))


@pytest.mark.parametrize("s", ["00000000", "00011110", "10000011", "11111111"])
def test_paper_examples_of_uniform_patterns(s):
    assert is_uniform(np.array([bits(s)]), 8)[0]


@pytest.mark.parametrize("s,t", [("01010101", 8), ("00010100", 4), ("10100000", 4), ("00000001", 2)])
def test_transition_counts(s, t):
    assert circular_transitions(np.array([bits(s)]), 8)[0] == t


@pytest.mark.parametrize("P,uniform,labels", [(8, 58, 59), (16, 242, 243), (4, 14, 15)])
def test_label_counts_match_paper(P, uniform, labels):
    assert n_uniform_patterns(P) == uniform
    assert n_labels(P) == labels
    table = uniform_mapping(P)
    assert table.size == 2 ** P
    assert table.max() == labels - 1
    assert len(np.unique(table)) == labels  # every label is used
    # each uniform pattern has its own label; all non-uniform ones share the last
    codes = np.arange(2 ** P)
    uni = is_uniform(codes, P)
    assert len(np.unique(table[uni])) == uniform
    assert np.all(table[~uni] == labels - 1)


def test_uniform_labels_are_ascending_in_code():
    table = uniform_mapping(8)
    codes = np.arange(256)
    uni_codes = codes[is_uniform(codes, 8)]
    np.testing.assert_array_equal(table[uni_codes], np.arange(58))


def test_mapping_is_read_only():
    with pytest.raises(ValueError):
        uniform_mapping(8)[0] = 3


def test_sampling_points_lie_on_the_circle_and_cover_it():
    for P, R in [(8, 1.0), (8, 2.0), (16, 2.0), (12, 3.5)]:
        dy, dx = sampling_offsets(P, R)
        np.testing.assert_allclose(np.hypot(dy, dx), R, atol=1e-4)
        assert len({(round(a, 4), round(b, 4)) for a, b in zip(dy, dx)}) == P
    dy, dx = sampling_offsets(8, 2.0)
    assert (dy[0], dx[0]) == (0.0, 2.0)  # bit 0 is "east"
    assert dy[2] == -2.0 and dx[2] == 0.0  # bit 2 is "north" (row index decreases)


@pytest.mark.parametrize("P,R", [(8, 1.0), (8, 2.0), (16, 2.0), (4, 1.0), (12, 3.0)])
def test_raw_codes_equal_scikit_image(P, R):
    skf = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(0)
    img = rng.random((45, 38))
    mine = lbp_codes(img, P, R, border="constant")
    ref = skf.local_binary_pattern(img, P, R, method="default")
    np.testing.assert_array_equal(mine, ref.astype(np.int64))


@pytest.mark.parametrize("P,R", [(8, 1.0), (8, 2.0), (16, 2.0)])
def test_u2_labels_partition_pixels_like_scikit_image_nri_uniform(P, R):
    skf = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(3)
    img = rng.random((60, 60))
    mine = lbp_u2(img, P, R, border="constant")
    ref = skf.local_binary_pattern(img, P, R, method="nri_uniform").astype(np.int64)
    pairs = set(zip(mine.ravel().tolist(), ref.ravel().tolist()))
    # one-to-one correspondence <=> the two labellings are the same partition
    assert len(pairs) == len(set(mine.ravel().tolist())) == len(set(ref.ravel().tolist()))


def test_constant_image_gives_all_ones_pattern():
    img = np.full((10, 12), 77.0)
    codes = lbp_codes(img, 8, 2.0)
    assert np.all(codes == 255)  # g_p >= g_c everywhere
    assert np.all(lbp_u2(img, 8, 2.0) == uniform_mapping(8)[255])


def test_circular_operator_is_invariant_to_affine_grey_changes():
    # bilinear interpolation is affine, so a*x+b (a>0) cannot change any comparison
    rng = np.random.default_rng(5)
    img = rng.random((30, 30))
    np.testing.assert_array_equal(lbp_codes(img, 8, 2.0), lbp_codes(3.7 * img + 11.0, 8, 2.0))


def test_output_shape_and_border_modes():
    img = np.random.default_rng(0).random((17, 23))
    for border in ("edge", "reflect", "constant"):
        assert lbp_u2(img, 16, 2.0, border).shape == img.shape
    with pytest.raises(ValueError):
        lbp_u2(img, 8, 1.0, "wrap")


def test_interpolation_at_half_pixel():
    # R=sqrt(2) puts the diagonal sample exactly on the corner neighbour only for P=4 at 45deg;
    # check bilinear weights on a simple ramp instead: value at (y, x+0.5) is the mean of two pixels
    img = np.tile(np.arange(10.0), (10, 1))  # img[y, x] = x
    # P=4, R=0.5 -> samples at x+0.5, y-0.5, x-0.5, y+0.5 ; ramp: x+0.5 > x, x-0.5 < x
    codes = lbp_codes(img, 4, 0.5)
    inner = codes[2:-2, 2:-2]
    # bit0 (east): x+.5 >= x -> 1 ; bit1 (north): x >= x -> 1 ; bit2 (west): x-.5 >= x -> 0 ; bit3 (south): 1
    assert np.all(inner == 0b1011)
