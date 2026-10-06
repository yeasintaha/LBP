import numpy as np
import pytest

from lbpface.preprocess import (TARGET_LEFT_EYE, TARGET_RIGHT_EYE, elliptical_mask, equalise_masked,
                                preprocess_face, register_by_eyes, similarity_from_eyes)


def blob_image(shape, centres, sigma=2.5):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    img = np.zeros(shape)
    for cx, cy in centres:
        img += 200 * np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))
    return img


def centroid(img, x_range):
    lo, hi = x_range
    sub = img[:, lo:hi]
    yy, xx = np.mgrid[0 : img.shape[0], lo:hi]
    return (sub * xx).sum() / sub.sum(), (sub * yy).sum() / sub.sum()


def test_similarity_maps_eyes_exactly():
    a, b = similarity_from_eyes((100, 120), (160, 110), TARGET_LEFT_EYE, TARGET_RIGHT_EYE)
    assert a * complex(100, 120) + b == pytest.approx(complex(*TARGET_LEFT_EYE))
    assert a * complex(160, 110) + b == pytest.approx(complex(*TARGET_RIGHT_EYE))
    with pytest.raises(ValueError):
        similarity_from_eyes((1, 1), (1, 1), (0, 0), (1, 0))


@pytest.mark.parametrize("le,re", [((80.0, 100.0), (130.0, 100.0)),       # scale + shift
                                   ((90.0, 80.0), (140.0, 110.0)),        # rotation + scale
                                   ((50.5, 70.25), (92.0, 60.0))])        # sub-pixel, shrink
def test_registration_puts_eyes_on_the_target_positions(le, re):
    raw = blob_image((220, 240), [le, re])
    out = register_by_eyes(raw, le, re)
    assert out.shape == (150, 130)
    x_mid = int(round((TARGET_LEFT_EYE[0] + TARGET_RIGHT_EYE[0]) / 2))
    cl = centroid(out, (0, x_mid))
    cr = centroid(out, (x_mid, 130))
    assert cl == pytest.approx(TARGET_LEFT_EYE, abs=0.5)
    assert cr == pytest.approx(TARGET_RIGHT_EYE, abs=0.5)


def test_registration_is_identity_when_eyes_are_already_in_place():
    rng = np.random.default_rng(0)
    img = rng.random((150, 130)) * 255
    out = register_by_eyes(img, TARGET_LEFT_EYE, TARGET_RIGHT_EYE)
    np.testing.assert_allclose(out[5:-5, 5:-5], img[5:-5, 5:-5], atol=1e-6)


def test_elliptical_mask():
    m = elliptical_mask((150, 130))
    assert m.shape == (150, 130) and m.dtype == bool
    assert m[75, 65] and not m[0, 0] and not m[0, 129] and not m[149, 0]
    assert m.mean() == pytest.approx(np.pi / 4, abs=0.02)
    np.testing.assert_array_equal(m, m[:, ::-1])  # left/right symmetric
    m2 = elliptical_mask((50, 50), center=(10, 10), radii=(5, 5))
    assert m2[10, 10] and not m2[10, 20]


def test_equalisation_flattens_histogram_inside_mask_only():
    rng = np.random.default_rng(1)
    mask = elliptical_mask((150, 130))
    img = np.clip(rng.normal(90, 12, size=(150, 130)), 0, 255)  # narrow grey range
    out = equalise_masked(img, mask)
    assert out.dtype == np.uint8
    assert np.all(out[~mask] == 0)
    inside = out[mask].astype(float)
    assert inside.min() <= 2 and inside.max() == 255
    # after equalisation the cdf is close to linear
    hist = np.bincount(out[mask], minlength=256).cumsum() / mask.sum()
    assert np.max(np.abs(hist - np.linspace(0, 1, 256))) < 0.03
    assert inside.std() > 60  # was ~12 before


def test_equalisation_of_constant_image_does_not_crash():
    mask = elliptical_mask((20, 20))
    out = equalise_masked(np.full((20, 20), 50.0), mask)
    assert np.all(out == 0)
    with pytest.raises(ValueError):
        equalise_masked(np.zeros((4, 4)), np.zeros((4, 4), bool))


def test_preprocess_face_shape_dtype_and_mask():
    raw = blob_image((220, 240), [(100, 100), (150, 100)]) + 30
    out = preprocess_face(raw, (100, 100), (150, 100))
    assert out.shape == (150, 130) and out.dtype == np.uint8
    assert np.all(out[~elliptical_mask((150, 130))] == 0)
