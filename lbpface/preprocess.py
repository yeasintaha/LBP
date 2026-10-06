"""FERET-style preprocessing (paper, Section 3, step 1).

    "The images are registered using eye coordinates and cropped with an elliptical
     mask to exclude non-face area from the image.  After this, the grey histogram
     over the non-masked area is equalised."

Only those three steps are described in the paper; the exact CSU settings (target eye
positions, ellipse size) are **not**.  The defaults below are therefore assumptions
chosen to give a sensible 130x150 face crop - every one of them is a parameter.  If you
have the CSU normalisation parameters, pass them in.
Coordinates are ``(x, y)`` = (column, row) in pixels, origin at the top-left pixel centre.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy import ndimage

Point = Tuple[float, float]

# Assumed defaults (NOT taken from the paper): 130 wide x 150 high output, eyes on a
# horizontal line 40 % down the image, 1/3 and 2/3 across.
OUT_SHAPE = (150, 130)  # (height, width)
TARGET_LEFT_EYE: Point = (130 / 3.0, 60.0)
TARGET_RIGHT_EYE: Point = (2 * 130 / 3.0, 60.0)


def similarity_from_eyes(left_eye: Point, right_eye: Point, target_left: Point,
                         target_right: Point) -> Tuple[complex, complex]:
    """Similarity transform (rotation + uniform scale + translation) mapping the eyes.
    Returns complex ``(a, b)`` such that ``z_out = a * z_in + b`` for ``z = x + i*y``,
    with ``a * left_eye + b == target_left`` and ``a * right_eye + b == target_right``.
    """
    e_l, e_r = complex(*left_eye), complex(*right_eye)
    t_l, t_r = complex(*target_left), complex(*target_right)
    if e_l == e_r:
        raise ValueError("eye coordinates coincide")
    a = (t_r - t_l) / (e_r - e_l)
    b = t_l - a * e_l
    return a, b


def register_by_eyes(image: np.ndarray, left_eye: Point, right_eye: Point,
                     out_shape: Tuple[int, int] = OUT_SHAPE,
                     target_left: Point = TARGET_LEFT_EYE, target_right: Point = TARGET_RIGHT_EYE,
                     order: int = 1) -> np.ndarray:
    """Rotate / scale / translate ``image`` so the eyes land on the target positions.
    ``left_eye`` / ``right_eye`` are the (x, y) eye centres in ``image`` (the subject's
    left/right eye appear in that order from the viewer's left to right - only the
    consistency between images matters).  Bilinear interpolation (``order=1``); pixels
    that fall outside the source image become 0.  Output is float64 of shape ``out_shape``.
    """
    a, b = similarity_from_eyes(left_eye, right_eye, target_left, target_right)
    c = 1.0 / a  # inverse map  z_in = c * (z_out - b)
    off = -c * b
    # scipy wants (row, col) = (y, x):  [y_in, x_in] = M @ [y_out, x_out] + offset
    m_xy = np.array([[c.real, -c.imag], [c.imag, c.real]])  # acts on (x, y)
    m_rc = m_xy[::-1, ::-1]
    offset = np.array([off.imag, off.real])
    return ndimage.affine_transform(np.asarray(image, dtype=np.float64), m_rc, offset=offset,
                                    output_shape=out_shape, order=order, mode="constant", cval=0.0)


def elliptical_mask(shape: Tuple[int, int] = OUT_SHAPE, center: Optional[Point] = None,
                    radii: Optional[Point] = None) -> np.ndarray:
    """Boolean mask, True inside the ellipse.  Defaults: the ellipse inscribed in the image.
    ``center = (x, y)`` and ``radii = (rx, ry)`` are in pixels.
    """
    h, w = shape
    cx, cy = center if center is not None else ((w - 1) / 2.0, (h - 1) / 2.0)
    rx, ry = radii if radii is not None else (w / 2.0, h / 2.0)
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def equalise_masked(image: np.ndarray, mask: np.ndarray, levels: int = 256) -> np.ndarray:
    """Histogram-equalise the grey values *inside the mask only*; pixels outside become 0.
    Classic CDF mapping on ``levels`` grey levels.  Returns uint8 when ``levels == 256``.
    """
    img = np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, levels - 1).astype(np.int64)
    inside = img[mask]
    if inside.size == 0:
        raise ValueError("mask is empty")
    hist = np.bincount(inside, minlength=levels)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    denom = inside.size - cdf_min
    if denom <= 0:  # constant image inside the mask
        lut = np.zeros(levels)
    else:
        lut = np.clip(np.rint((cdf - cdf_min) / denom * (levels - 1)), 0, levels - 1)
    out = np.where(mask, lut[img], 0)
    return out.astype(np.uint8 if levels <= 256 else np.int64)


def preprocess_face(image: np.ndarray, left_eye: Point, right_eye: Point,
                    out_shape: Tuple[int, int] = OUT_SHAPE,
                    target_left: Point = TARGET_LEFT_EYE, target_right: Point = TARGET_RIGHT_EYE,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Register by eyes -> elliptical mask -> histogram equalisation over the mask."""
    reg = register_by_eyes(image, left_eye, right_eye, out_shape, target_left, target_right)
    if mask is None:
        mask = elliptical_mask(out_shape)
    return equalise_masked(reg, mask)

