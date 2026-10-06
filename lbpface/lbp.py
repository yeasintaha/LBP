"""Local Binary Pattern operators (paper, Section 2).

Three things live here:

* ``lbp_3x3``      - the original 3x3 operator of Fig. 1.
* ``lbp_codes``    - the circular (P, R) operator with bilinear interpolation (Fig. 2).
* ``lbp_u2``       - LBP^{u2}_{P,R}: uniform patterns keep their own label, every
                     non-uniform pattern shares one extra label.

* Sampling point ``p`` (``p = 0 .. P-1``) sits at angle ``2*pi*p/P``, measured
  counter-clockwise from "east" (``dy = -R sin(a)``, ``dx = R cos(a)`` because image
  rows grow downwards).  Bit ``p`` of the code is ``s(g_p - g_c)`` with
  ``s(x) = 1 if x >= 0 else 0``.  This is the same convention as
  ``skimage.feature.local_binary_pattern``; the test-suite uses that function as an
  independent reference.

* Uniform patterns are labelled ``0 .. P(P-1)+1`` in ascending order of their integer
  code; all non-uniform patterns get the last label ``P(P-1)+2``.  That gives the
  59 labels (P=8) and 243 labels (P=16) quoted in Section 3 of the paper.  Any
  relabelling is a permutation of histogram bins, which leaves every distance
  measure unchanged.

* The label image has the same shape as the input. Sampling points that fall outside
  the image read from a padded copy (``border="edge"`` replicates the outermost
  pixel). Keeping the full image size matters: the paper's 92x112 ORL images with
  30x37 windows only give the 3x3 window grid if the border pixels are kept.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

# Largest P for which a full 2**P lookup table is built.
MAX_P = 20


# --------------------------------------------------------------------------- #
# original 3x3 operator (Fig. 1)
# --------------------------------------------------------------------------- #
def lbp_3x3(image: np.ndarray) -> np.ndarray:
    """The basic LBP operator of Fig. 1.
    Each interior pixel is compared with its 8 neighbours, read *clockwise starting
    at the top-left corner*, and the resulting bits are read as a binary number with
    the first neighbour as the most significant bit.  For the example in Fig. 1
    (neighbourhood ``[[5,9,1],[4,4,6],[7,2,3]]``) the answer is ``11010011 = 211``.
    Returns an array of shape ``(H-2, W-2)``.
    """
    img = np.asarray(image)
    if img.ndim != 2 or min(img.shape) < 3:
        raise ValueError("image must be 2-D and at least 3x3")
    h, w = img.shape
    centre = img[1:-1, 1:-1]
    # (row offset, col offset) of the 8 neighbours, clockwise from top-left
    ring = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    out = np.zeros(centre.shape, dtype=np.int64)
    for i, (dy, dx) in enumerate(ring):
        nb = img[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        out |= (nb >= centre).astype(np.int64) << (7 - i)
    return out


# --------------------------------------------------------------------------- #
# circular (P, R) operator with bilinear interpolation (Fig. 2)
# --------------------------------------------------------------------------- #
def sampling_offsets(P: int, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(dy, dx)`` offsets of the P sampling points of the (P, R) neighbourhood.
    Offsets are rounded to 5 decimals so that e.g. ``R*sin(pi)`` is exactly 0.
    """
    angles = 2.0 * np.pi * np.arange(P, dtype=np.float64) / P
    dy = np.round(-R * np.sin(angles), 5)
    dx = np.round(R * np.cos(angles), 5)
    return dy, dx


def _pad(image: np.ndarray, width: int, border: str) -> np.ndarray:
    if border == "edge":
        return np.pad(image, width, mode="edge")
    if border == "reflect":
        return np.pad(image, width, mode="reflect")
    if border == "constant":
        return np.pad(image, width, mode="constant", constant_values=0.0)
    raise ValueError(f"unknown border mode {border!r} (use 'edge', 'reflect' or 'constant')")


def lbp_codes(image: np.ndarray, P: int = 8, R: float = 1.0, border: str = "edge") -> np.ndarray:
    """Raw LBP_{P,R} code of every pixel (an integer in ``[0, 2**P)``).
    Grey values at non-integer sampling positions are obtained by bilinear
    interpolation.  The output has the same shape as ``image``.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D")
    if not (1 <= P <= MAX_P):
        raise ValueError(f"P must be in [1, {MAX_P}]")
    if R <= 0:
        raise ValueError("R must be positive")

    h, w = img.shape
    dy, dx = sampling_offsets(P, R)
    pad = int(np.ceil(R)) + 1
    padded = _pad(img, pad, border)
    centre = padded[pad : pad + h, pad : pad + w]

    codes = np.zeros((h, w), dtype=np.int64)
    for p in range(P):
        y0 = int(np.floor(dy[p]))
        x0 = int(np.floor(dx[p]))
        ty = dy[p] - y0  # fractional parts: the same for every pixel
        tx = dx[p] - x0
        r0, c0 = pad + y0, pad + x0
        a = padded[r0 : r0 + h, c0 : c0 + w]
        if tx == 0.0 and ty == 0.0:
            g = a
        else:
            b = padded[r0 : r0 + h, c0 + 1 : c0 + 1 + w]
            c = padded[r0 + 1 : r0 + 1 + h, c0 : c0 + w]
            d = padded[r0 + 1 : r0 + 1 + h, c0 + 1 : c0 + 1 + w]
            top = (1.0 - tx) * a + tx * b
            bottom = (1.0 - tx) * c + tx * d
            g = (1.0 - ty) * top + ty * bottom
        codes |= (g >= centre).astype(np.int64) << p
    return codes


# --------------------------------------------------------------------------- #
# uniform patterns
# --------------------------------------------------------------------------- #
def n_uniform_patterns(P: int) -> int:
    """Number of uniform P-bit patterns: ``P(P-1)+2``."""
    return P * (P - 1) + 2


def n_labels(P: int) -> int:
    """Length of an LBP^{u2}_{P,R} histogram: ``P(P-1)+3`` (59 for P=8, 243 for P=16)."""
    return n_uniform_patterns(P) + 1


def _popcount(x: np.ndarray, bits: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(bits):
        out += (x >> i) & 1
    return out


def circular_transitions(codes: np.ndarray, P: int) -> np.ndarray:
    """Number of 0->1 / 1->0 transitions of the circular P-bit string(s) ``codes``."""
    codes = np.asarray(codes, dtype=np.int64)
    mask = (1 << P) - 1
    rotated = ((codes << 1) | (codes >> (P - 1))) & mask
    return _popcount(codes ^ rotated, P)


def is_uniform(codes: np.ndarray, P: int) -> np.ndarray:
    """True where the circular pattern has at most two bitwise transitions."""
    return circular_transitions(codes, P) <= 2


@lru_cache(maxsize=None)
def uniform_mapping(P: int) -> np.ndarray:
    """Lookup table of length ``2**P`` mapping raw codes to u2 labels (see module docs)."""
    if not (2 <= P <= MAX_P):
        raise ValueError(f"P must be in [2, {MAX_P}]")
    codes = np.arange(1 << P, dtype=np.int64)
    uniform = is_uniform(codes, P)
    n_uni = int(uniform.sum())
    if n_uni != n_uniform_patterns(P):  # pragma: no cover - sanity check of the maths
        raise AssertionError(f"expected {n_uniform_patterns(P)} uniform patterns, found {n_uni}")
    table = np.full(1 << P, n_uni, dtype=np.int64)  # non-uniform -> last label
    table[uniform] = np.arange(n_uni, dtype=np.int64)  # ascending code order
    table.flags.writeable = False  # shared through the cache
    return table


def lbp_u2(image: np.ndarray, P: int = 8, R: float = 1.0, border: str = "edge") -> np.ndarray:
    """LBP^{u2}_{P,R} label image (labels in ``[0, n_labels(P))``), same shape as ``image``."""
    return uniform_mapping(P)[lbp_codes(image, P, R, border)]

