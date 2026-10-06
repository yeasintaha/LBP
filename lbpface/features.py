"""Spatially enhanced LBP histograms (paper, Section 2, Eqs. 1-2).

The face image is cut into ``rows x cols`` equally sized rectangular windows; the
LBP^{u2} label histogram of every window is computed and the histograms are
concatenated (Eq. 2).

Window geometry follows the numbers in the paper:

* 130x150 image, 7x7 grid   -> 18x21 windows   (``130 // 7 = 18``, ``150 // 7 = 21``)
* 92x112 ORL image, 30x37 windows -> 3x3 grid  (``92 // 30 = 3``, ``112 // 37 = 3``)

Window sizes are written ``width x height`` as in the paper. When the image is not an
exact multiple of the window size the leftover pixels are dropped. The paper does not
say from which side; here they are split evenly between both sides
(``anchor="center"``) so that a left/right mirrored window is an exact mirror image,
which the weight-learning step (Section 4) relies on. ``anchor="topleft"`` drops them
at the bottom/right instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .lbp import lbp_u2, n_labels


def window_grid(shape: Tuple[int, int], grid: Optional[Tuple[int, int]] = None,
                window: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """Resolve the window layout for an image of ``shape = (height, width)``.
    Exactly one of ``grid=(rows, cols)`` or ``window=(width, height)`` must be given.
    Returns ``(rows, cols, win_h, win_w)``.
    """
    if (grid is None) == (window is None):
        raise ValueError("give exactly one of `grid` or `window`")
    h, w = shape
    if grid is not None:
        rows, cols = grid
        win_h, win_w = h // rows, w // cols
    else:
        win_w, win_h = window
        rows, cols = h // win_h, w // win_w
    if min(rows, cols, win_h, win_w) < 1:
        raise ValueError(f"window layout {grid or window} does not fit an image of shape {shape}")
    return rows, cols, win_h, win_w


def k_by_k_window(shape: Tuple[int, int], k: int) -> Tuple[int, int]:
    """Window size ``(width, height)`` for a k x k division: ``(W // k, H // k)``.
    ``k_by_k_window((150, 130), 4) == (32, 37)`` ... ``k=16 -> (8, 9)``: the sizes on
    the x-axis of Fig. 4.
    """
    h, w = shape
    return w // k, h // k


def spatial_histogram(labels: np.ndarray, n_bins: int, grid: Optional[Tuple[int, int]] = None,
                      window: Optional[Tuple[int, int]] = None, anchor: str = "center") -> np.ndarray:
    """Eq. 2: concatenated per-window label histograms of a label image.
    Returns a float64 vector of length ``rows * cols * n_bins``; window ``j`` (row-major)
    occupies ``[j * n_bins, (j + 1) * n_bins)``.  Raw counts, as in the paper.
    """
    lab = np.asarray(labels)
    rows, cols, win_h, win_w = window_grid(lab.shape, grid, window)
    if anchor == "center":
        y0, x0 = (lab.shape[0] - rows * win_h) // 2, (lab.shape[1] - cols * win_w) // 2
    elif anchor == "topleft":
        y0 = x0 = 0
    else:
        raise ValueError("anchor must be 'center' or 'topleft'")
    crop = lab[y0 : y0 + rows * win_h, x0 : x0 + cols * win_w]
    if crop.size and (crop.min() < 0 or crop.max() >= n_bins):
        raise ValueError("label outside [0, n_bins)")
    m = rows * cols
    blocks = crop.reshape(rows, win_h, cols, win_w).transpose(0, 2, 1, 3).reshape(m, win_h * win_w)
    flat = blocks.astype(np.int64) + (np.arange(m, dtype=np.int64) * n_bins)[:, None]
    return np.bincount(flat.ravel(), minlength=m * n_bins).astype(np.float64)


@dataclass(frozen=True)
class LBPDescriptor:
    """Face descriptor: LBP^{u2}_{P,R} histograms over a grid of windows.
    Parameters
    ----------
    P, R : neighbourhood (paper: (8,2) for FERET, (16,2) for ORL).
    grid : ``(rows, cols)`` of windows, *or*
    window : ``(width, height)`` of one window in pixels (exactly one of the two).
    border : how sampling points outside the image are read (see ``lbp.py``).
    anchor : where leftover pixels are dropped (see module docstring).
    """

    P: int = 8
    R: float = 2.0
    grid: Optional[Tuple[int, int]] = None
    window: Optional[Tuple[int, int]] = None
    border: str = "edge"
    anchor: str = "center"

    def __post_init__(self):
        if (self.grid is None) == (self.window is None):
            raise ValueError("give exactly one of `grid` or `window`")

    @classmethod
    def feret(cls) -> "LBPDescriptor":
        """Setting used for the FERET results: LBP^{u2}_{8,2}, 18x21 windows (7x7 on 130x150)."""
        return cls(P=8, R=2.0, window=(18, 21))

    @classmethod
    def orl(cls) -> "LBPDescriptor":
        """Setting used for the ORL experiment: LBP^{u2}_{16,2}, 30x37 windows."""
        return cls(P=16, R=2.0, window=(30, 37))

    @property
    def n_bins(self) -> int:
        return n_labels(self.P)

    def layout(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        return window_grid(shape, self.grid, self.window)

    def n_regions(self, shape: Tuple[int, int]) -> int:
        rows, cols, _, _ = self.layout(shape)
        return rows * cols

    def feature_length(self, shape: Tuple[int, int]) -> int:
        return self.n_regions(shape) * self.n_bins

    def label_image(self, image: np.ndarray) -> np.ndarray:
        return lbp_u2(image, self.P, self.R, self.border)

    def transform_one(self, image: np.ndarray) -> np.ndarray:
        return spatial_histogram(self.label_image(image), self.n_bins, self.grid, self.window, self.anchor)

    def transform(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Stack of descriptors, shape ``(n_images, feature_length)``."""
        feats = [self.transform_one(im) for im in images]
        if not feats:
            raise ValueError("no images")
        return np.stack(feats, axis=0)

    def describe(self) -> dict:
        return {"P": self.P, "R": self.R, "grid": self.grid, "window": self.window,
                "border": self.border, "anchor": self.anchor}


def uniform_fraction(images: Sequence[np.ndarray], P: int, R: float, border: str = "edge") -> float:
    """Fraction of pixels whose LBP_{P,R} pattern is uniform.
    The paper reports 79.3 % for LBP_{16,2} on preprocessed FERET images (Section 3).
    """
    last = n_labels(P) - 1
    total = uniform = 0
    for im in images:
        lab = lbp_u2(im, P, R, border)
        total += lab.size
        uniform += int((lab != last).sum())
    return uniform / total

