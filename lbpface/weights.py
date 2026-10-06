"""Window weights for the weighted chi-square measure (paper, Section 4, Eq. 6, Fig. 5b).

Procedure described in the paper:

1. Classify a training set using **one window at a time** (unweighted chi-square) and
   record the recognition rate of every window.
2. Average the rates of corresponding windows in the left and right half of the face
   (i.e. symmetrise the 7x7 map left <-> right).
3. Windows whose rate lies below the 0.2 percentile get weight 0; windows above the
   0.8 percentile get weight 2.0; above the 0.9 percentile weight 4.0; all others 1.0.

The paper does not say how percentiles are interpolated or how ties at a threshold are
treated.  Here: ``numpy.percentile`` with linear interpolation, and strict comparisons
(``<`` for the zero weight, ``>`` for the 2.0 and 4.0 weights), so a window exactly on a
threshold keeps the weight of the lower class.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from .classify import recognition_rate
from .distances import pairwise_distances


def window_recognition_rates(gallery_feats: np.ndarray, gallery_labels: np.ndarray,
                             probe_feats: np.ndarray, probe_labels: np.ndarray,
                             n_regions: int, n_bins: int, metric: str = "chi2") -> np.ndarray:
    """Recognition rate obtained with each window alone, shape ``(n_regions,)``."""
    if gallery_feats.shape[1] != n_regions * n_bins:
        raise ValueError("feature length != n_regions * n_bins")
    rates = np.empty(n_regions)
    for j in range(n_regions):
        sl = slice(j * n_bins, (j + 1) * n_bins)
        d = pairwise_distances(probe_feats[:, sl], gallery_feats[:, sl], metric)
        rates[j] = recognition_rate(d, gallery_labels, probe_labels)
    return rates


def symmetrise(rate_grid: np.ndarray) -> np.ndarray:
    """Average each window's rate with that of its left/right mirror window."""
    return 0.5 * (rate_grid + rate_grid[:, ::-1])


def weights_from_rates(rate_grid: np.ndarray, zero_pct: float = 20.0, two_pct: float = 80.0,
                       four_pct: float = 90.0, sym: bool = True) -> np.ndarray:
    """Rates on the ``(rows, cols)`` window grid -> weights in ``{0, 1, 2, 4}`` (same shape)."""
    rates = np.asarray(rate_grid, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError("rate_grid must be 2-D (rows, cols)")
    if sym:
        rates = symmetrise(rates)
    lo, hi2, hi4 = np.percentile(rates, [zero_pct, two_pct, four_pct])
    w = np.ones_like(rates)
    w[rates > hi2] = 2.0
    w[rates > hi4] = 4.0
    w[rates < lo] = 0.0
    return w


def learn_region_weights(descriptor, gallery_images, gallery_labels, probe_images, probe_labels,
                         metric: str = "chi2") -> Tuple[np.ndarray, np.ndarray]:
    """Run the whole procedure on a training set.
    ``gallery_*`` / ``probe_*`` are the training gallery and probe images (in the paper:
    the ``subfc`` set - the ``fa`` images are the gallery and the ``fc`` images the
    probes).  Returns ``(weights, rates)`` flattened row-major over the window grid.
    """
    shape = np.asarray(gallery_images[0]).shape
    rows, cols, _, _ = descriptor.layout(shape)
    g = descriptor.transform(gallery_images)
    p = descriptor.transform(probe_images)
    rates = window_recognition_rates(g, np.asarray(gallery_labels), p, np.asarray(probe_labels),
                                     rows * cols, descriptor.n_bins, metric)
    weights = weights_from_rates(rates.reshape(rows, cols)).reshape(-1)
    return weights, rates

