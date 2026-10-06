"""Histogram dissimilarity measures (paper, Section 2, Eqs. 3-6).
All measures take the *sample* ``S`` (the probe) and the *model* ``M`` (a gallery image)
and return a number where **smaller means more similar**:

===============      =========================================       ==================================
name                 paper                                           returned value
===============      =========================================       ==================================
``chi2``             Eq. 5 / 6 (weighted)  sum_j w_j (S-M)^2/(S+M)   as is; 0/0 terms count as 0
``intersection``     Eq. 3  D = sum min(S,M)  (a *similarity*)       ``-D`` (negated; only order matters)
``log_likelihood``   Eq. 4  L = -sum S log M                         as is, with ``log(M + eps)``
===============      =========================================       ==================================

Weights are given per *region* (window) and apply to every bin of that region, as in
Eq. 6.  Regions with weight 0 are dropped from the computation entirely - this is what
shortens the paper's final feature vector to 39 windows x 59 bins = 2301 values.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import numpy as np

LOG_EPS = 1e-10  # added to the model histogram inside log(); Eq. 4 is undefined at 0


def expand_region_weights(region_weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-region weights -> per-feature weights (each weight repeated ``n_bins`` times)."""
    return np.repeat(np.asarray(region_weights, dtype=np.float64), n_bins)


def _select(features: np.ndarray, region_weights: Optional[np.ndarray], n_bins: Optional[int]):
    """Drop zero-weight regions; return (features, per-feature weights or None)."""
    if region_weights is None:
        return features, None
    if n_bins is None:
        raise ValueError("n_bins is required when region_weights are given")
    rw = np.asarray(region_weights, dtype=np.float64)
    if rw.ndim != 1 or features.shape[-1] != rw.size * n_bins:
        raise ValueError(
            f"feature length {features.shape[-1]} != {rw.size} regions x {n_bins} bins")
    if np.any(rw < 0):
        raise ValueError("weights must be non-negative")
    keep = np.repeat(rw > 0, n_bins)
    w = expand_region_weights(rw, n_bins)[keep]
    return features[..., keep], w


# --------------------------------------------------------------------------- #
# single-pair reference implementations (vectorised over leading axes)
# --------------------------------------------------------------------------- #
def chi_square(S: np.ndarray, M: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Eq. 5 (Eq. 6 if per-feature ``weights`` are given), summed over the last axis."""
    S = np.asarray(S, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    num = (S - M) ** 2
    den = S + M
    term = np.divide(num, den, out=np.zeros(np.broadcast(S, M).shape), where=den > 0)
    if weights is not None:
        term = term * weights
    return term.sum(axis=-1)


def histogram_intersection(S: np.ndarray, M: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Eq. 3: ``sum_i min(S_i, M_i)``.  A similarity: larger = more alike."""
    term = np.minimum(np.asarray(S, dtype=np.float64), np.asarray(M, dtype=np.float64))
    if weights is not None:
        term = term * weights
    return term.sum(axis=-1)


def log_likelihood(S: np.ndarray, M: np.ndarray, weights: Optional[np.ndarray] = None, eps: float = LOG_EPS) -> np.ndarray:
    """Eq. 4: ``-sum_i S_i log(M_i + eps)``."""
    term = np.asarray(S, dtype=np.float64) * np.log(np.asarray(M, dtype=np.float64) + eps)
    if weights is not None:
        term = term * weights
    return -term.sum(axis=-1)


def dissimilarity(S: np.ndarray, M: np.ndarray, metric: str = "chi2", weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Dissimilarity (smaller = more similar) between histograms, see the module table."""
    if metric == "chi2":
        return chi_square(S, M, weights)
    if metric == "intersection":
        return -histogram_intersection(S, M, weights)
    if metric == "log_likelihood":
        return log_likelihood(S, M, weights)
    raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")


METRICS: Dict[str, Callable] = {"chi2": chi_square, "intersection": histogram_intersection, "log_likelihood": log_likelihood}


# --------------------------------------------------------------------------- #
# all-pairs distance matrix
# --------------------------------------------------------------------------- #
def pairwise_distances(probes: np.ndarray, gallery: np.ndarray, metric: str = "chi2",
                       region_weights: Optional[np.ndarray] = None, n_bins: Optional[int] = None,
                       max_block_elems: int = 4_000_000) -> np.ndarray:
    """Dissimilarity matrix of shape ``(n_probes, n_gallery)``.
    ``probes[i]`` plays the role of the sample S and ``gallery[j]`` the model M (this
    matters only for the asymmetric log-likelihood).  Work is done in blocks of probes
    so that the ``(block, n_gallery, n_features)`` temporary stays around
    ``max_block_elems`` elements.
    """
    if metric not in ("chi2", "intersection", "log_likelihood"):
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    P = np.atleast_2d(np.asarray(probes, dtype=np.float64))
    G = np.atleast_2d(np.asarray(gallery, dtype=np.float64))
    if P.shape[1] != G.shape[1]:
        raise ValueError("probe and gallery feature lengths differ")
    P, w = _select(P, region_weights, n_bins)
    G, _ = _select(G, region_weights, n_bins)
    n_p, n_g, d = P.shape[0], G.shape[0], P.shape[1]
    block = max(1, max_block_elems // max(1, n_g * d))
    out = np.empty((n_p, n_g), dtype=np.float64)
    for start in range(0, n_p, block):
        s = P[start : start + block, None, :]  # (b, 1, d)
        m = G[None, :, :]  # (1, g, d)
        out[start : start + block] = dissimilarity(s, m, metric, w)
    return out

