"""Permutation-based evaluation (paper, Section 3 and 4).
The paper uses the CSU permutation tool: on every permutation one image of each subject
is randomly chosen for the gallery and another one for the probe set, and the
recognition rate is computed.  The mean rate, a 95 % interval and the probability that
one algorithm beats another follow from 10000 such permutations.

Reproducibility notes
* The 95 % interval is the empirical 2.5 / 97.5 percentile range of the permutation
  rates, as printed in Table 4 (lower / mean / upper).
"""
from __future__ import annotations
from typing import Dict, Mapping, Tuple
import numpy as np


def _subject_table(labels: np.ndarray) -> np.ndarray:
    """Image indices grouped by subject: array ``(n_subjects, images_per_subject)``."""
    labels = np.asarray(labels)
    ids, counts = np.unique(labels, return_counts=True)
    if counts.min() != counts.max():
        raise ValueError("every subject must have the same number of images "
                         f"(found counts {counts.min()}..{counts.max()})")
    order = np.argsort(labels, kind="stable")
    return order.reshape(ids.size, int(counts[0]))


def draw_splits(labels: np.ndarray, n_perm: int, seed: int, n_gallery_per_subject: int = 1,
                n_probe_per_subject: int = 1):
    """Yield ``(gallery_idx, probe_idx)`` for each permutation.
    Both are arrays of image indices, grouped subject by subject (row-major over the
    subjects in ascending label order). Gallery and probe images of one subject are
    always different images.
    """
    table = _subject_table(labels)
    n_subj, n_img = table.shape
    if n_gallery_per_subject + n_probe_per_subject > n_img:
        raise ValueError("not enough images per subject for this gallery/probe split")
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(n_perm):
        order = np.argsort(rng.random((n_subj, n_img)), axis=1, kind="stable")
        picked = np.take_along_axis(table, order, axis=1)
        gal = picked[:, :n_gallery_per_subject].reshape(-1)
        prb = picked[:, n_gallery_per_subject : n_gallery_per_subject + n_probe_per_subject].reshape(-1)
        yield gal, prb


def _rate_for_split(dist: np.ndarray, labels: np.ndarray, gal: np.ndarray, prb: np.ndarray) -> float:
    """Rank-1 rate of one split (same tie rule as ``classify.true_ranks``)."""
    sub = dist[np.ix_(prb, gal)]  # (n_probe, n_gallery)
    glabels = labels[gal]
    plabels = labels[prb]
    same = glabels[None, :] == plabels[:, None]
    # best same-subject distance; count of other-subject gallery images at least as close
    own = np.where(same, sub, np.inf).min(axis=1)
    bad = ((sub <= own[:, None]) & ~same).any(axis=1)
    return float((~bad).mean())


def permutation_rates(dists: Mapping[str, np.ndarray], labels: np.ndarray, n_perm: int = 10000,
                      seed: int = 0, n_gallery_per_subject: int = 1,
                      n_probe_per_subject: int = 1) -> Dict[str, np.ndarray]:
    """Recognition rate of each algorithm on each permutation.
    ``dists[name]`` is a full ``(N, N)`` distance matrix between *all* N images
    (row = probe role, column = gallery role); ``labels`` holds the subject of each image.
    Returns ``{name: rates of shape (n_perm,)}``.
    """
    labels = np.asarray(labels)
    for name, d in dists.items():
        if d.shape != (labels.size, labels.size):
            raise ValueError(f"distance matrix {name!r} has shape {d.shape}, expected "
                             f"({labels.size}, {labels.size})")
    out = {name: np.empty(n_perm) for name in dists}
    for i, (gal, prb) in enumerate(draw_splits(labels, n_perm, seed, n_gallery_per_subject,
                                               n_probe_per_subject)):
        for name, d in dists.items():
            out[name][i] = _rate_for_split(d, labels, gal, prb)
    return out


def summarise(rates: np.ndarray) -> Dict[str, float]:
    """Mean, sample standard deviation and empirical 95 % interval of permutation rates."""
    r = np.asarray(rates, dtype=np.float64)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"mean": float(r.mean()), "std": float(r.std(ddof=1)) if r.size > 1 else 0.0,
            "lower": float(lo), "upper": float(hi), "n_permutations": int(r.size)}


def prob_outperforms(rates_a: np.ndarray, rates_b: np.ndarray) -> float:
    """``P(A > B)``: fraction of permutations where A's rate is strictly higher."""
    a, b = np.asarray(rates_a), np.asarray(rates_b)
    if a.shape != b.shape:
        raise ValueError("rate arrays must come from the same permutations")
    return float((a > b).mean())


def pairwise_outperform_table(rates: Mapping[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
    """``P(A > B)`` for every ordered pair of distinct algorithms."""
    names = list(rates)
    return {(a, b): prob_outperforms(rates[a], rates[b]) for a in names for b in names if a != b}

