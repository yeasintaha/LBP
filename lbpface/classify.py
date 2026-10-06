"""Nearest-neighbour identification and rank curves (paper, Sections 2-3).
A *distance matrix* has one row per probe image and one column per gallery image.
Identification is done the way the FERET protocol does it: the gallery may hold several
images, but rank is counted over **subjects**, i.e. a subject is ranked by its closest
gallery image.

Ties: a probe only counts as correct at rank ``r`` if fewer than ``r`` *other* subjects
are closer **or equally close**.  Ties therefore never flatter the result (a degenerate
all-equal distance matrix scores ~0, not 100 %).
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def subject_min_distances(dist: np.ndarray, gallery_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse gallery columns to one column per subject (the subject's closest image).
    Returns ``(subject_ids, dist_by_subject)`` with ``subject_ids`` sorted ascending.
    """
    dist = np.asarray(dist)
    labels = np.asarray(gallery_labels)
    if dist.ndim != 2 or dist.shape[1] != labels.size:
        raise ValueError("dist must be (n_probes, n_gallery) and match gallery_labels")
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    ids, starts = np.unique(sorted_labels, return_index=True)
    return ids, np.minimum.reduceat(dist[:, order], starts, axis=1)


def true_ranks(dist: np.ndarray, gallery_labels: np.ndarray, probe_labels: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of the correct subject for every probe"""
    ids, by_subject = subject_min_distances(dist, gallery_labels)
    probe_labels = np.asarray(probe_labels)
    col = np.searchsorted(ids, probe_labels)
    col_clipped = np.clip(col, 0, ids.size - 1)
    if np.any(ids[col_clipped] != probe_labels):
        missing = np.unique(probe_labels[ids[col_clipped] != probe_labels])
        raise ValueError(f"probe subjects missing from the gallery: {missing[:10].tolist()}")
    own = by_subject[np.arange(by_subject.shape[0]), col_clipped]
    # number of *other* subjects at least as close as the true one
    not_worse = (by_subject <= own[:, None]).sum(axis=1) - 1
    return not_worse + 1


def cumulative_match_curve(dist: np.ndarray, gallery_labels: np.ndarray, probe_labels: np.ndarray,
                           max_rank: int = 50) -> np.ndarray:
    """Cumulative match scores for ranks ``1 .. max_rank`` (the curves of Fig. 6)."""
    ranks = true_ranks(dist, gallery_labels, probe_labels)
    return np.array([(ranks <= r).mean() for r in range(1, max_rank + 1)])


def recognition_rate(dist: np.ndarray, gallery_labels: np.ndarray, probe_labels: np.ndarray) -> float:
    """Rank-1 identification rate (the numbers of Table 4)."""
    return float((true_ranks(dist, gallery_labels, probe_labels) == 1).mean())


def nearest_neighbour(dist: np.ndarray, gallery_labels: np.ndarray) -> np.ndarray:
    """Predicted subject of every probe: the label of its closest gallery image."""
    return np.asarray(gallery_labels)[np.argmin(dist, axis=1)]

