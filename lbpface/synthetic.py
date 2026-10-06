"""Procedurally generated "faces" for tests and for exercising the FERET pipeline.

FERET cannot be redistributed, so the FERET code path (eye registration, elliptical mask,
histogram equalisation, gallery/probe sets, weight learning, permutation test) is
exercised on these synthetic faces instead.  

They are not realistic - they only need identity-specific texture + layout, plus nuisance variation in pose (similarity
transform), lighting, expression and noise, with known ground-truth eye positions.

Everything is driven by an explicit seed and uses only ``Generator.random`` /
``Generator.standard_normal`` so the images are identical on every machine with the same NumPy stream.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

CANON_SHAPE = (150, 130)  # (h, w) of the canonical face frame
RAW_SHAPE = (220, 200)  # (h, w) of the simulated "photograph"


def _blob(yy, xx, cy, cx, sy, sx, amp):
    return amp * np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))


def _subject_params(rng: np.random.Generator) -> Dict:
    tex = ndimage.gaussian_filter(rng.standard_normal(CANON_SHAPE), sigma=2.0)
    tex /= tex.std()
    return {
        "eye_dx": 22 + 3 * rng.random(),
        "eye_y": 60 + 3 * rng.random(),
        "eye_size": 4.0 + 1.0 * rng.random(),
        "brow": 12 + 6 * rng.random(),
        "nose_len": 16 + 5 * rng.random(),
        "mouth_y": 108 + 5 * rng.random(),
        "mouth_w": 14 + 5 * rng.random(),
        "skin": 120 + 40 * rng.random(),
        "texture": tex,
        "tex_amp": 7 + 3 * rng.random(),
    }


def _render_canonical(p: Dict, rng: np.random.Generator, smile: float, light: float) -> Tuple[np.ndarray, Tuple, Tuple]:
    h, w = CANON_SHAPE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx = w / 2.0
    img = np.full((h, w), p["skin"])
    img += p["tex_amp"] * p["texture"]
    lx, rx, ey = cx - p["eye_dx"], cx + p["eye_dx"], p["eye_y"]
    for ex in (lx, rx):
        img += _blob(yy, xx, ey, ex, p["eye_size"] * 0.8, p["eye_size"] * 1.3, -70)  # eye
        img += _blob(yy, xx, ey - 9, ex, 2.5, p["eye_size"] * 2.0, -p["brow"] * 2)  # brow
    img += _blob(yy, xx, ey + p["nose_len"], cx, p["nose_len"] * 0.6, 3.5, -25)  # nose
    mouth_y = p["mouth_y"] - 3 * smile
    img += _blob(yy, xx, mouth_y, cx, 2.5 + 2 * smile, p["mouth_w"] * (1 + 0.25 * smile), -60)  # mouth
    img += light * (xx - cx) / w  # lighting gradient
    return img, (lx, ey), (rx, ey)


def make_raw_face(params: Dict, rng: np.random.Generator, strength: float = 1.0):
    """One simulated photograph of a subject.
    Returns ``(image uint8 of RAW_SHAPE, left_eye (x, y), right_eye (x, y))`` where the
    eye positions are the true positions in the returned photograph.  ``strength``
    scales the lighting / expression / pose nuisance.
    """
    smile = rng.random() * strength
    light = (rng.random() - 0.5) * 70 * strength
    canon, le, re_ = _render_canonical(params, rng, smile, light)
    ang = np.deg2rad((rng.random() - 0.5) * 12 * strength)
    scale = 1.0 + (rng.random() - 0.5) * 0.2 * strength
    shift = (rng.random(2) - 0.5) * 20 * strength  # (dx, dy)
    # forward map canonical -> raw:  z_raw = s e^{i ang} (z - z0) + z_c + shift
    z0 = complex(CANON_SHAPE[1] / 2.0, CANON_SHAPE[0] / 2.0)
    zc = complex(RAW_SHAPE[1] / 2.0 + shift[0], RAW_SHAPE[0] / 2.0 + shift[1])
    a = scale * np.exp(1j * ang)
    fwd = lambda z: a * (z - z0) + zc  # noqa: E731
    c = 1.0 / a
    off = z0 - c * zc  # z_canon = c (z_raw - zc) + z0  ->  c z_raw + (z0 - c zc)
    m_xy = np.array([[c.real, -c.imag], [c.imag, c.real]])
    raw = ndimage.affine_transform(canon, m_xy[::-1, ::-1], offset=np.array([off.imag, off.real]),
                                   output_shape=RAW_SHAPE, order=1, mode="nearest")
    raw += rng.standard_normal(RAW_SHAPE) * 6.0 * (0.5 + strength)
    zl, zr = fwd(complex(*le)), fwd(complex(*re_))
    img = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
    return img, (zl.real, zl.imag), (zr.real, zr.imag)


def make_feret_like_dataset(out_dir: str, n_subjects: int = 24, seed: int = 0) -> str:
    """Write a small FERET-like dataset and return the path of its manifest CSV.
    Per subject: one ``fa`` (gallery) image and probes ``fb`` (mild variation), ``fc``
    (strong lighting change), ``dup1`` and ``dup2`` (stronger variation). ``dup2`` is also
    tagged ``dup1`` as in FERET. The first third of the subjects additionally carry the
    ``wgal`` / ``wprobe`` tags (weight-learning subset). The ``perm`` tag marks 4 images
    per subject (fa, fb, dup1, dup2).
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    rows: List[List] = []
    kinds = [("fa", 0.3, "fa|perm"), ("fb", 0.5, "fb|perm"), ("fc", 1.6, "fc"),
             ("dup1", 0.9, "dup1|perm"), ("dup2", 1.1, "dup1|dup2|perm")]
    for s in range(n_subjects):
        subject = 1000 + s
        params = _subject_params(rng)
        for kind, strength, tags in kinds:
            img, le, re_ = make_raw_face(params, rng, strength)
            fn = f"{subject:05d}_{kind}.png"
            Image.fromarray(img).save(os.path.join(out_dir, fn))
            t = tags
            if s < n_subjects // 3:
                if kind == "fa":
                    t += "|wgal"
                if kind == "fc":
                    t += "|wprobe"
            rows.append([fn, subject, t, f"{le[0]:.3f}", f"{le[1]:.3f}", f"{re_[0]:.3f}", f"{re_[1]:.3f}"])
    manifest = os.path.join(out_dir, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "subject", "sets", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"])
        w.writerows(rows)
    return manifest

