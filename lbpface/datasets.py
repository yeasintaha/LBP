"""Dataset loading and fingerprinting.

ORL
    The AT&T / Olivetti Research Laboratory database of faces: 40 subjects x 10 images,
    92x112, 8-bit, laid out as ``<root>/s<subject>/<image>.pgm``.  It is not bundled
    (its licence asks for credit to the Olivetti Research Laboratory); 
    ``python -m lbpface.datasets data/ORL`` prints the fingerprint
    that the expected results in ``expected/`` were produced with.

FERET
    Not redistributable (NIST licence). ``load_feret_manifest`` reads a CSV manifest that
    you build from your copy of FERET.

Both loaders sort files deterministically so that image order - and hence every
downstream random draw - is identical on every machine.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------- #
# fingerprinting
# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(root: str, files: Sequence[str]) -> str:
    """One SHA-256 over (relative path, file SHA-256) of every file, in sorted order."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.replace(os.sep, "/").encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(os.path.join(root, rel)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# ORL
# --------------------------------------------------------------------------- #
@dataclass
class FaceSet:
    images: np.ndarray  # (N, H, W) uint8
    subjects: np.ndarray  # (N,) int
    names: List[str]  # (N,) relative file names
    fingerprint: str
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


_ORL_RE = re.compile(r"^s(\d+)[/\\](\d+)\.pgm$")


def load_orl(root: str) -> FaceSet:
    """Load ORL from ``root`` (folder containing ``s1`` ... ``s40``).  Sorted by (subject, image)."""
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"ORL folder {root!r} not found. Expected {root}/s1/1.pgm ... {root}/s40/10.pgm "
            "(see README, 'Getting the data').")
    entries: List[Tuple[int, int, str]] = []
    for sdir in os.listdir(root):
        if not re.fullmatch(r"s\d+", sdir):
            continue
        for fn in os.listdir(os.path.join(root, sdir)):
            m = _ORL_RE.match(f"{sdir}/{fn}")
            if m:
                entries.append((int(m.group(1)), int(m.group(2)), f"{sdir}/{fn}"))
    entries.sort()
    if not entries:
        raise FileNotFoundError(f"no s<N>/<M>.pgm files found under {root!r}")
    images = []
    for _, _, rel in entries:
        with Image.open(os.path.join(root, rel)) as im:
            images.append(np.array(im.convert("L"), dtype=np.uint8))
    shapes = {im.shape for im in images}
    if len(shapes) != 1:
        raise ValueError(f"images have different sizes: {sorted(shapes)}")
    names = [e[2] for e in entries]
    return FaceSet(np.stack(images), np.array([e[0] for e in entries]), names, fingerprint(root, names))


# --------------------------------------------------------------------------- #
# FERET manifest
# --------------------------------------------------------------------------- #
FERET_COLUMNS = ["path", "subject", "sets", "left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"]
FERET_TAGS = {"fa", "fb", "fc", "dup1", "dup2", "perm", "wgal", "wprobe"}


@dataclass
class FeretManifest:
    root: str
    paths: List[str]
    subjects: np.ndarray
    left_eyes: np.ndarray  # (N, 2) x, y
    right_eyes: np.ndarray
    tags: List[frozenset]
    fingerprint: str

    def select(self, tag: str) -> np.ndarray:
        """Indices of images carrying ``tag``, in manifest order."""
        return np.array([i for i, t in enumerate(self.tags) if tag in t], dtype=np.int64)

    def load_image(self, i: int) -> np.ndarray:
        with Image.open(os.path.join(self.root, self.paths[i])) as im:
            return np.array(im.convert("L"), dtype=np.uint8)


def load_feret_manifest(csv_path: str, root: Optional[str] = None) -> FeretManifest:
    """Read a FERET manifest.
    CSV columns (header required):
    ``path``        image file relative to ``root`` (default: the CSV's folder)
    ``subject``     integer subject id
    ``sets``        ``|``-separated tags from
                    ``fa fb fc dup1 dup2``  the standard FERET gallery/probe sets,
                    ``perm``   member of the 160-subject x 4-image permutation list (``list640.srt`` of the CSU system),
                    ``wgal`` / ``wprobe``  images used as gallery / probe when learning
                               the window weights (paper: the ``subfc`` set, i.e. the
                               ``fa`` images -> ``wgal`` and ``fc`` images -> ``wprobe`` of subjects 1013-1109)
    ``left_eye_x, left_eye_y, right_eye_x, right_eye_y``
                    eye centres in pixels of the *original* image
    Rows keep the order of the file."""
    root = root if root is not None else os.path.dirname(os.path.abspath(csv_path))
    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        missing = [c for c in FERET_COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError(f"manifest is missing columns: {missing}")
        rows = list(rd)
    if not rows:
        raise ValueError("manifest is empty")
    tags = []
    for r in rows:
        t = frozenset(x for x in r["sets"].split("|") if x)
        unknown = t - FERET_TAGS
        if unknown:
            raise ValueError(f"unknown set tag(s) {sorted(unknown)} in row {r['path']!r}")
        tags.append(t)
    paths = [r["path"] for r in rows]
    le = np.array([[float(r["left_eye_x"]), float(r["left_eye_y"])] for r in rows])
    re_ = np.array([[float(r["right_eye_x"]), float(r["right_eye_y"])] for r in rows])
    subj = np.array([int(r["subject"]) for r in rows])
    return FeretManifest(root, paths, subj, le, re_, tags, fingerprint(root, paths))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m lbpface.datasets <ORL folder>", file=sys.stderr)
        return 2
    fs = load_orl(argv[0])
    print(f"images      : {fs.images.shape[0]} of size {fs.images.shape[2]}x{fs.images.shape[1]}")
    print(f"subjects    : {len(set(fs.subjects.tolist()))}")
    print(f"fingerprint : {fs.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

