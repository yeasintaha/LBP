"""Every experiment writes a JSON file with two parts:

``results``   the scientific output (rates, curves, ...).  Contains no timestamps and no
              paths, only quantities derived from ranks and counts, so it is robust to
              last-bit floating point differences between machines.
``meta``      config, seed, dataset fingerprint, code fingerprint, library versions.

``result_digest`` is the SHA-256 of the canonical JSON of ``results``.  Two runs that
reproduce each other have the same digest.
"""
from __future__ import annotations
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import sys
from typing import Any, Dict
import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's and NumPy's *global* generators and return a fresh Generator.
    The library itself never touches global random state (every function that needs
    randomness takes an explicit seed); this is only a safety net for user code.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    return np.random.Generator(np.random.PCG64(seed))


def environment_info() -> Dict[str, Any]:
    import PIL
    import scipy
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pillow": PIL.__version__,
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }


def code_fingerprint() -> str:
    """SHA-256 over the source of the ``lbpface`` package (so results can be tied to code)."""
    pkg = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in sorted(os.listdir(pkg)):
        if fn.endswith(".py"):
            h.update(fn.encode())
            with open(os.path.join(pkg, fn), "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars / arrays (and tuples) to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def result_digest(results: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(to_jsonable(results)).encode("utf-8")).hexdigest()


def write_results(path: str, name: str, config: Dict[str, Any], dataset_fingerprint: str,
                  results: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``{name, config, results, result_digest, meta}`` to ``path`` and return it."""
    results = to_jsonable(results)
    doc = {
        "experiment": name,
        "config": to_jsonable(config),
        "results": results,
        "result_digest": result_digest(results),
        "meta": {
            "dataset_fingerprint": dataset_fingerprint,
            "code_fingerprint": code_fingerprint(),
            "environment": environment_info(),
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return doc


def check_against(doc: Dict[str, Any], expected_path: str) -> bool:
    """Compare ``doc['result_digest']`` with the digest stored in an earlier results file."""
    with open(expected_path) as f:
        expected = json.load(f)
    ok = expected.get("result_digest") == doc["result_digest"]
    tag = "MATCH" if ok else "MISMATCH"
    print(f"[{tag}] result digest {doc['result_digest'][:16]}... vs expected "
          f"{str(expected.get('result_digest'))[:16]}... ({expected_path})")
    return ok

