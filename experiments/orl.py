"""ORL experiment of the paper (Section 4, last paragraph).
    "Randomly selecting 5 images for the gallery set and the other 5 for the probe set,
     the preliminary experiments result in 0.98 of average recognition rate and 0.012 of
     standard deviation of 100 random permutations using LBP^{u2}_{16,2}, a window size
     of 30*37 and chi-square as a dissimilarity measure.  Window weights were not used."
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from lbpface import repro
from lbpface.datasets import load_orl
from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor, uniform_fraction
from lbpface.stats import permutation_rates, summarise

PAPER = {"mean": 0.98, "std": 0.012}

def run(data: str = "data/ORL", seed: int = 0, n_perm: int = 100, P: int = 16, R: float = 2.0,
        window=(30, 37), grid=None, metric: str = "chi2", border: str = "edge",
        n_gallery: int = 5, verbose: bool = True):
    """Run the ORL protocol; returns ``(config, dataset_fingerprint, results)``."""
    repro.seed_everything(seed)
    fs = load_orl(data)
    n_probe = fs.images.shape[0] // len(np.unique(fs.subjects)) - n_gallery
    desc = LBPDescriptor(P=P, R=R, window=None if grid else tuple(window), grid=grid, border=border)
    t0 = time.time()
    feats = desc.transform(fs.images)
    dist = pairwise_distances(feats, feats, metric)
    t1 = time.time()
    rates = permutation_rates({metric: dist}, fs.subjects, n_perm, seed, n_gallery, n_probe)[metric]
    summary = summarise(rates)
    results = {
        "feature_length": int(feats.shape[1]),
        "n_images": int(feats.shape[0]),
        "uniform_pixel_fraction": round(uniform_fraction(fs.images, P, R, border), 6),
        "summary": summary,
        "rates": rates,
    }
    config = {"data": "ORL", "seed": seed, "n_perm": n_perm, "descriptor": desc.describe(),
              "metric": metric, "n_gallery_per_subject": n_gallery, "n_probe_per_subject": n_probe,
              "preprocessing": "none", "weights": "none"}
    if verbose:
        print(f"ORL: {feats.shape[0]} images, feature length {feats.shape[1]}, "
              f"features+distances {t1 - t0:.1f}s")
        print(f"mean recognition rate {summary['mean']:.4f}  std {summary['std']:.4f}  "
              f"95% [{summary['lower']:.3f}, {summary['upper']:.3f}]  ({n_perm} permutations)")
        print(f"paper reports         {PAPER['mean']:.4f}  std {PAPER['std']:.4f}")
    return config, fs.fingerprint, results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ORL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--R", type=float, default=2.0)
    ap.add_argument("--window", type=int, nargs=2, default=[30, 37], metavar=("W", "H"))
    ap.add_argument("--metric", default="chi2", choices=["chi2", "intersection", "log_likelihood"])
    ap.add_argument("--border", default="edge", choices=["edge", "reflect", "constant"])
    ap.add_argument("--out", default="results/orl.json")
    ap.add_argument("--expect", default=None, help="earlier results JSON whose digest must match")
    a = ap.parse_args(argv)
    config, fp, results = run(a.data, a.seed, a.n_perm, a.P, a.R, tuple(a.window), None, a.metric, a.border)
    doc = repro.write_results(a.out, "orl", config, fp, results)
    print(f"wrote {a.out}   digest {doc['result_digest'][:16]}...")
    if a.expect and not repro.check_against(doc, a.expect):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

