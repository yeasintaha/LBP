"""Operator / window-size / dissimilarity sweep on ORL (analogue of Table 1 and Fig. 4).

The paper runs these comparisons on FERET. This script applies the same questions to ORL so they can be reproduced

* mean recognition rate of LBP^{u2}_{8,1}, _{8,2}, _{16,2} as a function of the window
  size (k x k windows, Fig. 4),
* for every (operator, window) the probability that one of the three dissimilarity
  measures outperforms another on random gallery/probe permutations (Table 1):
  P(HI > LL), P(chi2 > HI), P(chi2 > LL).

The numbers are not comparable with the paper's FERET numbers - only the trends are.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from lbpface import repro
from lbpface.datasets import load_orl
from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor, k_by_k_window
from lbpface.stats import permutation_rates, prob_outperforms, summarise

OPERATORS = [(8, 1.0), (8, 2.0), (16, 2.0)]
METRICS = ["chi2", "intersection", "log_likelihood"]

def run(data="data/ORL", seed=0, n_perm=100, ks=(2, 3, 4, 5, 6), operators=tuple(OPERATORS), verbose=True):
    fs = load_orl(data)
    shape = fs.images.shape[1:]
    n_subj = len(np.unique(fs.subjects))
    per = fs.images.shape[0] // n_subj
    n_gal = per // 2
    results = {}
    for P, R in operators:
        for k in ks:
            win = k_by_k_window(shape, k)
            desc = LBPDescriptor(P=P, R=R, window=win)
            t0 = time.time()
            feats = desc.transform(fs.images)
            dists = {m: pairwise_distances(feats, feats, m) for m in METRICS}
            rates = permutation_rates(dists, fs.subjects, n_perm, seed, n_gal, per - n_gal)
            key = f"LBP_{P}_{R:g}|k={k}"
            results[key] = {
                "P": P, "R": R, "k": k, "window": list(win), "feature_length": int(feats.shape[1]),
                "mean_rate": {m: summarise(rates[m])["mean"] for m in METRICS},
                "P(HI>LL)": prob_outperforms(rates["intersection"], rates["log_likelihood"]),
                "P(chi2>HI)": prob_outperforms(rates["chi2"], rates["intersection"]),
                "P(chi2>LL)": prob_outperforms(rates["chi2"], rates["log_likelihood"]),
            }
            if verbose:
                r = results[key]
                print(f"LBP({P},{R:g}) k={k} win={win[0]}x{win[1]} len={r['feature_length']:6d} "
                      f"chi2={r['mean_rate']['chi2']:.3f} HI={r['mean_rate']['intersection']:.3f} "
                      f"LL={r['mean_rate']['log_likelihood']:.3f} | P(HI>LL)={r['P(HI>LL)']:.3f} "
                      f"P(chi2>HI)={r['P(chi2>HI)']:.3f} P(chi2>LL)={r['P(chi2>LL)']:.3f} "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    config = {"data": "ORL", "seed": seed, "n_perm": n_perm, "ks": list(ks),
              "operators": [list(o) for o in operators], "border": "edge",
              "n_gallery_per_subject": n_gal, "n_probe_per_subject": per - n_gal}
    return config, fs.fingerprint, results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ORL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--out", default="results/sweep_orl.json")
    ap.add_argument("--expect", default=None)
    a = ap.parse_args(argv)
    config, fp, results = run(a.data, a.seed, a.n_perm, tuple(a.ks))
    doc = repro.write_results(a.out, "sweep_orl", config, fp, results)
    print(f"wrote {a.out}   digest {doc['result_digest'][:16]}...")
    if a.expect and not repro.check_against(doc, a.expect):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

