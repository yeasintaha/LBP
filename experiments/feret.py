"""FERET protocol of the paper (Sections 3-4, Tables 2-4, Fig. 6), driven by a manifest.

Pipeline:

1. preprocess every image: eye registration -> elliptical mask -> histogram equalisation
2. LBP^{u2}_{8,2} histograms over 18x21 windows (7x7 grid on 130x150), 59 bins each
3. learn the window weights on the weight-training subset (tags ``wgal``/``wprobe``)
4. identification against the ``fa`` gallery with chi-square, unweighted and weighted-> rank-1 rates for fb / fc / dup1 / dup2 and rank curves (Table 4, Fig. 6)
5. permutation test on the ``perm`` images (one image per subject as gallery, another as
   probe, 10000 permutations) -> mean, 95 % interval and P(weighted > nonweighted)
    python -m experiments.feret --demo demo_data        # synthetic stand-in, no FERET needed

Manifest format: see ``lbpface.datasets.load_feret_manifest``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np

from lbpface import repro
from lbpface.classify import cumulative_match_curve, recognition_rate
from lbpface.datasets import load_feret_manifest
from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor
from lbpface.preprocess import OUT_SHAPE, TARGET_LEFT_EYE, TARGET_RIGHT_EYE, elliptical_mask, preprocess_face
from lbpface.stats import permutation_rates, prob_outperforms, summarise
from lbpface.weights import learn_region_weights

PROBE_SETS = ["fb", "fc", "dup1", "dup2"]
PAPER_TABLE4 = {  # rank-1 rates, Table 4
    "LBP, weighted": {"fb": 0.97, "fc": 0.79, "dup1": 0.66, "dup2": 0.64, "lower": 0.76, "mean": 0.81, "upper": 0.85},
    "LBP, nonweighted": {"fb": 0.93, "fc": 0.51, "dup1": 0.61, "dup2": 0.50, "lower": 0.71, "mean": 0.76, "upper": 0.81},
}


def run(manifest_path: str, root=None, seed: int = 0, n_perm: int = 10000, P: int = 8, R: float = 2.0,
        window=(18, 21), metric: str = "chi2", max_rank: int = 50, verbose: bool = True):
    repro.seed_everything(seed)
    man = load_feret_manifest(manifest_path, root)
    gal_idx = man.select("fa")
    if gal_idx.size == 0:
        raise ValueError("manifest has no images tagged 'fa' (the gallery)")
    desc = LBPDescriptor(P=P, R=R, window=tuple(window))
    mask = elliptical_mask(OUT_SHAPE)

    # 1-2. preprocess
    needed = sorted({i for t in (["fa", "perm", "wgal", "wprobe"] + PROBE_SETS) for i in man.select(t)})
    t0 = time.time()
    faces: Dict[int, np.ndarray] = {}
    feats: Dict[int, np.ndarray] = {}
    for i in needed:
        faces[i] = preprocess_face(man.load_image(i), tuple(man.left_eyes[i]), tuple(man.right_eyes[i]),
                                   OUT_SHAPE, TARGET_LEFT_EYE, TARGET_RIGHT_EYE, mask)
        feats[i] = desc.transform_one(faces[i])
    if verbose:
        print(f"preprocessed + described {len(needed)} images in {time.time() - t0:.1f}s "
              f"(feature length {desc.feature_length(OUT_SHAPE)})")

    def F(idx):
        return np.stack([feats[i] for i in idx])

    # 3. weights
    weights = None
    results: Dict = {"n_images": len(needed)}
    wg, wp = man.select("wgal"), man.select("wprobe")
    if wg.size and wp.size:
        weights, rates = learn_region_weights(desc, [faces[i] for i in wg], man.subjects[wg],
                                              [faces[i] for i in wp], man.subjects[wp], metric)
        rows, cols, _, _ = desc.layout(OUT_SHAPE)
        results["weights"] = weights.reshape(rows, cols)
        results["window_rates"] = np.round(rates.reshape(rows, cols), 6)
        results["n_zero_weight_windows"] = int((weights == 0).sum())
        results["effective_feature_length"] = int((weights > 0).sum() * desc.n_bins)
    elif verbose:
        print("no wgal/wprobe images in the manifest -> weighted variant skipped")

    variants = {"nonweighted": None}
    if weights is not None:
        variants["weighted"] = weights

    # 4. fixed gallery / probe sets
    results["rank1"], results["cmc"] = {}, {}
    for pset in PROBE_SETS:
        prb_idx = man.select(pset)
        if prb_idx.size == 0:
            continue
        results["rank1"][pset], results["cmc"][pset] = {}, {}
        for vname, w in variants.items():
            d = pairwise_distances(F(prb_idx), F(gal_idx), metric, w, desc.n_bins)
            results["rank1"][pset][vname] = recognition_rate(d, man.subjects[gal_idx], man.subjects[prb_idx])
            results["cmc"][pset][vname] = np.round(
                cumulative_match_curve(d, man.subjects[gal_idx], man.subjects[prb_idx], max_rank), 6)

    # 5. permutation test
    perm_idx = man.select("perm")
    if perm_idx.size:
        dists = {v: pairwise_distances(F(perm_idx), F(perm_idx), metric, w, desc.n_bins) for v, w in variants.items()}
        per = permutation_rates(dists, man.subjects[perm_idx], n_perm, seed)
        results["permutation"] = {v: summarise(r) for v, r in per.items()}
        if "weighted" in per:
            results["permutation"]["P(weighted>nonweighted)"] = prob_outperforms(per["weighted"], per["nonweighted"])

    config = {"manifest": os.path.basename(manifest_path), "seed": seed, "n_perm": n_perm,
              "descriptor": desc.describe(), "metric": metric, "out_shape": list(OUT_SHAPE),
              "target_left_eye": list(TARGET_LEFT_EYE), "target_right_eye": list(TARGET_RIGHT_EYE)}
    if verbose:
        _print_table(results)
    return config, man.fingerprint, results


def _print_table(results: Dict) -> None:
    print("\nrank-1 recognition rate            fb     fc   dup1   dup2 |  perm: lower  mean upper")
    for vname in ("weighted", "nonweighted"):
        cells: List[str] = []
        for pset in PROBE_SETS:
            v = results["rank1"].get(pset, {}).get(vname)
            cells.append(f"{v:6.2f}" if v is not None else "     -")
        perm = results.get("permutation", {}).get(vname)
        ptxt = f"{perm['lower']:.2f}  {perm['mean']:.2f}  {perm['upper']:.2f}" if perm else "-"
        if any(c.strip() != "-" for c in cells):
            print(f"  LBP, {vname:<12s}             {' '.join(cells)} |        {ptxt}")
    if "permutation" in results and "P(weighted>nonweighted)" in results["permutation"]:
        print(f"  P(weighted > nonweighted) = {results['permutation']['P(weighted>nonweighted)']:.3f}  (paper: 0.976)")
    print("  paper, Table 4 (real FERET): weighted 0.97 0.79 0.66 0.64 | 0.76 0.81 0.85 ;"
          " nonweighted 0.93 0.51 0.61 0.50 | 0.71 0.76 0.81")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="manifest CSV built from your FERET copy")
    ap.add_argument("--root", default=None, help="folder the manifest's image paths are relative to")
    ap.add_argument("--demo", metavar="DIR", help="generate a synthetic FERET-like dataset in DIR and use it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default="results/feret.json")
    ap.add_argument("--expect", default=None)
    a = ap.parse_args(argv)
    if a.demo:
        from lbpface.synthetic import make_feret_like_dataset
        a.manifest = make_feret_like_dataset(a.demo, seed=a.seed)
        print(f"synthetic dataset written to {a.demo} (NOT real FERET - numbers are not comparable)")
    if not a.manifest:
        ap.error("give --manifest or --demo")
    config, fp, results = run(a.manifest, a.root, a.seed, a.n_perm)
    doc = repro.write_results(a.out, "feret", config, fp, results)
    print(f"wrote {a.out}   digest {doc['result_digest'][:16]}...")
    if a.expect and not repro.check_against(doc, a.expect):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

