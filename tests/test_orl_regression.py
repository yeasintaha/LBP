"""Regression tests on the real ORL data."""
import json
import os

import numpy as np
import pytest

from experiments import orl
from lbpface import repro
from lbpface.datasets import load_orl

EXPECTED = os.path.join(os.path.dirname(os.path.dirname(__file__)), "expected", "orl_seed0.json")


def test_orl_has_the_documented_shape(orl_dir):
    fs = load_orl(orl_dir)
    assert fs.images.shape == (400, 112, 92) and len(np.unique(fs.subjects)) == 40
    assert np.all(np.bincount(fs.subjects)[1:] == 10)


def test_paper_setting_reaches_the_reported_accuracy(orl_dir):
    # Paper: mean 0.98, std 0.012 over 100 permutations.  A 20-permutation run must already be close.
    _, _, res = orl.run(orl_dir, seed=0, n_perm=20, verbose=False)
    assert res["feature_length"] == 9 * 243
    assert 0.97 <= res["summary"]["mean"] <= 0.995


def test_bit_exact_reproduction_of_the_stored_run(orl_dir):
    with open(EXPECTED) as f:
        expected = json.load(f)
    config, fp, res = orl.run(orl_dir, seed=0, n_perm=100, verbose=False)
    if fp != expected["meta"]["dataset_fingerprint"]:
        pytest.skip("ORL copy differs from the one the stored results were made with "
                    f"(fingerprint {fp[:12]}... vs {expected['meta']['dataset_fingerprint'][:12]}...)")
    assert repro.result_digest(res) == expected["result_digest"]
    assert res["summary"]["mean"] == expected["results"]["summary"]["mean"]

