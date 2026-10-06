"""End-to-end tests on synthetic faces (no downloads needed) + reproducibility."""
import csv
import json
import os

import numpy as np
import pytest

from experiments import feret
from lbpface import repro
from lbpface.datasets import load_feret_manifest, load_orl
from lbpface.synthetic import make_feret_like_dataset


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    return str(make_feret_like_dataset(str(d), n_subjects=30, seed=0))


def test_feret_pipeline_runs_and_is_sane(synth):
    config, fp, res = feret.run(synth, n_perm=50, verbose=False)
    r1 = res["rank1"]
    assert set(r1) == {"fb", "fc", "dup1", "dup2"}
    assert r1["fb"]["nonweighted"] >= 0.9 and r1["fb"]["weighted"] >= 0.9
    assert r1["dup1"]["nonweighted"] >= 0.7
    assert r1["fc"]["nonweighted"] < r1["fb"]["nonweighted"]  # strong lighting change is harder
    w = np.array(res["weights"])
    assert w.shape == (7, 7) and set(np.unique(w)) <= {0.0, 1.0, 2.0, 4.0}
    np.testing.assert_array_equal(w, w[:, ::-1])
    assert res["effective_feature_length"] == int((w > 0).sum()) * 59
    for v in ("weighted", "nonweighted"):
        cmc = np.array(res["cmc"]["fb"][v])
        assert cmc.shape == (50,) and np.all(np.diff(cmc) >= 0) and cmc[-1] == 1.0
    perm = res["permutation"]
    assert perm["nonweighted"]["n_permutations"] == 50
    assert perm["nonweighted"]["lower"] <= perm["nonweighted"]["mean"] <= perm["nonweighted"]["upper"]
    assert 0.0 <= perm["P(weighted>nonweighted)"] <= 1.0


def test_two_runs_with_the_same_seed_are_identical(synth):
    *_, a = feret.run(synth, seed=3, n_perm=40, verbose=False)
    *_, b = feret.run(synth, seed=3, n_perm=40, verbose=False)
    assert repro.result_digest(a) == repro.result_digest(b)
    *_, c = feret.run(synth, seed=4, n_perm=40, verbose=False)
    assert repro.result_digest(a) != repro.result_digest(c)  # the permutation seed matters


def test_registration_matters_wrong_eye_positions_hurt(synth, tmp_path):
    good = feret.run(synth, n_perm=10, verbose=False)[2]["rank1"]
    # jitter all eye coordinates by up to +-12 px and re-run on the same images
    rng = np.random.default_rng(0)
    with open(synth, newline="") as f:
        rows = list(csv.DictReader(f))
    bad_manifest = tmp_path / "bad.csv"
    with open(bad_manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            for k in ("left_eye_x", "left_eye_y", "right_eye_x", "right_eye_y"):
                r[k] = float(r[k]) + rng.uniform(-12, 12)
            w.writerow(r)
    bad = feret.run(str(bad_manifest), root=os.path.dirname(synth), n_perm=10, verbose=False)[2]["rank1"]
    assert bad["dup1"]["nonweighted"] < good["dup1"]["nonweighted"]


def test_manifest_validation(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,subject\nx.png,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_feret_manifest(str(p))
    p.write_text("path,subject,sets,left_eye_x,left_eye_y,right_eye_x,right_eye_y\n")
    with pytest.raises(ValueError, match="empty"):
        load_feret_manifest(str(p))


def test_orl_loader_is_order_and_content_deterministic(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    for s in (2, 10, 1):  # created out of order on purpose; also tests numeric (not lexical) sort
        (tmp_path / f"s{s}").mkdir()
        for i in (3, 1, 10, 2):
            Image.fromarray(rng.integers(0, 256, (12, 10), dtype=np.uint8)).save(tmp_path / f"s{s}" / f"{i}.pgm")
    fs = load_orl(str(tmp_path))
    assert fs.subjects.tolist() == [1] * 4 + [2] * 4 + [10] * 4
    assert fs.names[:4] == ["s1/1.pgm", "s1/2.pgm", "s1/3.pgm", "s1/10.pgm"]
    assert load_orl(str(tmp_path)).fingerprint == fs.fingerprint
    # changing one pixel changes the fingerprint
    im = np.array(Image.open(tmp_path / "s1" / "1.pgm"))
    im[0, 0] ^= 1
    Image.fromarray(im).save(tmp_path / "s1" / "1.pgm")
    assert load_orl(str(tmp_path)).fingerprint != fs.fingerprint
    with pytest.raises(FileNotFoundError):
        load_orl(str(tmp_path / "nope"))


def test_results_file_roundtrip_and_digest_ignores_metadata(tmp_path):
    results = {"a": np.float64(0.5), "b": np.arange(3), "c": {"d": (1, 2)}}
    p1, p2 = tmp_path / "r1.json", tmp_path / "r2.json"
    d1 = repro.write_results(str(p1), "t", {"x": 1}, "fp", results)
    d2 = repro.write_results(str(p2), "t", {"x": 1}, "fp", results)
    assert d1["result_digest"] == d2["result_digest"]
    doc = json.loads(p1.read_text())
    assert doc["results"] == {"a": 0.5, "b": [0, 1, 2], "c": {"d": [1, 2]}}
    assert "timestamp_utc" in doc["meta"] and "timestamp_utc" not in json.dumps(doc["results"])
    assert repro.check_against(d1, str(p2))
    d3 = repro.write_results(str(tmp_path / "r3.json"), "t", {"x": 1}, "fp", {"a": 0.6})
    assert not repro.check_against(d3, str(p1))
