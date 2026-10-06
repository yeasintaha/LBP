import os
import numpy as np
from scipy import ndimage
from lbpface.datasets import load_orl
from lbpface.features import LBPDescriptor
from lbpface.distances import pairwise_distances
from lbpface.classify import recognition_rate

DATA = os.environ.get("ORL_DIR")
if not DATA or not os.path.exists(DATA):
    DATA = "dataset" if os.path.exists("dataset") else "data/ORL"

fs      = load_orl(DATA)
is_gal  = (np.arange(len(fs.images)) % 10) < 5     # images 1-5 = gallery
gal_raw = fs.images[is_gal]
prb_raw = fs.images[~is_gal]
gl, pl  = fs.subjects[is_gal], fs.subjects[~is_gal]

# Distortions (same as the original)
def illum_ramp(img, s):
    return np.clip(img * (1 + s * np.linspace(-1, 1, img.shape[1])), 0, 255)

def rotate(img, deg):
    return ndimage.rotate(img.astype(float), deg,
                          reshape=False, order=1, mode="nearest")

# Preprocessing helpers
def histeq(img):
    img = np.clip(np.rint(img), 0, 255).astype(np.int64)
    hist = np.bincount(img.ravel(), minlength=256)
    cdf  = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(hist)[0][0]]
    denom = img.size - cdf_min
    lut   = np.clip(np.rint((cdf - cdf_min) / max(denom, 1) * 255), 0, 255)
    return lut[img].astype(np.float64)

def tan_triggs(img, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0):
    img = np.asarray(img, dtype=np.float64)
    g = np.abs(img) ** gamma * np.sign(img)
    g = ndimage.gaussian_filter(g, sigma0) - ndimage.gaussian_filter(g, sigma1)
    mean_abs = np.mean(np.abs(g) ** alpha) ** (1.0 / alpha)
    g /= max(mean_abs, 1e-6)
    g = np.tanh(g / tau) * tau
    return g

def gauss_smooth(img, sigma=0.8):
    return ndimage.gaussian_filter(img.astype(np.float64), sigma)

DESC_ORL = LBPDescriptor.orl()

def feats(images, preproc=None):
    imgs = [preproc(im) if preproc else im for im in images]
    return DESC_ORL.transform(imgs)

def _score(G, probe_images, preproc):
    P = feats(probe_images, preproc)
    return recognition_rate(pairwise_distances(P, G, "chi2"), gl, pl)

records = []

def evaluate(name, preproc=None):
    print(f"\nEvaluating: {name}")
    G     = feats(gal_raw, preproc)
    score = lambda imgs: _score(G, imgs, preproc)

    b_score   = score(prb_raw)
    ub_score  = score([0.5 * i + 20 for i in prb_raw])
    r02_score = score([illum_ramp(i, 0.2) for i in prb_raw])
    r04_score = score([illum_ramp(i, 0.4) for i in prb_raw])
    r06_score = score([illum_ramp(i, 0.6) for i in prb_raw])
    r08_score = score([illum_ramp(i, 0.8) for i in prb_raw])
    rot5      = score([rotate(i, 5) for i in prb_raw])
    rot10     = score([rotate(i, 10) for i in prb_raw])
    rot20     = score([rotate(i, 20) for i in prb_raw])

    print(f"  baseline (no distortion)        {b_score:.4f}")
    print(f"  uniform brightness (0.5x + 20)  {ub_score:.4f}")
    print("  Lighting ramp:")
    print(f"    ramp 0.2: {r02_score:.4f} | ramp 0.4: {r04_score:.4f} | ramp 0.6: {r06_score:.4f} | ramp 0.8: {r08_score:.4f}")
    print("  Rotation:")
    print(f"    rot 5 deg: {rot5:.4f} | rot 10 deg: {rot10:.4f} | rot 20 deg: {rot20:.4f}")

    records.append({
        "Configuration": name,
        "Baseline": f"{b_score:.4f}",
        "Uniform Brightness": f"{ub_score:.4f}",
        "Lighting Ramp (s=0.2)": f"{r02_score:.4f}",
        "Lighting Ramp (s=0.4)": f"{r04_score:.4f}",
        "Lighting Ramp (s=0.6)": f"{r06_score:.4f}",
        "Lighting Ramp (s=0.8)": f"{r08_score:.4f}",
        "Rotation (5 deg)": f"{rot5:.4f}",
        "Rotation (10 deg)": f"{rot10:.4f}",
        "Rotation (20 deg)": f"{rot20:.4f}",
    })

evaluate("Baseline (Raw Images)", preproc=None)
evaluate("Global Histogram Equalization", preproc=histeq)
evaluate("Tan-Triggs Illumination Normalization", preproc=tan_triggs)

import csv
os.makedirs("results", exist_ok=True)
csv_path = os.path.join("results", "robustness_improvements.csv")

if records:
    fieldnames = list(records[0].keys())
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSaved results table to: {csv_path}")
    except PermissionError:
        fallback_path = os.path.join("results", "robustness_improvements_latest.csv")
        with open(fallback_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n[Warning] {csv_path} is currently locked by another application (like Excel).")
        print(f"Saved results table to: {fallback_path} instead.")


