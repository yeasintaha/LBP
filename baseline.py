import numpy as np
from scipy import ndimage
from lbpface.datasets import load_orl
from lbpface.features import LBPDescriptor
from lbpface.distances import pairwise_distances
from lbpface.classify import recognition_rate
import os

DATA = os.environ.get("ORL_DIR", "data/ORL")
fs = load_orl(DATA)                                 # <-- this line was missing
desc = LBPDescriptor.orl()                          # LBP(16,2), 30x37 windows
is_gallery = (np.arange(len(fs.images)) % 10) < 5   # images 1-5 of each subject
gal, prb = fs.images[is_gallery], fs.images[~is_gallery]
gl, pl = fs.subjects[is_gallery], fs.subjects[~is_gallery]
G = desc.transform(gal)

def score(probe_images):
    P = desc.transform(probe_images)
    return recognition_rate(pairwise_distances(P, G, "chi2"), gl, pl)

def illum_ramp(img, s):          # darker on the left, brighter on the right
    return np.clip(img * (1 + s * np.linspace(-1, 1, img.shape[1])), 0, 255)

def rotate(img, deg):
    return ndimage.rotate(img.astype(float), deg, reshape=False, order=1, mode="nearest")

print("baseline           ", score(prb))
print("uniform brightness ", score([0.5 * i + 20 for i in prb]))
for s in [0.2, 0.4, 0.6, 0.8]: print("lighting ramp", s, score([illum_ramp(i, s) for i in prb]))
for d in [5, 10, 20]:          print("rotate", d, "deg", score([rotate(i, d) for i in prb]))