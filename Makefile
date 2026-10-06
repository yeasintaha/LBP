PY ?= python3

.PHONY: install test orl sweep feret-demo reproduce fingerprint

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

# Paper's ORL experiment (LBP^u2_{16,2}, 30x37 windows, chi^2, 100 permutations)
orl:
	$(PY) -m experiments.orl --data data/ORL --seed 0 --out results/orl_seed0.json

# Table 1 / Fig. 4 analogues on ORL (a few minutes)
sweep:
	$(PY) -m experiments.sweep --data data/ORL --seed 0 --out results/sweep_orl.json

# Whole FERET pipeline on synthetic faces (no FERET needed)
feret-demo:
	$(PY) -m experiments.feret --demo results/demo_data --out results/feret_demo.json --n-perm 1000

# Re-run the stored experiment and fail unless the result digest is identical
reproduce:
	$(PY) -m experiments.orl --data data/ORL --seed 0 --out results/orl_seed0.json --expect expected/orl_seed0.json

fingerprint:
	$(PY) -m lbpface.datasets data/ORL
